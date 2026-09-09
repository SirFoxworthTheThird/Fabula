"""One playable scene, wired up.

The engine knows nothing about frontends, but every client needs the
same handful of operations — say something, move, let time pass, look
around — and every client needs them expressed as *the user character's
POV*, never as the raw log. That shared surface lives here, so the CLI
and the HTTP service are both thin layers and a POV rule fixed once is
fixed for both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from fabula.chronology import derive_skip_minutes
from fabula.concurrency import DEFAULT_WORKERS
from fabula.db import EventStore
from fabula.director import Director
from fabula.llm import LLMClient, get_default_llm
from fabula.memory import co_present
from fabula.loader import Scene, load_pressures, load_scenario, player_name
from fabula.player import Player
from fabula.models import Character, Event, ProjectedEvent
from fabula.narrator import Narrator
from fabula.persistence import begin_scene
from fabula.pressures import has_ended, next_scene, scene_state
from fabula.discovery import discover
from fabula.discovery import restore as discovery_restore
from fabula.world import OFFSTAGE, Item, World, find_room


class Session:
    def __init__(
        self,
        world: World,
        characters: dict[str, Character],
        scene: Scene,
        store: EventStore,
        director: Director,
        user_character: Character,
    ):
        self.world = world
        self.characters = characters
        self.scene = scene
        self.store = store
        self.director = director
        self.user_character = user_character
        # Where everyone's trust stood when this scene opened. Never
        # returned by any method here — `fabula.reveal` reads it to show
        # what the evening did to people, which is the one place the
        # engine is allowed to step outside a point of view.
        self.trust_at_open: dict[str, dict[str, float]] = {}
        # The last thing the player did, kept so it can be played again.
        # A closure rather than a command string: the engine has no verbs,
        # and re-parsing text here would give the clients a second, worse
        # dispatcher to disagree with.
        self._last_take: Callable[[], list[ProjectedEvent]] | None = None
        # The sequence number the open turn started at. A client showing a
        # transcript needs it on a retake: everything from here on was
        # discarded and must be dropped before the new take is rendered.
        # Where this world was loaded from, and what it generates with —
        # a story has to be able to open its own next scene.
        self.world_dir: Path | None = None
        self.llm: LLMClient | None = None
        self.turn_started_at: int = 0
        # Whether the scene had already reached its end when the open turn
        # began, and how many turns have been played. A client reporting
        # "this ended it" has to compare against the state the turn
        # started from — which a retake rewinds to, so it cannot be
        # measured before the command runs.
        self.ended_at_turn_start: bool = False
        # How far the story got, and how many takes were played. They
        # differ on a retake: the turn is played again, so the story is
        # no further on, but a take did happen and a caller asking "did
        # that command play a turn" must still be told yes.
        self.turns_played: int = 0
        self.takes_played: int = 0
        self._turns_before_take: int = 0

    @classmethod
    def open(
        cls,
        world_dir: Path,
        scene_name: str,
        db_path: str = ":memory:",
        llm: LLMClient | None = None,
        interpret_beliefs: bool = True,
        store: EventStore | None = None,
        workers: int = DEFAULT_WORKERS,
        player: Player | None = None,
        direct_beats: bool = True,
    ) -> Session:
        world, characters, scene = load_scenario(world_dir, scene_name, player)
        called = None
        if player and player.called:
            authored = player_name(world_dir)
            if authored:
                called = (authored, player.called)
        pressures = load_pressures(world_dir, called)

        # Only the cast is in the scene. Everything downstream — bidding,
        # presence, projection, the reveal — walks this dict, so a
        # character left in the world but out of the cast was taking
        # turns in every scene the world had. `cast` was decorative until
        # this line: a list you write and the engine ignores is worse
        # than no list at all.
        missing = [cid for cid in scene.cast + scene.may_arrive if cid not in characters]
        if missing:
            raise ValueError(f"scene {scene.id} names unknown character(s): {missing}")
        overlap = sorted(set(scene.cast) & set(scene.may_arrive))
        if overlap:
            raise ValueError(f"scene {scene.id} both casts and awaits: {overlap}")

        # Whoever may still walk in waits off-stage, which is not a room:
        # unreachable in both directions, so until their arrival they
        # perceive nothing and nobody perceives them — through the
        # ordinary perception path, with no special case in it.
        waiting = {
            cid: characters[cid].model_copy(update={"location_id": OFFSTAGE})
            for cid in scene.may_arrive
        }
        characters = {cid: characters[cid] for cid in scene.cast}
        # One store for a whole story: every scene after the first joins
        # the one before it, which is what carries beliefs, trust and
        # closed goals across the seam.
        store = store or EventStore(db_path)
        llm = llm or get_default_llm()

        user_character = next(
            (
                characters[cid]
                for cid in scene.cast
                if cid in characters and characters[cid].is_user
            ),
            None,
        )
        if user_character is None:
            raise ValueError("scene has no user-controlled character (is_user: true) in its cast")

        # The narrator is told whose character it must never play.
        narrator = Narrator(
            llm, protagonist=user_character.name, protagonist_id=user_character.id
        )
        director = Director(
            store, world, characters, scene, narrator, llm, pressures,
            interpret_beliefs=interpret_beliefs,
            waiting=waiting,
            workers=workers,
            direct_beats=direct_beats,
        )

        # Characters are durable: with a real db path they arrive carrying
        # what they already believe, aged by the time between scenes.
        # Anywhere found on an earlier visit is on the map again before
        # anybody stands in it.
        discovery_restore(world, store)
        begin_scene(store, characters)

        session = cls(world, characters, scene, store, director, user_character)
        session.world_dir = world_dir
        session.llm = llm
        # A resumed story carries on counting rather than starting again.
        saved = store.get_story()
        if saved:
            session.turns_played = saved["turns"]
        # After seeding, not from the YAML: a character on their second
        # evening arrives carrying what the first one did to them, and
        # "was" has to mean when this scene started.
        session.trust_at_open = {
            character_id: {
                toward: relationship.trust
                for toward, relationship in store.get_relationships(character_id).items()
            }
            for character_id in characters
        }
        # The scene says its first line before the player has to. A story
        # that opens on a bare prompt is a text box: the room has a name
        # and nothing in it until somebody thinks to type /look.
        began = not store.any_events()
        # The scene's own first words, to the player alone: what they
        # have walked into, before anybody asks them what they say about
        # it. Authored, so it costs nothing and reads the same every time.
        session.director.brief(user_character, user_character.location_id)
        opening = session.director.establish(user_character.location_id)
        # Who the player said they were, once, at the start of the story
        # and never again. An ordinary event in their own room, so the
        # people standing there perceive it and the people elsewhere
        # never do — which is why it is asked for as what anyone can see.
        if began and player and player.look.strip():
            opening = session.director.introduce(
                user_character.location_id, player.look.strip()
            ) or opening
        # And then whoever is standing there gets to speak first if they
        # want to. Arriving somewhere and having to talk to the air to
        # find out you are not alone is the wrong way round: a story
        # should meet you. Only for people actually in the room — this
        # costs a bid each, once.
        if opening is not None and session.present():
            session.director.open_turn(opening)
        return session

    def here(self) -> str:
        return self.director.current_location(self.user_character)

    def present(self) -> list[Character]:
        """Who the user's character can see is here with them.

        Derived by comparing against their own room, so it can only ever
        name someone standing in it. A client asking "who is here" must
        never be answered with where everyone in the scene is.
        """
        return co_present(
            self.user_character, self.characters, self.store.get_events(self.scene.id)
        )

    def pov(self, events: list[Event]) -> list[ProjectedEvent]:
        """What the user's character perceived of these events.

        The full log is projected (perception depends on where they were
        at the time), then narrowed to the events asked about. Anything
        they could not perceive is simply absent from the result — it is
        never included and marked, because a client that receives it can
        leak it.
        """
        all_events = self.store.get_events(self.scene.id)
        seqs = {event.seq for event in events}
        return [
            projected
            for projected in self.director.contexts.project(self.user_character, all_events)
            if projected.event.seq in seqs
        ]

    def perceived_so_far(self) -> list[ProjectedEvent]:
        """The whole scene as this character experienced it — what a
        client needs to catch up on reconnect."""
        return self.director.contexts.project(
            self.user_character, self.store.get_events(self.scene.id)
        )

    # --- Turns, and taking one again -----------------------------------

    def _play(self, take: Callable[[], list[ProjectedEvent]]) -> list[ProjectedEvent]:
        """Run one player action as a single unit of work.

        Opening a turn settles the one before it, so the take that is
        still discardable is always the most recent — exactly the one a
        player would want back.

        A take that raises is thrown away whole, and the scene is left
        exactly where it stood. Half a turn is the worst outcome
        available: a model that failed on the third of five characters
        would otherwise leave two of them having heard something the
        others never will, permanently, in the file. The machinery for
        this is the same savepoint `/again` uses — the only new part is
        that a failure counts as a reason to use it.
        """
        self.store.begin_turn()
        self.turn_started_at = self.store.next_seq(self.scene.id)
        self.ended_at_turn_start = self.ended()
        self._turns_before_take = self.turns_played
        was_playable = self._last_take
        self.turns_played += 1
        self.takes_played += 1
        self._last_take = take
        # Inside the turn, so a take that is thrown away does not leave
        # the library claiming it happened.
        self.store.touch_story(self.scene.id, self.turns_played)
        try:
            return take()
        except Exception:
            self.store.rollback_turn()
            self.turns_played = self._turns_before_take
            self.takes_played -= 1
            # A take that never landed is not one to offer back: `/again`
            # still means the last moment that actually happened.
            self._last_take = was_playable
            raise

    def use(
        self,
        llm: LLMClient | None,
        workers: int | None = None,
        interpret_beliefs: bool | None = None,
        direct_beats: bool | None = None,
    ) -> None:
        """Change which model answers, mid-story.

        Four objects hold the client — the session, the director, the
        context builder and the narrator — and a change that reached
        three of them would leave a character still talking to the old
        endpoint. Nothing about the story moves: the log, the beliefs and
        the trust are all the engine's, and the model is only who is
        asked next.
        """
        self.llm = llm
        self.director.llm = llm
        self.director.contexts.llm = llm
        self.director.narrator.llm = llm
        if workers is not None:
            self.director.workers = workers
        if interpret_beliefs is not None:
            self.director.interpret_beliefs = interpret_beliefs
        if direct_beats is not None:
            self.director.direct_beats = direct_beats

    def close(self) -> None:
        """Settle the scene and let go of the database.

        The open turn is uncommitted on purpose — that is what makes it
        discardable — so somebody has to say when play is over. Nothing
        did, and the last turn of every session on a file database was
        silently lost: eight events in the process, four on disk.

        Idempotent, so a client can call it on every exit path without
        checking which one it took.
        """
        self.store.commit_turn()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def can_regenerate(self) -> bool:
        return self._last_take is not None

    def regenerate(self) -> list[ProjectedEvent]:
        """Throw the last take away and play it again.

        The player is a director calling "again", not a player being
        asked to approve the world — which is why nothing in the engine
        stops to ask permission before changing something. It acts, and
        this is how the take gets rejected.

        Everything durable lives in one connection, so discarding the
        savepoint undoes the whole turn at once: the events, the beliefs
        encoded from them, the readings, the rehearsals, the interaction
        counts, and any trust that moved. The scene is left exactly where
        it stood, down to the sequence number, so the retake occupies the
        slot the discarded one did rather than being pasted after it.
        """
        if self._last_take is None:
            raise ValueError("nothing has been played yet")
        take = self._last_take
        self.store.rollback_turn()
        # A retake is the same turn played again, not a second one. The
        # rollback puts the saved count back on disk; without this the
        # session's own counter would carry on climbing and the next
        # turn would write the inflated number over it.
        self.turns_played = self._turns_before_take
        return self._play(take)

    def say(self, text: str) -> list[ProjectedEvent]:
        """The whole turn as this character perceived it, their own line
        included. Trimming the echo is a presentation choice, so it
        belongs to the client — a session that withheld a perceived event
        would hand different transcripts to different clients."""
        def take() -> list[ProjectedEvent]:
            event = self.director.build_event(
                "utterance", self.user_character.id, self.here(), text
            )
            return self.pov(self.director.run_turn(event))

        return self._play(take)

    def move(self, room_id: str) -> list[ProjectedEvent]:
        here = self.here()
        found = find_room(self.world, room_id)
        if found is None:
            # Not a place the author wrote. In a world that allows it, the
            # map grows: a school has corridors, and answering "there is
            # no library to go to" is answering with the scaffolding.
            room = discover(
                self.world, room_id, here, self.store, self.director.narrator
            )
            if room is None:
                raise ValueError(f"there is no room {room_id!r} in this world")
            found = room.id
        room_id = found
        if room_id == here:
            return []
        def take() -> list[ProjectedEvent]:
            arrival = self.director.build_event(
                "arrival",
                self.user_character.id,
                room_id,
                f"{self.user_character.name} comes in from {self.world.room_name(self.here())}.",
                audibility="adjacent",
            )
            return self.pov(self.director.run_turn(arrival))

        return self._play(take)

    def next_scene(self) -> str | None:
        """The scene this one leads to, given what happened in it."""
        events = self.store.get_events(self.scene.id)
        return next_scene(
            self.scene.next,
            scene_state(events, self.characters, self.world),
            self.world.facts,
        )

    def go_on(self) -> Session | None:
        """Open the next scene of the story on the same store.

        A new `Session`, because the cast, the map positions and the
        director are all the next scene's rather than this one's — but the
        same store, so everybody arrives carrying what the last scene did
        to them, aged on the way in. The open turn is settled first: this
        is a seam in the story, and the take before it is no longer one
        the player can ask to have again.
        """
        following = self.next_scene()
        if following is None:
            return None
        self.close()
        # Crossing the seam is itself progress worth saving: a player who
        # finishes a scene and stops must be resumed into the scene they
        # reached, not handed back the one they just played out.
        self.store.touch_story(following, self.turns_played)
        return Session.open(
            self.world_dir,
            following,
            llm=self.llm,
            interpret_beliefs=self.director.interpret_beliefs,
            store=self.store,
            workers=self.director.workers,
            direct_beats=self.director.direct_beats,
        )

    def items_here(self) -> list[Item]:
        """What is in this room to be read."""
        return self.world.items_in(self.here())

    def read(self, wanted: str) -> list[ProjectedEvent]:
        """Read something in this room.

        Two events, because reading is two things. The room sees you open
        it — an ordinary action, perceived by whoever is standing there.
        What it says is private and addressed to you, so the same
        perception rules that keep a conversation in one room keep the
        contents in one pair of eyes.

        It follows for free that reading a file is not the same as saying
        what is in it: the private event's only perceiver is its own
        actor, so nothing counts it as spoken and an arc waiting for
        somebody to say it out loud is still waiting.
        """
        item = self.world.find_item(self.here(), wanted)
        if item is None:
            raise ValueError(f"there is no {wanted!r} here")

        def take() -> list[ProjectedEvent]:
            you = self.user_character
            here = self.here()
            opened = self.store.append_event(
                self.director.build_event(
                    "action",
                    you.id,
                    here,
                    self.world.phrasing.say("opens", name=you.name, thing=item.name),
                    metadata={"opened": item.id},
                )
            )
            self.director._absorb()
            contents = self.store.append_event(
                self.director.build_event(
                    "action",
                    you.id,
                    here,
                    item.text.strip(),
                    audibility="private",
                    addressed_to=[you.id],
                    metadata={"read": item.id},
                )
            )
            self.director._absorb()
            return self.pov([opened, contents])

        return self._play(take)

    def ended(self) -> bool:
        """Has this scene reached its declared end condition?

        Advisory rather than enforced: the engine does not lock the scene,
        it reports that the thing the author was building toward has
        happened. What a client does with that — offer the reveal, roll
        credits, keep going — is the client's call.
        """
        events = self.store.get_events(self.scene.id)
        return has_ended(
            self.scene.end_condition,
            scene_state(events, self.characters, self.world),
            self.world.facts,
        )

    def pending_skip(self) -> int | None:
        """How long the next derived skip would be, so a client can ask
        the user before spending their character's time."""
        return self.director.pending_skip()

    def wait(self, consent: Callable[[int], bool] | None = None) -> list[ProjectedEvent]:
        return self._play(lambda: self.pov(self.director.advance_time(consent=consent)))

    def look(self) -> list[ProjectedEvent]:
        """Take in the room: coarse off-screen events here expand into
        what is visible now."""
        def take() -> list[ProjectedEvent]:
            revealed = []
            for summary in self.director.unmaterialized_here(self.user_character):
                event = self.director.materialize(summary, self.user_character)
                if event is not None:
                    revealed.append(event)
            return self.pov(revealed)

        return self._play(take)
