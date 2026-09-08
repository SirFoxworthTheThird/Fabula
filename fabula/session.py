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
from fabula.db import EventStore
from fabula.director import Director
from fabula.llm import LLMClient, get_default_llm
from fabula.memory import co_present
from fabula.loader import Scene, load_pressures, load_scenario
from fabula.models import Character, Event, ProjectedEvent
from fabula.narrator import Narrator
from fabula.persistence import begin_scene
from fabula.pressures import has_ended, scene_state
from fabula.world import World


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

    @classmethod
    def open(
        cls,
        world_dir: Path,
        scene_name: str,
        db_path: str = ":memory:",
        llm: LLMClient | None = None,
        interpret_beliefs: bool = True,
    ) -> Session:
        world, characters, scene = load_scenario(world_dir, scene_name)
        pressures = load_pressures(world_dir)
        store = EventStore(db_path)
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
        )

        # Characters are durable: with a real db path they arrive carrying
        # what they already believe, aged by the time between scenes.
        begin_scene(store, characters)

        session = cls(world, characters, scene, store, director, user_character)
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

    def say(self, text: str) -> list[ProjectedEvent]:
        """The whole turn as this character perceived it, their own line
        included. Trimming the echo is a presentation choice, so it
        belongs to the client — a session that withheld a perceived event
        would hand different transcripts to different clients."""
        event = self.director.build_event("utterance", self.user_character.id, self.here(), text)
        return self.pov(self.director.run_turn(event))

    def move(self, room_id: str) -> list[ProjectedEvent]:
        if room_id not in self.world.rooms:
            raise ValueError(f"there is no room {room_id!r} in this world")
        here = self.here()
        if room_id == here:
            return []
        arrival = self.director.build_event(
            "arrival",
            self.user_character.id,
            room_id,
            f"{self.user_character.name} comes in from {self.world.room_name(here)}.",
            audibility="adjacent",
        )
        return self.pov(self.director.run_turn(arrival))

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
            scene_state(events, self.characters),
            self.world.facts,
        )

    def pending_skip(self) -> int | None:
        """How long the next derived skip would be, so a client can ask
        the user before spending their character's time."""
        return derive_skip_minutes(self.characters, self.store.get_events(self.scene.id))

    def wait(self, consent: Callable[[int], bool] | None = None) -> list[ProjectedEvent]:
        return self.pov(self.director.advance_time(consent=consent))

    def look(self) -> list[ProjectedEvent]:
        """Take in the room: coarse off-screen events here expand into
        what is visible now."""
        revealed = []
        for summary in self.director.unmaterialized_here(self.user_character):
            event = self.director.materialize(summary, self.user_character)
            if event is not None:
                revealed.append(event)
        return self.pov(revealed)
