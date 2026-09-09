"""The director: speaker selection and turn-loop arbitration (spec §7).

Arbitration itself is plain code — argmax over bids — because by the time
bids exist, the model calls that matter (character generation, and any
LLM-assisted ambiguous bid) have already happened. The director's own
model use is reserved for pressure selection and time advancement,
which land in later milestones; M0 has neither.

The director sees every bid, including narrator bids, but only ever
emits a speaker id (or "yield"/"narrate"). Bid rationales stop here —
they are never written into an Event, so they can never reach another
character's projection.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Literal

from fabula.agents import generate_utterance
from fabula.bidding import get_bid, prefilter_candidates
from fabula.classify import classify
from fabula.concurrency import DEFAULT_WORKERS, in_parallel
from fabula.chronology import (
    LARGE_SKIP_MINUTES,
    derive_skip_minutes,
    describe_duration,
    due_intentions,
)
from fabula.db import EventStore
from fabula.discovery import invents_a_fact
from fabula.loader import Scene
from fabula.memory import ContextBuilder, form_belief, location_at_seq, record_rehearsals
from fabula.interpret import (
    INTERPRETATION_WINDOW,
    keep_interpretation,
    propose_interpretation,
)
from fabula.persistence import (
    begin_scene,
    close_reached_goals,
    encode_belief,
    witnessed_withholding,
    worth_keeping,
)
from fabula.models import Belief, Bid, Character, Event, Intention, Pressure, ProjectedEvent
from fabula.beats import NOBODY_SPEAKS, Beat
from fabula.beats import available as available_beats
from fabula.beats import choose as choose_beat
from fabula.narrator import NARRATOR_ID, Narrator, as_bid
from fabula.pressures import scene_state, select_pressure
from fabula.world import World, resolve_perception

DecisionKind = Literal["speak", "narrate", "yield_to_user", "fire_pressure", "withhold"]

YIELD_FLOOR = 0.12


def _fold(text: str) -> str:
    """Strip diacritics and lowercase, so "Tomas" matches "Tomás"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def detect_addressed_to(content: str, characters: dict[str, Character], exclude_id: str | None) -> list[str]:
    """Deterministic, text-only detection of who a line names by first
    name — this feeds turn-taking (spec §7 step 3's "addressed by name"
    criterion), not knowledge filtering, so it carries no invariant-1 risk."""
    folded = _fold(content)
    addressed = []
    for character_id, character in characters.items():
        if character_id == exclude_id:
            continue
        first_name = _fold(character.name.split()[0])
        if re.search(rf"\b{re.escape(first_name)}\b", folded):
            addressed.append(character_id)
    return addressed


@dataclass
class Decision:
    kind: DecisionKind
    character_id: str | None = None
    pressure: Pressure | None = None
    bids: list[Bid] = field(default_factory=list)


def arbitrate(
    bids: list[Bid],
    narrator_bid: Bid | None,
    consecutive_agent_turns: int,
    max_consecutive_agent_turns: int,
    pressure_choice: tuple[Pressure, float] | None = None,
) -> Decision:
    """Guard first: a hard cap on consecutive non-user turns, so
    agent-to-agent exchange always terminates and the user is never
    starved out of their own scene (spec §7 guards). A pressure never
    overrides that guard — the user's turn outranks the drama."""
    if consecutive_agent_turns >= max_consecutive_agent_turns:
        return Decision(kind="yield_to_user", bids=bids)

    contenders = list(bids)
    if narrator_bid is not None:
        contenders.append(narrator_bid)

    best_desire = max((b.desire for b in contenders), default=0.0)

    if pressure_choice is not None and pressure_choice[1] >= max(best_desire, YIELD_FLOOR):
        return Decision(kind="fire_pressure", pressure=pressure_choice[0], bids=bids)

    if not contenders:
        return Decision(kind="yield_to_user", bids=bids)

    winner = max(contenders, key=lambda b: b.desire)
    if winner.desire < YIELD_FLOOR:
        return Decision(kind="yield_to_user", bids=bids)
    if winner.character_id == NARRATOR_ID:
        return Decision(kind="narrate", character_id=NARRATOR_ID, bids=bids)
    if winner.kind == "withhold":
        return Decision(kind="withhold", character_id=winner.character_id, bids=bids)
    return Decision(kind="speak", character_id=winner.character_id, bids=bids)


class Director:
    def __init__(
        self,
        store: EventStore,
        world: World,
        characters: dict[str, Character],
        scene: Scene,
        narrator: Narrator,
        llm=None,
        pressures: list[Pressure] | None = None,
        interpret_beliefs: bool = True,
        waiting: dict[str, Character] | None = None,
        classify_reports: bool = True,
        workers: int = DEFAULT_WORKERS,
        direct_beats: bool = True,
    ):
        self.store = store
        self.world = world
        self.characters = characters
        self.scene = scene
        self.narrator = narrator
        self.llm = llm
        # Reading each remembered moment is one model call per character
        # per moment, and it dominates the bill for a scene. Turning it
        # off costs the annotations and nothing that was provable.
        self.interpret_beliefs = interpret_beliefs
        # Reading a line for what it reports costs a model call, but only
        # for a line that names one of the world's facts — almost none do.
        self.classify_reports = classify_reports
        # Whether the model gets to choose which beat the narrator
        # writes. One call per narration, and only when there is more
        # than one thing the moment could be; off, the first of the
        # offered beats is taken, which is the deterministic order.
        self.direct_beats = direct_beats
        # How many of the independent calls in a turn — the bids, the
        # readings — may be in flight at once. Somebody else's endpoint
        # is on the other end of them; 1 is the old sequential engine.
        self.workers = workers
        self.pressures = pressures or []
        # Written for this world, not in the room when it opened. An
        # authored pressure may bring one on; nothing else can, and the
        # narrator least of all — it writes prose, and a person who
        # arrives has to arrive as an event.
        self.waiting = waiting or {}
        self.contexts = ContextBuilder(world, characters, scene.id, store, llm)

    def build_event(
        self,
        kind: str,
        actor_id: str | None,
        location_id: str,
        content: str,
        audibility: str = "room",
        metadata: dict | None = None,
        story_time: datetime | None = None,
        detail_level: str = "full",
        addressed_to: list[str] | None = None,
    ) -> Event:
        all_events = self.store.get_events(self.scene.id)
        if story_time is None:
            story_time = all_events[-1].story_time if all_events else self.scene.start_time
        if addressed_to is None:
            addressed_to = (
                detect_addressed_to(content, self.characters, exclude_id=actor_id)
                if kind in ("utterance", "action")
                else []
            )
        return Event(
            scene_id=self.scene.id,
            seq=self.store.next_seq(self.scene.id),
            story_time=story_time,
            kind=kind,
            actor_id=actor_id,
            location_id=location_id,
            content=content,
            audibility=audibility,
            addressed_to=addressed_to,
            metadata=metadata or {},
            detail_level=detail_level,
        )

    def run_turn(self, user_event: Event) -> list[Event]:
        """Runs spec §7 steps 2-8 for one user input. `user_event` must
        already be constructed (kind/location/etc set by the caller); this
        appends it and then loops until yield or turn-budget exhaustion."""
        stored_user_event = self.store.append_event(user_event)
        self._absorb()
        events_this_turn = [stored_user_event]
        reported = self.record_transmission(stored_user_event)
        if reported is not None:
            events_this_turn.append(reported)
            self._absorb()
        return events_this_turn + self._loop(stored_user_event, must_answer=True)

    def open_turn(self, opening: Event) -> list[Event]:
        """Let the room have the first word.

        A story that waits for the player to speak first puts the whole
        burden of starting it on them: you arrive somewhere, nobody says
        anything, and the only way to find out you are not alone is to
        talk to the air. Whoever is standing there bids on the scene's
        opening exactly as they would on a line — so somebody who has
        nothing to say still says nothing, and it costs one bid each,
        once, for the people actually in the room with you.

        One beat, and no pressures. A greeting is the room noticing you;
        a pressure is the director escalating, and a story whose first
        move is its own complication has started without you. The budget
        of one says the same thing from the other side: an opening is a
        hello, not a conversation you were not in.
        """
        return self._loop(opening, budget=1, pressures=False)

    def _loop(
        self,
        last_event: Event,
        budget: int | None = None,
        pressures: bool = True,
        must_answer: bool = False,
    ) -> list[Event]:
        """Bid, arbitrate, act, repeat — until somebody yields to the
        player or the scene runs out of budget."""
        events_this_turn: list[Event] = []
        consecutive_agent_turns = 0
        answered = False

        for _ in range(budget if budget is not None else self.scene.turn_budget):
            all_events = self.store.get_events(self.scene.id)
            candidates = prefilter_candidates(last_event, self.characters, all_events, self.world)

            # Every bid, the narrator's included, goes out at once.
            # Nobody's bid can see anybody else's — that is what makes
            # these separate agents — so making them queue was pure
            # waiting, and the fuller the room the longer it got.
            def bid_for(character: Character) -> Callable[[], Bid]:
                location = location_at_seq(
                    character.id, character.location_id, all_events, last_event.seq + 1
                )
                level = resolve_perception(last_event, character.id, location, self.world)
                return lambda: get_bid(
                    character, last_event, level, all_events, self.contexts, self.llm
                )

            # What the room could use, decided here rather than left to
            # the narrator to infer from the last line. The director is
            # the only omniscient component, so what it hands over is one
            # id from a closed set plus something anybody standing there
            # can already see — never a sentence, and never anything read
            # from beliefs or trust.
            offered = available_beats(
                last_event,
                all_events,
                self.world,
                self.characters,
                protagonist_id=self.protagonist_id(),
                alone=not candidates,
            )
            beat = offered[0] if offered else None
            bids: list[Bid] = in_parallel(
                [bid_for(character) for character in candidates], self.workers
            )
            narrator_bid = as_bid(beat)
            pressure_choice = (
                select_pressure(
                    self.pressures,
                    scene_state(all_events, self.characters, self.world),
                    self.world.facts,
                    self.scene.mode,
                    max((b.desire for b in bids), default=0.0),
                )
                if pressures
                else None
            )
            decision = arbitrate(
                bids,
                narrator_bid,
                consecutive_agent_turns,
                self.scene.max_consecutive_agent_turns,
                pressure_choice,
            )

            if decision.kind == "yield_to_user":
                # A turn the player perceives nothing of is the worst
                # answer the app can give: they say something into a room
                # with somebody standing in it and get "No one answers."
                # Nobody bid and no beat was due — so the room takes the
                # turn rather than nobody having it.
                #
                # *Perceived*, not merely appended: a pressure firing two
                # rooms away is the story moving, and it is still silence
                # where the player is standing.
                if not must_answer or answered:
                    break
                anchor = self._heard_last(all_events)
                if anchor is None:
                    break
                last_event, beat = anchor
                decision = Decision(kind="narrate", character_id=NARRATOR_ID, bids=bids)

            if decision.kind == "fire_pressure":
                new_event = self._fire(decision.pressure)
            elif decision.kind == "withhold":
                character = self.characters[decision.character_id]
                location = location_at_seq(
                    character.id, character.location_id, all_events, last_event.seq + 1
                )
                # Not answering is something the room can see, so it is an
                # ordinary event with an actor — visible, attributable, and
                # perception-filtered like anything else.
                new_event = self.build_event(
                    "action",
                    character.id,
                    location,
                    self.narrator.render_withholding(character, self.world, location),
                    metadata={"withheld": True},
                )
            elif decision.kind == "narrate":
                # Who is standing there, and never the player among
                # them: naming them is inviting the one thing the
                # narrator must not do.
                here = [
                    character.name
                    for character in self.characters.values()
                    if not character.is_user
                    and location_at_seq(
                        character.id, character.location_id, all_events, last_event.seq + 1
                    )
                    == last_event.location_id
                ]
                # Which of the moments this could be is a judgement, not
                # a rule — the room has gone quiet *and* somebody has
                # stopped talking *and* there is a letter nobody has
                # picked up. Asked only now, once the narrator has won
                # the turn, so the call is paid for by a narration that
                # is definitely being written.
                beat = self.pick_beat(offered, last_event, here) or beat
                content = self.narrator.generate(
                    last_event, all_events, self.world, beat=beat, present=here
                )
                named = invents_a_fact(content, self.world)
                if named:
                    # A narration that names a fact raises the subject in
                    # front of everybody in the room — it satisfies
                    # `fact_spoken` and can end an arc that was waiting
                    # for somebody to say it out loud. Nobody said it, so
                    # the beat is dropped and the turn goes on.
                    consecutive_agent_turns += 1
                    continue
                new_event = self.build_event("narration", None, last_event.location_id, content)
            else:
                character = self.characters[decision.character_id]
                location = location_at_seq(
                    character.id, character.location_id, all_events, last_event.seq + 1
                )
                content = generate_utterance(character, all_events, self.contexts, self.llm)
                new_event = self.build_event("utterance", character.id, location, content)

            last_event = self.store.append_event(new_event)
            events_this_turn.append(last_event)
            answered = answered or self._reaches_user(last_event)
            self._absorb()
            # Anybody can report having told somebody something, not only
            # the player.
            reported = self.record_transmission(last_event)
            if reported is not None:
                events_this_turn.append(reported)
                self._absorb()
            consecutive_agent_turns += 1

            if self._addresses_user(last_event):
                # Someone spoke to the player. Carrying on past that lets
                # another character answer a question the player was
                # asked, which is the sharpest way to make them a
                # spectator in their own scene — the turn budget alone
                # does not prevent it, because the loop simply had turns
                # left. The floor is theirs.
                break

        return events_this_turn

    def pick_beat(
        self, offered: list[Beat], last_event: Event, here: list[str]
    ) -> Beat | None:
        """Which of the available beats this moment wants.

        The model proposes and the engine disposes, exactly as everywhere
        else it is asked anything: it answers with one id, the id has to
        be one of the ids offered, and anything else falls back to the
        deterministic first choice. It cannot invent a beat, cannot write
        an instruction, and cannot reach past the closed vocabulary.

        What it is shown is what the room can see — the same material the
        narrator gets. The director is omniscient; there is no reason to
        hand any of that to something whose whole job is picking between
        three labels.
        """
        if self.llm is None or not self.direct_beats or len(offered) < 2:
            return None
        options = "\n".join(f"- {beat.id}: {beat.label}" for beat in offered)
        system = (
            "You direct a scene in an interactive story. You are choosing what the "
            "narrator should give a beat to next — not writing it. Answer with one id "
            "from the list and nothing else."
        )
        prompt = (
            f"Location: {self.world.room_name(last_event.location_id)}\n"
            + (f"Who is here: {', '.join(here)}\n" if here else "")
            + f"The last thing that happened ({last_event.kind}): {last_event.content}\n\n"
            f"What could take the beat:\n{options}\n\n"
            "Which one does this moment want? Answer with the id alone."
        )
        answer = self.llm.complete(system=system, prompt=prompt, key="beat")
        wanted = (answer or "").strip().strip(".\"'`").lower()
        for beat in offered:
            if beat.id == wanted:
                return beat
        # A model that answered with a sentence, or with a beat nobody
        # offered, has said nothing. The scene takes the first one.
        return None

    def _reaches_user(self, event: Event) -> bool:
        """Did the player perceive any of that, at any fidelity?"""
        user = next((c for c in self.characters.values() if c.is_user), None)
        if user is None:
            return False
        where = self.current_location(user)
        return resolve_perception(event, user.id, where, self.world) != "none"

    def _heard_last(self, all_events: list[Event]) -> tuple[Event, Beat] | None:
        """Something to hang a beat on, in the room the player is in.

        The anchor has to be a line they actually heard: `last_event` may
        be two rooms away, and putting it in the prompt would narrate
        their room out of words they never perceived. With nothing heard
        yet there is still the room itself, which needs no anchor at all.
        """
        user = next((c for c in self.characters.values() if c.is_user), None)
        if user is None:
            return None
        where = self.current_location(user)
        heard = [
            event
            for event in all_events
            if event.location_id == where
            and resolve_perception(event, user.id, where, self.world) == "full"
        ]
        if not heard:
            return None
        return heard[-1], Beat(NOBODY_SPEAKS, 1.0, "nobody had anything to say")

    def protagonist_id(self) -> str | None:
        return next((c.id for c in self.characters.values() if c.is_user), None)

    def _addresses_user(self, event: Event) -> bool:
        user = next((c for c in self.characters.values() if c.is_user), None)
        return bool(user and event.actor_id != user.id and user.id in event.addressed_to)

    def advance_time(
        self,
        minutes: int | None = None,
        consent: Callable[[int], bool] | None = None,
    ) -> list[Event]:
        """Skip forward to the next moment something is ready, log the
        skip, and resolve what happened off screen while it elapsed.

        Returns the events appended, empty if nothing was pending or the
        user declined a large skip.
        """
        all_events = self.store.get_events(self.scene.id)
        if minutes is None:
            minutes = derive_skip_minutes(self.characters, all_events, blocked=self.unreachable)
        if minutes is None or minutes <= 0:
            return []

        # The user is a character with agency: time is not something they
        # should ever lose without noticing (spec §8).
        if minutes > LARGE_SKIP_MINUTES and not (consent and consent(minutes)):
            return []

        previous_time = all_events[-1].story_time if all_events else self.scene.start_time
        user_location = next(
            (c.location_id for c in self.characters.values() if c.is_user),
            next(iter(self.world.rooms)),
        )
        skip = self.build_event(
            "time_skip",
            None,
            user_location,
            f"{describe_duration(minutes, self.world.phrasing)} pass.",
            audibility="building",
            metadata={"minutes": minutes},
            story_time=previous_time + timedelta(minutes=minutes),
        )
        appended = [self.store.append_event(skip)]
        appended.extend(self._resolve_offscreen(minutes))
        self._absorb()
        return appended

    def establish(self, where: str) -> Event | None:
        """The line that opens a scene.

        A story that begins with a blank prompt is a text box, not a
        story: the player is told the name of the room and left to guess
        what is in it. The narrator has always had a 0.9 "establish the
        scene" bid for exactly this and it could never fire — by the time
        anything is bid on, the player has already spoken and the log is
        no longer empty. So the opening is not bid on at all; it is the
        one narration the scene owes the player before they type.

        Nothing if the scene has already started, so resuming a story
        does not re-describe a room somebody is standing in the middle of.
        """
        # Anything but the scene's own opening words, which come first
        # and are addressed to the player rather than to the room.
        if [e for e in self.store.get_events(self.scene.id) if not e.metadata.get("opening")]:
            return None
        content = self.narrator.describe_place(where, self.world)
        named = invents_a_fact(content, self.world)
        if named:
            # An opening line that names a secret hands it to everyone in
            # the room before anyone has spoken — and can end an arc on
            # the first beat. Fall back to what the author wrote, and to
            # silence if that names one too.
            authored = self.world.rooms[where].description if where in self.world.rooms else ""
            if not authored or invents_a_fact(authored, self.world):
                return None
            content = authored
        # Not absorbed: the curtain going up is not something that
        # happened to anybody. Everyone in the room reads it in their
        # context like any other line, but nobody forms a durable memory
        # of what the room they are standing in looks like — which would
        # otherwise cost a reading per character before the player has
        # typed anything.
        return self.store.append_event(
            self.build_event("narration", None, where, content)
        )

    def brief(self, who: Character, where: str) -> Event | None:
        """The scene's own first words, to the player and nobody else.

        Every other platform on this shelf opens with one: a paragraph
        that says what you have walked into, so the first thing asked of
        somebody is not "what do you say" to a room they know nothing
        about. Authored rather than generated — the same every time, free,
        and good prose instead of whatever a model made of a room name.

        Private, through the ordinary perception path: `audibility:
        private` addressed to the player means the choke point every leak
        test covers already returns "none" for everybody else. That is
        what lets it be written in the second person, and lets it say
        what only this character would know coming in.
        """
        if not self.scene.opening.strip() or self.store.get_events(self.scene.id):
            return None
        return self.store.append_event(
            self.build_event(
                "narration",
                None,
                where,
                self.scene.opening.strip(),
                audibility="private",
                addressed_to=[who.id],
                metadata={"opening": True},
            )
        )

    def introduce(self, where: str, look: str) -> Event | None:
        """What the room can see of the player, in the player's own words.

        An ordinary event, so it is filtered like one: the people in the
        room perceive it and the people elsewhere never do. It is not
        world truth and nothing checks it against any — the same as
        anything anybody says about themselves — but it is *perceivable*,
        which is what makes it something the room can answer.

        A description that names one of the world's own facts is refused:
        it would hand a secret to everybody standing there before a word
        was spoken, and could end an arc on its first beat.
        """
        if not look.strip() or invents_a_fact(look, self.world):
            return None
        return self.store.append_event(self.build_event("narration", None, where, look.strip()))

    def current_location(self, character: Character) -> str:
        """Where a character is now, replayed from their arrivals in the
        log rather than read off the scene's starting snapshot."""
        all_events = self.store.get_events(self.scene.id)
        last_seq = all_events[-1].seq if all_events else 0
        return location_at_seq(character.id, character.location_id, all_events, last_seq + 1)

    def _resolve_offscreen(self, elapsed_minutes: int) -> list[Event]:
        """Off-screen action is resolved coarsely: elapsed time produces a
        few `detail_level: "summary"` events per character. Specifics are
        materialized lazily, when someone is actually there to see them.

        Only genuinely off-screen characters are resolved this way. Someone
        standing in the room with the user is not off screen, and flattening
        what they did into a coarse stub would both under-describe it and
        hand the user a summary-resolution line as if they had watched it
        happen. Their intention simply stays pending.
        """
        appended = []
        user = next((c for c in self.characters.values() if c.is_user), None)
        user_location = self.current_location(user) if user else None

        for character in self.characters.values():
            if character.is_user:
                continue
            if user_location is not None and self.current_location(character) == user_location:
                continue
            events = self.store.get_events(self.scene.id)
            for intention in due_intentions(character, events, elapsed_minutes):
                if intention.private and self._others_present(
                    intention.location_id, character.id
                ):
                    # He is not going to check the glue seam with his
                    # sister standing right there. It waits.
                    continue
                summary = self.build_event(
                    "action",
                    character.id,
                    intention.location_id,
                    f"{character.name} {intention.description}",
                    metadata={"intention_id": intention.id},
                    detail_level="summary",
                )
                appended.append(self.store.append_event(summary))
                if intention.state:
                    # A state is a fact about them, not a coarse stub of
                    # something they did, so it is logged in full and
                    # separately — `was_asleep` reads these, and a summary
                    # that later materialises must not change the answer.
                    appended.append(
                        self.store.append_event(
                            self.build_event(
                                "state_change",
                                character.id,
                                intention.location_id,
                                f"{character.name} is {intention.state}.",
                                metadata={
                                    "character_id": character.id,
                                    "state": intention.state,
                                    "intention_id": intention.id,
                                },
                            )
                        )
                    )
        return appended

    def unreachable(self, character: Character, intention: Intention) -> bool:
        """Would resolution refuse this intention as things stand?

        The same two conditions `_resolve_offscreen` applies, so the skip
        derivation cannot propose a jump that resolution will decline.
        """
        user = next((c for c in self.characters.values() if c.is_user), None)
        if user is not None and self.current_location(character) == self.current_location(user):
            return True
        return bool(intention.private and self._others_present(intention.location_id, character.id))

    def pending_skip(self) -> int | None:
        """How long the next skip would be, counting only what could
        actually come of it."""
        return derive_skip_minutes(
            self.characters, self.store.get_events(self.scene.id), blocked=self.unreachable
        )

    def _others_present(self, location_id: str, actor_id: str) -> bool:
        """Is anyone but the actor standing in this room right now?"""
        return any(
            character.id != actor_id and self.current_location(character) == location_id
            for character in self.characters.values()
        )

    def materialize(self, summary_event: Event, observer: Character) -> Event | None:
        """Expand a coarsely-resolved off-screen event into specifics — the
        moment the user finds the kitchen ransacked is when "Tomás searched
        the house" becomes a description.

        Only from inside the room: standing elsewhere, there is nothing to
        see, so there is nothing to materialize. Appends rather than
        rewrites, because the log is append-only; the specifics are a new
        perceivable event, filtered like any other.
        """
        all_events = self.store.get_events(self.scene.id)
        observer_location = location_at_seq(
            observer.id, observer.location_id, all_events, all_events[-1].seq + 1
        )
        if summary_event.location_id != observer_location:
            return None
        if summary_event.detail_level != "summary":
            return None
        if any(e.metadata.get("materializes") == summary_event.id for e in all_events):
            return None

        content = self.narrator.materialize(summary_event, self.world)
        event = self.build_event(
            "narration",
            None,
            summary_event.location_id,
            content,
            metadata={"materializes": summary_event.id},
        )
        appended = self.store.append_event(event)
        self._absorb()
        return appended

    def unmaterialized_here(self, observer: Character) -> list[Event]:
        """Coarse off-screen events waiting in the observer's room.

        An empty log is a real case, not an edge one: looking around is
        the first thing many players do, and before that there is nothing
        to look past. This used to index the last event of an empty list
        and crash — in every world, since M4 — because every test spoke
        before it looked.
        """
        all_events = self.store.get_events(self.scene.id)
        if not all_events:
            return []
        location = location_at_seq(
            observer.id, observer.location_id, all_events, all_events[-1].seq + 1
        )
        done = {e.metadata.get("materializes") for e in all_events}
        return [
            e
            for e in all_events
            if e.detail_level == "summary" and e.location_id == location and e.id not in done
        ]

    def _fire(self, pressure: Pressure) -> Event:
        """Turn an authored pressure into one ordinary event. The firing is
        recorded in the event's metadata, so cooldown and max_fires derive
        from the log rather than from state held on the side — an unlogged
        firing would be the same class of bug as an unlogged utterance."""
        effect = pressure.effect
        location_id = effect["location"]
        actor_id = effect.get("actor")
        if effect["kind"] == "arrival" and actor_id in self.waiting:
            self.admit(actor_id)
        metadata = {"pressure_id": pressure.id}
        if effect["kind"] == "state_change":
            metadata |= {"character_id": actor_id, "state": effect["state"]}
        content = self.narrator.render_pressure(pressure, location_id, self.world)
        return self.build_event(
            effect["kind"],
            actor_id,
            location_id,
            content,
            audibility=effect.get("audibility", "room"),
            metadata=metadata,
        )

    def record_transmission(self, event: Event) -> Event | None:
        """If this line reports having told somebody something, log that
        telling as the perception it describes.

        A private beat, placed where the recipient is and addressed to
        them, so the existing perception rules do all the work: they get
        it in full, nobody else gets it at all, and the speaker knows
        what they themselves said. From there it is an ordinary perceived
        event — it becomes a belief, it shows in the reveal, and the turn
        it landed in can be taken again.

        Its story time is *now*, not backdated. The log records when
        something entered the story, and a timestamp running backwards
        would walk the clock in every client for no gain; the content and
        the metadata say it happened before tonight.
        """
        speaker = self.characters.get(event.actor_id)
        if speaker is None:
            return None
        all_events = self.store.get_events(self.scene.id)
        reachable = {**self.characters, **self.waiting}
        told = classify(
            event,
            speaker,
            reachable,
            self.world,
            # Everything they had perceived *before* this line. Their own
            # utterance is in their projection the moment it is appended,
            # so including it would let anybody bootstrap knowledge by
            # asserting it: "I already told Maria about the music box"
            # would make the music box theirs to pass on.
            [
                perceived
                for perceived in self.contexts.project(speaker, all_events)
                if perceived.event.seq < event.seq
            ],
            self.store,
            self.llm if self.classify_reports else None,
        )
        if told is None:
            return None

        recipient = reachable[told.recipient_id]
        subject = self.world.facts[told.fact_id].keywords[0]
        return self.store.append_event(
            self.build_event(
                "action",
                speaker.id,
                self.current_location(recipient),
                self.world.phrasing.transmission.format(
                    speaker=speaker.name, recipient=recipient.name, subject=subject
                ),
                audibility="private",
                addressed_to=[recipient.id],
                metadata={"transmission": told.fact_id},
            )
        )

    def admit(self, character_id: str) -> Character:
        """Bring somebody who was waiting off-stage into the scene.

        Only ever called for an arrival, which is what makes their memory
        answer itself: `location_at_seq` replays their own arrivals from
        wherever they started, and they started nowhere reachable — so
        everything before the event that brings them in resolves to "none"
        and they walk in knowing only what they walk in on. No backfill,
        no decision about what they might have overheard, and nothing to
        get wrong.
        """
        character = self.waiting.pop(character_id)
        # In place: the session, the context builder and this object all
        # hold the same dict, and somebody who joined only the director's
        # copy would bid without ever being seen to be in the room.
        self.characters[character_id] = character
        begin_scene(self.store, self.characters)
        return character

    def _absorb(self) -> None:
        """After each append, let every character take in what they just
        perceived: rehearse the earlier moments this one re-mentions,
        encode it as a belief if it was salient enough to keep, count the
        interaction, and move their regard for whoever acted.

        All of it runs per character against their own projection, so an
        event they could not perceive is neither rehearsed, remembered,
        counted nor held against anyone — it does not exist for them.
        """
        all_events = self.store.get_events(self.scene.id)
        if not all_events:
            return
        appended = all_events[-1].seq

        # First, what each character took in, worked out before anything
        # is written down: who perceived the new event at all, and what
        # they would remember of it.
        taking_in: list[tuple[Character, list[ProjectedEvent], Belief | None]] = []
        for character in self.characters.values():
            projected = self.contexts.project(character, all_events)
            record_rehearsals(
                character, projected, self.store, self.world.phrasing.stopwords
            )
            if not projected or projected[-1].event.seq != appended:
                # They did not perceive the event that was just appended,
                # so there is nothing new for them to take in. Without
                # this check an older event stays "newest" for them and
                # gets absorbed again on every append they miss — one
                # withheld beat charging them four times over while the
                # others talk in another room.
                continue
            taking_in.append(
                (character, projected, form_belief(character, projected[-1]))
            )

        def reading_for(
            character: Character, projected: list[ProjectedEvent], belief: Belief | None
        ) -> Callable[[], str | None]:
            """What this character privately made of it — or None for a
            moment nobody is going to be asked to read."""
            # Not for the player. Nothing reads their memory back into a
            # prompt — they are holding it — so writing down what they
            # privately think would be the engine deciding their inner
            # life, and paying a model call to do it.
            reader = None if character.is_user or not self.interpret_beliefs else self.llm
            # And never for a moment about to be discarded: paying to
            # read something below the memory floor is money spent on
            # the weather.
            if reader is None or not worth_keeping(belief):
                return lambda: None
            # The run-up is what makes the moment readable: "he said
            # nothing" means one thing after small talk and another after
            # being asked where the music box went. Their perceived lines
            # only, so the reading cannot see further than they did.
            window = projected[-INTERPRETATION_WINDOW:]
            return lambda: propose_interpretation(
                character, window, self.world, self.store, reader
            )

        # One private reading each, and no character's reading can see
        # another's, so they go out together rather than one after the
        # next — this is the loop that costs a call per remembered moment
        # per character, and it was the whole cast waiting in a queue.
        readings = in_parallel(
            [reading_for(*entry) for entry in taking_in], self.workers
        )

        # Then the writing, back on this thread and in cast order: a turn
        # stays one unit of work, and a parallel turn leaves the store in
        # exactly the state a sequential one would.
        for (character, projected, belief), reading in zip(taking_in, readings):
            if reading is not None:
                keep_interpretation(
                    self.store, character, projected[-INTERPRETATION_WINDOW:], reading
                )
            encode_belief(
                self.store, character, belief, interpret=lambda kept=reading: kept or ""
            )

            newest = projected[-1]
            actor_id = newest.event.actor_id
            if actor_id and actor_id != character.id:
                self.store.bump_interaction(character.id, actor_id)
            # Watching someone refuse to answer is the one thing in the
            # engine that deterministically moves how they are regarded —
            # for everyone but the player. Deciding that Elena believes
            # her brother less tonight is telling the person holding her
            # how they feel, which is the same overreach as narrating
            # their actions for them.
            if not character.is_user:
                witnessed_withholding(self.store, character.id, newest)
            # A goal whose subject they have now heard raised is closed.
            # Judged from their own projection: a secret that came out in
            # a room they were not in has not stopped being a secret to
            # them, and they go on guarding it.
            close_reached_goals(self.store, character, projected, self.world)
