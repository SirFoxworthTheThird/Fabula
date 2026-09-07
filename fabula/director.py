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
from fabula.chronology import (
    LARGE_SKIP_MINUTES,
    derive_skip_minutes,
    describe_duration,
    due_intentions,
)
from fabula.db import EventStore
from fabula.loader import Scene
from fabula.memory import ContextBuilder, form_belief, location_at_seq, record_rehearsals
from fabula.persistence import encode_belief
from fabula.models import Bid, Character, Event, Pressure
from fabula.narrator import NARRATOR_ID, Narrator
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
    ):
        self.store = store
        self.world = world
        self.characters = characters
        self.scene = scene
        self.narrator = narrator
        self.llm = llm
        self.pressures = pressures or []
        self.contexts = ContextBuilder(world, scene.id, store, llm)

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
    ) -> Event:
        all_events = self.store.get_events(self.scene.id)
        if story_time is None:
            story_time = all_events[-1].story_time if all_events else self.scene.start_time
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
        self._record_rehearsals()
        events_this_turn = [stored_user_event]
        last_event = stored_user_event
        consecutive_agent_turns = 0

        for _ in range(self.scene.turn_budget):
            all_events = self.store.get_events(self.scene.id)
            candidates = prefilter_candidates(last_event, self.characters, all_events, self.world)

            bids: list[Bid] = []
            for character in candidates:
                location = location_at_seq(
                    character.id, character.location_id, all_events, last_event.seq + 1
                )
                level = resolve_perception(last_event, character.id, location, self.world)
                bids.append(get_bid(character, last_event, level, all_events, self.contexts, self.llm))

            narrator_bid = self.narrator.bid(last_event, all_events, self.world)
            pressure_choice = select_pressure(
                self.pressures,
                scene_state(all_events, self.characters),
                self.world.facts,
                self.scene.mode,
                max((b.desire for b in bids), default=0.0),
            )
            decision = arbitrate(
                bids,
                narrator_bid,
                consecutive_agent_turns,
                self.scene.max_consecutive_agent_turns,
                pressure_choice,
            )

            if decision.kind == "yield_to_user":
                break

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
                content = self.narrator.generate(last_event, all_events, self.world)
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
            self._record_rehearsals()
            consecutive_agent_turns += 1

        return events_this_turn

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
            minutes = derive_skip_minutes(self.characters, all_events)
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
            f"{describe_duration(minutes)} pass.",
            audibility="building",
            metadata={"minutes": minutes},
            story_time=previous_time + timedelta(minutes=minutes),
        )
        appended = [self.store.append_event(skip)]
        appended.extend(self._resolve_offscreen(minutes))
        self._record_rehearsals()
        return appended

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
        return appended

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
        self._record_rehearsals()
        return appended

    def unmaterialized_here(self, observer: Character) -> list[Event]:
        """Coarse off-screen events waiting in the observer's room."""
        all_events = self.store.get_events(self.scene.id)
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
        content = self.narrator.render_pressure(pressure, location_id, self.world)
        return self.build_event(
            effect["kind"],
            effect.get("actor"),
            location_id,
            content,
            audibility=effect.get("audibility", "room"),
            metadata={"pressure_id": pressure.id},
        )

    def _record_rehearsals(self) -> None:
        """After each append: note which of a character's own earlier
        perceived events the newest one re-mentions, and encode the newest
        one as a belief if it was salient enough to keep.

        Both are done per character against their own projection, so an
        event they could not perceive is neither rehearsed nor
        remembered — it does not exist for them.
        """
        all_events = self.store.get_events(self.scene.id)
        for character in self.characters.values():
            projected = self.contexts.project(character, all_events)
            record_rehearsals(character, projected, self.store)
            if not projected:
                continue

            newest = projected[-1]
            encode_belief(self.store, character, form_belief(character, newest))

            actor_id = newest.event.actor_id
            if actor_id and actor_id != character.id:
                self.store.bump_interaction(character.id, actor_id)
