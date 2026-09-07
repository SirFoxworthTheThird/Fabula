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
from typing import Literal

from fabula.agents import generate_utterance
from fabula.bidding import get_bid, prefilter_candidates
from fabula.db import EventStore
from fabula.loader import Scene
from fabula.memory import ContextBuilder, location_at_seq, record_rehearsals
from fabula.models import Bid, Character, Event
from fabula.narrator import NARRATOR_ID, Narrator
from fabula.world import World, resolve_perception

DecisionKind = Literal["speak", "narrate", "yield_to_user"]

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
    bids: list[Bid] = field(default_factory=list)


def arbitrate(
    bids: list[Bid],
    narrator_bid: Bid | None,
    consecutive_agent_turns: int,
    max_consecutive_agent_turns: int,
) -> Decision:
    """Guard first: a hard cap on consecutive non-user turns, so
    agent-to-agent exchange always terminates and the user is never
    starved out of their own scene (spec §7 guards)."""
    if consecutive_agent_turns >= max_consecutive_agent_turns:
        return Decision(kind="yield_to_user", bids=bids)

    contenders = list(bids)
    if narrator_bid is not None:
        contenders.append(narrator_bid)
    if not contenders:
        return Decision(kind="yield_to_user", bids=bids)

    winner = max(contenders, key=lambda b: b.desire)
    if winner.desire < YIELD_FLOOR:
        return Decision(kind="yield_to_user", bids=bids)
    if winner.character_id == NARRATOR_ID:
        return Decision(kind="narrate", character_id=NARRATOR_ID, bids=bids)
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
    ):
        self.store = store
        self.world = world
        self.characters = characters
        self.scene = scene
        self.narrator = narrator
        self.llm = llm
        self.contexts = ContextBuilder(world, scene.id, store, llm)

    def build_event(self, kind: str, actor_id: str | None, location_id: str, content: str) -> Event:
        all_events = self.store.get_events(self.scene.id)
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
            audibility="room",
            addressed_to=addressed_to,
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
            decision = arbitrate(bids, narrator_bid, consecutive_agent_turns, self.scene.max_consecutive_agent_turns)

            if decision.kind == "yield_to_user":
                break

            if decision.kind == "narrate":
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

    def _record_rehearsals(self) -> None:
        """After each append, note which of a character's own earlier
        perceived events the newest one re-mentions. Done per character
        against their own projection, so rehearsal never reveals that an
        event they cannot perceive was referenced at all."""
        all_events = self.store.get_events(self.scene.id)
        for character in self.characters.values():
            record_rehearsals(character, self.contexts.project(character, all_events), self.store)
