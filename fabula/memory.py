"""The memory projector: builds a character's context from the log.

The store is not the context. `project()` derives the filtered view of
the log a character is entitled to; `assemble_context()` sizes that view
to a token budget by tiering it into verbatim / summarized / gist
(spec §6). Only summarization calls a model — everything else here is
deterministic, because invariant 5 (reproducible projections) is what
makes the leak tests meaningful.

Every function below that could reach story content takes
`ProjectedEvent`s, never raw `Event`s. That is deliberate: there is no
path from tiering, retrieval, or summarization back to the global log.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from fabula.db import EventStore
from fabula.llm import LLMClient
from fabula.models import Belief, Character, Event, ProjectedEvent
from fabula.summaries import get_or_create_summary
from fabula.world import World, degrade_content, resolve_perception

Tier = Literal["verbatim", "summarized", "gist"]

VERBATIM_WINDOW = 12       # most recent N perceived events stay full-text
HIGH_SALIENCE = 0.7        # at or above this, never demoted (spec §6 rule 2)
GIST_AGE = 40              # older than this many perceived events -> one-line trace
SUMMARY_SPAN = 8           # mid-range events compacted this many at a time
SUMMARY_SPAN_TOKENS = 40   # rough size of one rendered summary line
DEFAULT_TOKEN_BUDGET = 1200
RETRIEVAL_TOP_K = 2
RETRIEVAL_MIN_OVERLAP = 2
GIST_CHARS = 60

_STOPWORDS = frozenset(
    """
    about after again against because been before being between both could
    does doing during each from have having here into itself just more most
    only other over same should some such than that their them then there
    these they this those through under until very were what when where
    which while with would your yours
    """.split()
)


class TieredEvent(BaseModel):
    projected: ProjectedEvent
    tier: Tier
    salience: float
    exempt: bool = False  # high salience or retrieved: never demoted by budget


def location_at_seq(
    character_id: str, initial_location: str, events: list[Event], seq: int
) -> str:
    """Where was this character when the event at `seq` occurred? Replays
    the character's own arrival events up to (not including) `seq`, so
    perception is judged against their location at the time, not wherever
    they are now."""
    location = initial_location
    for e in events:
        if e.seq >= seq:
            break
        if e.actor_id == character_id and e.kind == "arrival":
            location = e.location_id
    return location


def project(
    character: Character,
    events: list[Event],
    world: World,
    initial_location: str | None = None,
) -> list[ProjectedEvent]:
    """Filter the full event log down to what this character perceived,
    at whatever fidelity. This is the single choke point invariant 1
    depends on at read time: an event with perception "none" is dropped
    here and never reaches anything downstream."""
    start_location = initial_location if initial_location is not None else character.location_id
    projected: list[ProjectedEvent] = []
    for event in events:
        char_location = location_at_seq(character.id, start_location, events, event.seq)
        level = resolve_perception(event, character.id, char_location, world)
        if level == "none":
            continue
        content = event.content if level == "full" else degrade_content(event, world)
        projected.append(ProjectedEvent(event=event, perceived_content=content, perception=level))
    return projected


def score_salience(character: Character, projected: ProjectedEvent) -> float:
    """Salience is scored at encode time, from the event's base salience
    times this character's own bias (spec §6 rule 1). The jealous
    character rates an ambiguous glance higher than the placid one does."""
    bias = character.traits.salience_bias.get(projected.event.kind, 1.0)
    return max(0.0, min(1.0, projected.event.salience_base * bias))


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _significant_words(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9']+", text.lower())
        if len(word) > 3 and word not in _STOPWORDS
    }


def find_rehearsed(
    cue: ProjectedEvent, prior: list[ProjectedEvent], min_overlap: int = RETRIEVAL_MIN_OVERLAP
) -> list[int]:
    """Which earlier perceived events does this new one re-mention?
    Deterministic lexical overlap — a model is never asked, because
    rehearsal feeds tiering and tiering must stay reproducible."""
    cue_words = _significant_words(cue.perceived_content)
    if not cue_words:
        return []
    rehearsed = []
    for p in prior:
        if p.event.id is None:
            continue
        if len(cue_words & _significant_words(p.perceived_content)) >= min_overlap:
            rehearsed.append(p.event.id)
    return rehearsed


def record_rehearsals(character: Character, projected: list[ProjectedEvent], store: EventStore) -> None:
    """Rehearsal refreshes recency (spec §6 rule 3): an event re-mentioned
    in conversation climbs back up the tiers."""
    if len(projected) < 2:
        return
    cue = projected[-1]
    for event_id in find_rehearsed(cue, projected[:-1]):
        store.record_rehearsal(character.id, event_id, cue.event.seq)


def _effective_age(projected: list[ProjectedEvent], index: int, rehearsals: dict[int, int]) -> int:
    """How many perceived events have happened since this one last
    mattered — its own occurrence, or the last time it was re-mentioned."""
    event = projected[index].event
    effective_seq = event.seq
    if event.id is not None:
        effective_seq = max(effective_seq, rehearsals.get(event.id, effective_seq))
    return sum(1 for p in projected if p.event.seq > effective_seq)


def _lexical_overlap(a: str, b: str) -> int:
    return len(_significant_words(a) & _significant_words(b))


def retrieve(
    cue_text: str,
    candidates: list[TieredEvent],
    top_k: int = RETRIEVAL_TOP_K,
    min_overlap: int = RETRIEVAL_MIN_OVERLAP,
) -> set[int]:
    """The retrieval pass (spec §6 rule 4): pull an old detail back into
    context when the current scene cues it.

    `candidates` are already-projected events belonging to this character,
    so the match is against their own history and there is no way to reach
    the global log from here.
    """
    scored = []
    for index, tiered in enumerate(candidates):
        if tiered.tier == "verbatim":
            continue
        overlap = _lexical_overlap(cue_text, tiered.projected.perceived_content)
        if overlap >= min_overlap:
            scored.append((overlap, index))
    # Highest overlap first, then most recent, so ties resolve deterministically.
    scored.sort(key=lambda pair: (-pair[0], -pair[1]))
    return {index for _, index in scored[:top_k]}


def assign_tiers(
    character: Character, projected: list[ProjectedEvent], rehearsals: dict[int, int]
) -> list[TieredEvent]:
    tiered: list[TieredEvent] = []
    for index, p in enumerate(projected):
        salience = score_salience(character, p)
        age = _effective_age(projected, index, rehearsals)
        if salience >= HIGH_SALIENCE:
            tier: Tier = "verbatim"
            exempt = True
        elif age < VERBATIM_WINDOW:
            tier, exempt = "verbatim", False
        elif age < GIST_AGE:
            tier, exempt = "summarized", False
        else:
            tier, exempt = "gist", False
        tiered.append(TieredEvent(projected=p, tier=tier, salience=salience, exempt=exempt))
    return tiered


def _gist(projected: ProjectedEvent) -> str:
    who = projected.event.actor_id or "the world"
    text = projected.perceived_content
    if len(text) > GIST_CHARS:
        text = text[:GIST_CHARS].rstrip() + "…"
    return f"{who}: {text}"


def _render(
    character: Character,
    tiered: list[TieredEvent],
    scene_id: str,
    store: EventStore,
    llm: LLMClient,
) -> str:
    lines: list[str] = []
    index = 0
    while index < len(tiered):
        entry = tiered[index]

        if entry.tier == "summarized":
            span_end = index
            while (
                span_end < len(tiered)
                and tiered[span_end].tier == "summarized"
                and span_end - index < SUMMARY_SPAN
            ):
                span_end += 1
            span = [t.projected for t in tiered[index:span_end]]
            summary = get_or_create_summary(character, span, scene_id, store, llm)
            lines.append(f"[earlier, {len(span)} moments] {summary}")
            index = span_end
            continue

        if entry.tier == "gist":
            lines.append(f"[long ago] {_gist(entry.projected)}")
        else:
            p = entry.projected
            who = p.event.actor_id or "the world"
            tag = "" if p.perception == "full" else " (unclear)"
            lines.append(f"[{p.event.kind}{tag}] {who}: {p.perceived_content}")
        index += 1

    return "\n".join(lines)


def _entry_size(entry: TieredEvent) -> int:
    """Estimated tokens this entry costs once rendered at its tier — what
    the budget is actually spent on.

    Note the ordering this implies: a summarized event is the *cheapest*,
    because a whole span collapses into one line, while a gist still costs
    one capped line each. Gist is where age puts an event, not where
    budget pressure does.
    """
    if entry.tier == "verbatim":
        return estimate_tokens(entry.projected.perceived_content)
    if entry.tier == "gist":
        return estimate_tokens(_gist(entry.projected))
    return max(1, SUMMARY_SPAN_TOKENS // SUMMARY_SPAN)


def _fit_to_budget(tiered: list[TieredEvent], budget: int) -> list[TieredEvent]:
    """Compact, then forget, until the projection fits. High-salience and
    retrieved events are exempt: a betrayal from twenty scenes ago stays
    sharp even when the budget is tight (spec §6 rule 2)."""
    def size(entries: list[TieredEvent]) -> int:
        return sum(_entry_size(e) for e in entries)

    working = list(tiered)

    # Compact first, oldest verbatim first: full text is the expensive tier.
    for index, entry in enumerate(working):
        if size(working) <= budget:
            return working
        if entry.exempt or entry.tier != "verbatim":
            continue
        working[index] = entry.model_copy(update={"tier": "summarized"})

    # Only then forget, oldest first — everything left is already a line
    # or less, so there is nothing further to compact.
    while size(working) > budget:
        droppable = next((i for i, e in enumerate(working) if not e.exempt), None)
        if droppable is None:
            break
        working.pop(droppable)

    return working


def assemble_context(
    character: Character,
    projected: list[ProjectedEvent],
    scene_id: str,
    store: EventStore,
    llm: LLMClient,
    budget: int = DEFAULT_TOKEN_BUDGET,
) -> str:
    """Render a projection into the text a model call sees. Only
    `perceived_content` is ever read — never `event.content` — so a
    degraded or absent event has no path to leaking its true content."""
    if not projected:
        return "(nothing has happened yet)"

    rehearsals = store.get_rehearsals(character.id)
    tiered = assign_tiers(character, projected, rehearsals)

    cue = projected[-1].perceived_content
    for index in retrieve(cue, tiered):
        tiered[index] = tiered[index].model_copy(update={"tier": "verbatim", "exempt": True})

    tiered = _fit_to_budget(tiered, budget)
    return _render(character, tiered, scene_id, store, llm)


class ContextBuilder:
    """The single path from log to context.

    Bids and replies both go through this object, so a bid provably reads
    exactly what the reply would read (spec §7 step 4) — a bid cannot leak
    something a reply wouldn't, because it is the same assembly.
    """

    def __init__(
        self,
        world: World,
        scene_id: str,
        store: EventStore,
        llm: LLMClient,
        budget: int = DEFAULT_TOKEN_BUDGET,
    ):
        self.world = world
        self.scene_id = scene_id
        self.store = store
        self.llm = llm
        self.budget = budget

    def project(self, character: Character, events: list[Event]) -> list[ProjectedEvent]:
        return project(character, events, self.world, initial_location=character.location_id)

    def for_character(self, character: Character, events: list[Event]) -> str:
        return assemble_context(
            character,
            self.project(character, events),
            self.scene_id,
            self.store,
            self.llm,
            self.budget,
        )


def form_belief(character: Character, projected: ProjectedEvent) -> Belief | None:
    """Minimal belief encoding, scored at encode time — this is what lets
    two characters form measurably different beliefs about one event."""
    if projected.perception == "none":
        return None
    confidence = 1.0 if projected.perception == "full" else 0.5
    return Belief(
        character_id=character.id,
        subject_id=projected.event.actor_id or "world",
        content=projected.perceived_content,
        confidence=confidence,
        source_event_id=projected.event.id,
        formed_at=projected.event.story_time,
        last_rehearsed=projected.event.story_time,
        salience=score_salience(character, projected),
    )
