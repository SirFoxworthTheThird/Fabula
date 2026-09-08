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

from fabula.chronology import render_time_skip
from fabula.db import EventStore
from fabula.llm import LLMClient
from fabula.models import Belief, Character, Event, ProjectedEvent
from fabula.summaries import get_or_create_summary
from fabula.world import STOPWORDS, World, degrade_content, resolve_perception

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

# CJK scripts do not put spaces between words, so a whitespace-and-
# punctuation tokenizer hands back one enormous token per sentence and
# nothing ever overlaps. Character bigrams are not segmentation — a real
# tokenizer would be better — but they give these languages the same kind
# of signal the others get, instead of none at all.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


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


def co_present(
    character: Character, characters: dict[str, Character], events: list[Event]
) -> list[Character]:
    """Who this character can see is here with them.

    Derived by comparing rooms, so it can only ever name someone standing
    in theirs. Being in the same room is the plainest perception there
    is — a character who cannot be told this asks where their sister is
    while looking straight at her.
    """
    last_seq = events[-1].seq if events else 0
    here = location_at_seq(character.id, character.location_id, events, last_seq + 1)
    return [
        other
        for other in characters.values()
        if other.id != character.id
        and location_at_seq(other.id, other.location_id, events, last_seq + 1) == here
    ]


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
        if event.kind == "time_skip":
            # Everyone in the building lives through the same skip, but not
            # the same way: spec §8 makes that explicitly a projection
            # concern, so it is rendered per character here.
            content = render_time_skip(event, character.id, events, world.phrasing)
        elif level == "full":
            content = event.content
        else:
            content = degrade_content(event, world)
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


def _significant_words(text: str, stopwords: frozenset[str] | None = None) -> set[str]:
    r"""The words worth comparing two lines by.

    `\w` rather than `[a-z0-9]`: the ASCII range silently discarded every
    non-Latin script — Cyrillic and Japanese produced the empty set, so
    rehearsal and retrieval could never fire in those languages — and cut
    accented words at the accent, leaving "música" as "sica". `casefold`
    rather than `lower` because that is what non-English casing needs.
    """
    ignore = STOPWORDS if stopwords is None else stopwords
    words = set()
    for token in re.findall(r"[^\W_]+(?:['\u2019][^\W_]+)*", text.casefold()):
        if _CJK.search(token):
            words.update(token[i : i + 2] for i in range(len(token) - 1))
        elif len(token) > 3 and token not in ignore:
            words.add(token)
    return words


def find_rehearsed(
    cue: ProjectedEvent,
    prior: list[ProjectedEvent],
    min_overlap: int = RETRIEVAL_MIN_OVERLAP,
    stopwords: frozenset[str] | None = None,
) -> list[int]:
    """Which earlier perceived events does this new one re-mention?
    Deterministic lexical overlap — a model is never asked, because
    rehearsal feeds tiering and tiering must stay reproducible."""
    cue_words = _significant_words(cue.perceived_content, stopwords)
    if not cue_words:
        return []
    rehearsed = []
    for p in prior:
        if p.event.id is None:
            continue
        if len(cue_words & _significant_words(p.perceived_content, stopwords)) >= min_overlap:
            rehearsed.append(p.event.id)
    return rehearsed


def record_rehearsals(
    character: Character,
    projected: list[ProjectedEvent],
    store: EventStore,
    stopwords: frozenset[str] | None = None,
) -> None:
    """Rehearsal refreshes recency (spec §6 rule 3): an event re-mentioned
    in conversation climbs back up the tiers."""
    if len(projected) < 2:
        return
    cue = projected[-1]
    for event_id in find_rehearsed(cue, projected[:-1], stopwords=stopwords):
        store.record_rehearsal(character.id, event_id, cue.event.seq)


def _effective_age(projected: list[ProjectedEvent], index: int, rehearsals: dict[int, int]) -> int:
    """How many perceived events have happened since this one last
    mattered — its own occurrence, or the last time it was re-mentioned."""
    event = projected[index].event
    effective_seq = event.seq
    if event.id is not None:
        effective_seq = max(effective_seq, rehearsals.get(event.id, effective_seq))
    return sum(1 for p in projected if p.event.seq > effective_seq)


def _lexical_overlap(a: str, b: str, stopwords: frozenset[str] | None = None) -> int:
    return len(_significant_words(a, stopwords) & _significant_words(b, stopwords))


def retrieve(
    cue_text: str,
    candidates: list[TieredEvent],
    top_k: int = RETRIEVAL_TOP_K,
    min_overlap: int = RETRIEVAL_MIN_OVERLAP,
    stopwords: frozenset[str] | None = None,
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
        overlap = _lexical_overlap(cue_text, tiered.projected.perceived_content, stopwords)
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
    language: str = "en",
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
            summary = get_or_create_summary(character, span, scene_id, store, llm, language)
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
    stopwords: frozenset[str] | None = None,
    language: str = "en",
) -> str:
    """Render a projection into the text a model call sees. Only
    `perceived_content` is ever read — never `event.content` — so a
    degraded or absent event has no path to leaking its true content."""
    carried = _carried_in(character, scene_id, store)
    if not projected:
        # Nothing has happened yet in *this* scene, which is not the same
        # as an empty head: a character walking in from last week arrives
        # already believing things.
        return f"{carried}\n(nothing has happened yet)" if carried else "(nothing has happened yet)"

    rehearsals = store.get_rehearsals(character.id)
    tiered = assign_tiers(character, projected, rehearsals)

    cue = projected[-1].perceived_content
    for index in retrieve(cue, tiered, stopwords=stopwords):
        tiered[index] = tiered[index].model_copy(update={"tier": "verbatim", "exempt": True})

    tiered = _fit_to_budget(tiered, budget)
    body = _render(character, tiered, scene_id, store, llm, language)
    return f"{carried}\n{body}" if carried else body


def _carried_in(character: Character, scene_id: str, store: EventStore) -> str:
    """What this character walks in already believing.

    Every line here was encoded from their own projection in an earlier
    scene, which is what makes it safe to read back: a memory Maria never
    formed cannot appear, and one she formed from half-hearing something
    says what she half-heard.

    The echo comes first and the reading is appended to it, never
    substituted for it. A reading is the softest thing in the engine —
    written by a model, and only as good as the model — so letting it
    stand in place of what was perceived would let a weak one quietly
    delete a memory instead of colouring it.
    """
    remembered = store.get_beliefs_from_other_scenes(character.id, scene_id)
    if not remembered:
        return ""
    # Two moments can leave the same trace; a character does not believe
    # it twice as hard for having thought it twice.
    seen: dict[str, str] = {}
    for belief in remembered:
        seen.setdefault(belief.content, belief.interpretation)
    lines = "\n".join(
        f"- {content}" + (f" (you took it as: {reading})" if reading else "")
        for content, reading in seen.items()
    )
    return f"What you already believed, coming into this:\n{lines}\n"


class ContextBuilder:
    """The single path from log to context.

    Bids and replies both go through this object, so a bid provably reads
    exactly what the reply would read (spec §7 step 4) — a bid cannot leak
    something a reply wouldn't, because it is the same assembly.
    """

    def __init__(
        self,
        world: World,
        characters: dict[str, Character],
        scene_id: str,
        store: EventStore,
        llm: LLMClient,
        budget: int = DEFAULT_TOKEN_BUDGET,
    ):
        self.world = world
        self.characters = characters
        self.scene_id = scene_id
        self.store = store
        self.llm = llm
        self.budget = budget

    def project(self, character: Character, events: list[Event]) -> list[ProjectedEvent]:
        return project(character, events, self.world, initial_location=character.location_id)

    def situation(self, character: Character, events: list[Event]) -> str:
        """Where they are and who is with them.

        Only same-room company is ever named, so this states nothing the
        character could not see by looking up.
        """
        last_seq = events[-1].seq if events else 0
        here = location_at_seq(character.id, character.location_id, events, last_seq + 1)
        others = co_present(character, self.characters, events)
        company = (
            "With you: " + ", ".join(sorted(other.name for other in others)) + "."
            if others
            else "You are alone."
        )
        return f"(Right now: you are in {self.world.room_name(here)}. {company})"

    def for_character(self, character: Character, events: list[Event]) -> str:
        body = assemble_context(
            character,
            self.project(character, events),
            self.scene_id,
            self.store,
            self.llm,
            self.budget,
            self.world.phrasing.stopwords,
            self.world.language,
        )
        # Last, not first. The events above are history; this is the state
        # the character is standing in, and it belongs next to the question
        # being asked of them. Buried at the top of a grown scene it gets
        # contradicted — a character insisted the player was alone in a
        # room she was standing in with two other people.
        return f"{body}\n{self.situation(character, events)}"


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
