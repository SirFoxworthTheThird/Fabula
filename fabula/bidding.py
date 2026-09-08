"""Candidate prefiltering and bidding (spec §7 steps 3-4).

A bid is computed from exactly the same projection a reply would read —
so a bid can never leak, and a character who didn't perceive the
triggering event is never even asked.
"""
from __future__ import annotations

from fabula.db import EventStore
from fabula.llm import LLMClient
from fabula.memory import ContextBuilder, location_at_seq
from fabula.models import Bid, Character, Event, Relationship
from fabula.persistence import unresolved_goals
from fabula.world import World, mentions_fact, resolve_perception

AMBIGUOUS_LOW = 0.35
AMBIGUOUS_HIGH = 0.65

WITHHOLD_RETICENCE = 0.6   # below this, a character just answers
WITHHOLD_COOLDOWN = 6      # events; deflecting every turn stops being drama

GOAL_STAKE = 0.45          # a goal at full priority is worth about this much
NEUTRAL_TRUST = 0.5        # the authored default: no reason either way
DISTRUST_ATTENTION = 0.4   # at most +0.2, so it colours a bid, never decides it


def recently_withheld(character_id: str, events: list[Event], window: int = WITHHOLD_COOLDOWN) -> bool:
    if not events:
        return False
    cutoff = events[-1].seq - window
    return any(
        event.seq > cutoff
        and event.actor_id == character_id
        and event.metadata.get("withheld")
        for event in events
    )


def withholding_bid(
    character: Character, event: Event, level: str, events: list[Event], world: World
) -> tuple[float, str] | None:
    """Bid to visibly not answer, when pressed directly on something this
    character protects.

    Only on `full` perception, which is what makes reading `event.content`
    here safe: at full fidelity the perceived content *is* the content, so
    this asks nothing more than what the character actually heard. A
    degraded listener did not catch the question clearly enough to dodge it.
    """
    # Named directly, or asked of the room at large (spec §5.1: an empty
    # addressed_to is the whole room). A question about your secret that
    # names someone else is not yours to dodge.
    addressed = character.id in event.addressed_to or not event.addressed_to
    if level != "full" or not addressed:
        return None
    if character.traits.reticence < WITHHOLD_RETICENCE:
        return None
    if recently_withheld(character.id, events):
        return None

    for fact_id in character.protects:
        fact = world.facts.get(fact_id)
        if fact is not None and mentions_fact(fact, event.content):
            desire = min(1.0, 0.6 + character.traits.reticence * 0.35)
            return desire, "pressed on something they will not discuss"
    return None


def prefilter_candidates(
    event: Event, characters: dict[str, Character], events: list[Event], world: World
) -> list[Character]:
    """Cheap heuristic prefilter: only characters who perceived the
    triggering event, aren't its author, and aren't the user (the user
    acts via stdin, not bids)."""
    candidates = []
    for character in characters.values():
        if character.is_user or character.id == event.actor_id:
            continue
        location = location_at_seq(character.id, character.location_id, events, event.seq + 1)
        level = resolve_perception(event, character.id, location, world)
        if level == "none":
            continue
        candidates.append(character)
    return candidates


def goal_at_stake(
    character: Character, event: Event, level: str, world: World, store: EventStore
) -> float:
    """The priority of the most pressing open goal this event touches.

    Only at `full` perception, which is what makes reading `event.content`
    here safe — at full fidelity the perceived content *is* the content,
    so this asks nothing beyond what the character actually heard. A goal
    only counts if it names a fact: prose alone cannot be matched.
    """
    if level != "full":
        return 0.0
    priorities = [
        goal.priority
        for goal in unresolved_goals(store, character)
        if goal.about
        and goal.about in world.facts
        and mentions_fact(world.facts[goal.about], event.content)
    ]
    return max(priorities, default=0.0)


def heuristic_bid(
    character: Character,
    event: Event,
    level: str,
    regard: Relationship | None = None,
    stake: float = 0.0,
) -> tuple[float, str]:
    score = character.traits.talkativeness * 0.5
    reasons: list[str] = []

    if character.id in event.addressed_to:
        score += 0.5
        reasons.append("addressed directly")

    if stake:
        # Somebody is talking about the thing you are still trying to do
        # something about. Wanting is not the same as reacting: reactivity
        # is a reflex, this is a standing want that outlives the moment
        # and stops mattering once the goal is closed.
        score += stake * GOAL_STAKE
        reasons.append("this touches what they want")

    if regard is not None and regard.trust < NEUTRAL_TRUST:
        # Distrust is attention. You watch the person you have stopped
        # believing, and you are quicker to speak into what they say —
        # which is what gives a withheld beat a cost beyond the pause the
        # narrator renders for it.
        score += (NEUTRAL_TRUST - regard.trust) * DISTRUST_ATTENTION
        reasons.append("does not trust them")

    for trigger in event.metadata.get("triggers", []):
        weight = character.traits.reactivity.get(trigger, 0.0)
        if weight:
            score += weight * 0.4
            reasons.append(f"reacts to {trigger}")

    if level == "degraded":
        score *= 0.7
        reasons.append("perception degraded")

    score = max(0.0, min(1.0, score))
    reason = "; ".join(reasons) if reasons else "baseline talkativeness"
    return score, reason


def _parse_llm_bid(raw: str, fallback_desire: float, fallback_reason: str) -> tuple[float, str]:
    """Best-effort parse of a "<number> | <reason>" response. Falls back to
    the heuristic score on anything malformed (e.g. FakeLLM's placeholder
    text) rather than raising — a bid is never worth crashing the loop."""
    try:
        number_part, _, reason_part = raw.partition("|")
        desire = max(0.0, min(1.0, float(number_part.strip())))
        reason = reason_part.strip() or fallback_reason
        return desire, reason
    except (ValueError, TypeError):
        return fallback_desire, fallback_reason


def get_bid(
    character: Character,
    event: Event,
    level: str,
    events: list[Event],
    contexts: ContextBuilder,
    llm: LLMClient | None = None,
) -> Bid:
    """Compute one character's bid to speak. Resolves obvious cases by
    heuristic alone; only spends a model call when the heuristic score
    lands in the ambiguous band, per spec §7.

    The ambiguous path builds context through the same `ContextBuilder`
    the reply uses, so a bid reads exactly what a reply would read."""
    withhold = withholding_bid(character, event, level, events, contexts.world)
    if withhold is not None:
        desire, reason = withhold
        return Bid(
            character_id=character.id, desire=desire, one_line_reason=reason, kind="withhold"
        )

    # How this character regards whoever just acted. Their own stored
    # state, so consulting it cannot tell them anything new about the room.
    regard = (
        contexts.store.get_relationships(character.id).get(event.actor_id)
        if event.actor_id
        else None
    )
    stake = goal_at_stake(character, event, level, contexts.world, contexts.store)
    score, reason = heuristic_bid(character, event, level, regard, stake)

    if llm is not None and AMBIGUOUS_LOW <= score <= AMBIGUOUS_HIGH:
        context = contexts.for_character(character, events)
        system = (
            f"You are {character.name}. {character.persona}\n"
            "You are deciding whether to speak or act right now, not writing a line yet."
        )
        prompt = (
            f"What you know so far:\n{context}\n\n"
            "How much do you want to speak or act right now, from 0.0 (not at all) to "
            "1.0 (urgently)? Respond exactly as: <number> | <one line reason>"
        )
        raw = llm.complete(system=system, prompt=prompt, key=character.id)
        score, reason = _parse_llm_bid(raw, fallback_desire=score, fallback_reason=reason)

    return Bid(character_id=character.id, desire=score, one_line_reason=reason)
