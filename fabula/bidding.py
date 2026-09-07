"""Candidate prefiltering and bidding (spec §7 steps 3-4).

A bid is computed from exactly the same projection a reply would read —
so a bid can never leak, and a character who didn't perceive the
triggering event is never even asked.
"""
from __future__ import annotations

from fabula.llm import LLMClient
from fabula.memory import ContextBuilder, location_at_seq
from fabula.models import Bid, Character, Event
from fabula.world import World, resolve_perception

AMBIGUOUS_LOW = 0.35
AMBIGUOUS_HIGH = 0.65


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


def heuristic_bid(character: Character, event: Event, level: str) -> tuple[float, str]:
    score = character.traits.talkativeness * 0.5
    reasons: list[str] = []

    if character.id in event.addressed_to:
        score += 0.5
        reasons.append("addressed directly")

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
    score, reason = heuristic_bid(character, event, level)

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
