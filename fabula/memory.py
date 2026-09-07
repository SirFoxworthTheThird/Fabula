"""The memory projector: builds a character's context from the log.

M0/M1 scope: filtering by perception and assembling a prompt-ready
projection, plus minimal salience-scored belief formation. Tiering,
decay, summarization, and retrieval (spec §6) land in M2 — this module
is deliberately the simplest thing that still makes invariant 1 and the
divergence tests true, since those must hold from the first milestone.

The store is not the context: `project()` derives a filtered view of the
log; `assemble_context()` renders that view into the text actually sent
to a model. Nothing here calls a model.
"""
from __future__ import annotations

from fabula.models import Belief, Character, Event, ProjectedEvent
from fabula.world import World, degrade_content, resolve_perception


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


def assemble_context(character: Character, projected: list[ProjectedEvent]) -> str:
    """Render a projection into the text block a model call would see.
    Only `perceived_content` is used — never `event.content` — so a
    degraded or absent event has no path to leaking its true content."""
    if not projected:
        return "(nothing has happened yet)"
    lines = []
    for p in projected:
        who = p.event.actor_id or "the world"
        tag = "" if p.perception == "full" else " (unclear)"
        lines.append(f"[{p.event.kind}{tag}] {who}: {p.perceived_content}")
    return "\n".join(lines)


def form_belief(character: Character, projected: ProjectedEvent) -> Belief | None:
    """Minimal belief encoding: salience is scored at encode time from the
    event's base salience times this character's own salience_bias, per
    spec §6 rule 1 — this is what lets two characters form measurably
    different beliefs about the same event."""
    if projected.perception == "none":
        return None
    bias = character.traits.salience_bias.get(projected.event.kind, 1.0)
    salience = max(0.0, min(1.0, projected.event.salience_base * bias))
    confidence = 1.0 if projected.perception == "full" else 0.5
    return Belief(
        character_id=character.id,
        subject_id=projected.event.actor_id or "world",
        content=projected.perceived_content,
        confidence=confidence,
        source_event_id=projected.event.id,
        formed_at=projected.event.story_time,
        last_rehearsed=projected.event.story_time,
        salience=salience,
    )
