"""Story time (spec §8).

Time advances in narrative jumps, not wall clock, and there is no
background job ticking the world. Every jump is committed to the log as
a `time_skip` event with an explicit duration — projections replay over
it and decay reads off it, so an unlogged jump is the same class of bug
as an unlogged utterance.

Nothing here calls a model. The size of a jump is derived from what
characters intend to do; the prose about it is the narrator's job.
"""
from __future__ import annotations

from fabula.models import Character, Event, Intention

LARGE_SKIP_MINUTES = 60  # beyond this, the user is asked before time moves


def resolved_intention_ids(events: list[Event]) -> set[str]:
    return {
        event.metadata["intention_id"]
        for event in events
        if event.metadata.get("intention_id")
    }


def pending_intentions(character: Character, events: list[Event]) -> list[Intention]:
    resolved = resolved_intention_ids(events)
    return [i for i in character.intentions if i.id not in resolved]


def derive_skip_minutes(characters: dict[str, Character], events: list[Event]) -> int | None:
    """The next moment something worth discovering is ready.

    Deliberately the *minimum* pending readiness across the cast: skipping
    further would step over a moment someone was about to reach, and
    skipping to nothing is how you get an arbitrary "three hours pass"
    that lands on an empty room.
    """
    ready_at = [
        intention.ready_after_minutes
        for character in characters.values()
        if not character.is_user
        for intention in pending_intentions(character, events)
    ]
    return min(ready_at) if ready_at else None


def due_intentions(
    character: Character, events: list[Event], elapsed_minutes: int
) -> list[Intention]:
    return [
        intention
        for intention in pending_intentions(character, events)
        if intention.ready_after_minutes <= elapsed_minutes
    ]


def was_asleep(character_id: str, events: list[Event], before_seq: int) -> bool:
    """Sleep is a logged state, like everything else — the latest
    state_change for this character before the given point decides."""
    asleep = False
    for event in events:
        if event.seq >= before_seq:
            break
        if event.kind != "state_change" or event.metadata.get("character_id") != character_id:
            continue
        state = event.metadata.get("state")
        if state == "asleep":
            asleep = True
        elif state == "awake":
            asleep = False
    return asleep


def describe_duration(minutes: int) -> str:
    if minutes >= 60 and minutes % 60 == 0:
        hours = minutes // 60
        return "an hour" if hours == 1 else f"{hours} hours"
    if minutes == 1:
        return "a minute"
    return f"{minutes} minutes"


def render_time_skip(event: Event, character_id: str, events: list[Event]) -> str:
    """Elapsed time is perceived non-uniformly (spec §8). Someone asleep
    gets a discontinuity; someone awake and waiting lived every hour of
    it. This is a projection concern like any other, so it is a
    deterministic template, never a model call."""
    duration = describe_duration(int(event.metadata.get("minutes", 0)))
    if was_asleep(character_id, events, event.seq):
        return f"(a gap — you surface to find {duration} gone, unfelt)"
    return f"({duration} pass, and you feel every one of them)"
