"""Pressures: authored complications the director may fire (spec §5.5).

Improvisation lives in the wording and the timing, never in the
invention. Nothing here can produce a complication that an author did
not write down — the director only ever chooses among eligible authored
pressures, and the narrator only ever renders the chosen one. A director
allowed to invent complications produces generic beats (a knock,
thunder, a stranger) with no stake in the world.

Selection is deterministic. The director is the only omniscient
component (invariant 4), so it reads the global log freely here — but
its output stays narrow: a pressure id, never prose.
"""
from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field

from fabula.memory import location_at_seq
from fabula.models import Character, Event, Pressure
from fabula.world import Fact, mentions_fact

ARC_RAMP_TURNS = 30          # how quickly arc mode escalates toward its end
SANDBOX_LULL_THRESHOLD = 0.4  # sandbox only pushes when the scene has gone quiet

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}
_KNOWN_TRIGGER_KEYS = frozenset(
    {"fact_unspoken", "fact_spoken", "for_turns", "turns_elapsed", "character_at", "character_not_at"}
)


@dataclass
class SceneState:
    events: list[Event]
    turn_count: int
    locations: dict[str, str]
    fires: dict[str, list[int]] = field(default_factory=dict)


def scene_state(events: list[Event], characters: dict[str, Character]) -> SceneState:
    turn_count = events[-1].seq if events else 0
    locations = {
        character_id: location_at_seq(
            character_id, character.location_id, events, turn_count + 1
        )
        for character_id, character in characters.items()
    }
    fires: dict[str, list[int]] = {}
    for event in events:
        pressure_id = event.metadata.get("pressure_id")
        if pressure_id:
            fires.setdefault(pressure_id, []).append(event.seq)
    return SceneState(events=events, turn_count=turn_count, locations=locations, fires=fires)


def _first_spoken_seq(fact: Fact, events: list[Event]) -> int | None:
    for event in events:
        if mentions_fact(fact, event.content):
            return event.seq
    return None


def _compare(value: int, expr: str | int) -> bool:
    if isinstance(expr, int):
        return value == expr
    match = re.fullmatch(r"\s*(>=|<=|==|>|<)?\s*(\d+)\s*", expr)
    if not match:
        raise ValueError(f"unparseable comparison in pressure trigger: {expr!r}")
    op = match.group(1) or "=="
    return _OPS[op](value, int(match.group(2)))


def evaluate_trigger(trigger: dict, state: SceneState, facts: dict[str, Fact]) -> bool:
    unknown = set(trigger) - _KNOWN_TRIGGER_KEYS
    if unknown:
        # An unrecognised key would otherwise silently make a pressure
        # always-eligible, which is the worst possible authoring failure.
        raise ValueError(f"unknown pressure trigger key(s): {sorted(unknown)}")

    for_turns = trigger.get("for_turns")

    if "fact_unspoken" in trigger:
        fact = facts.get(trigger["fact_unspoken"])
        if fact is None or _first_spoken_seq(fact, state.events) is not None:
            return False
        if for_turns is not None and not _compare(state.turn_count, for_turns):
            return False

    if "fact_spoken" in trigger:
        fact = facts.get(trigger["fact_spoken"])
        if fact is None:
            return False
        spoken_at = _first_spoken_seq(fact, state.events)
        if spoken_at is None:
            return False
        if for_turns is not None and not _compare(state.turn_count - spoken_at, for_turns):
            return False

    if "turns_elapsed" in trigger and not _compare(state.turn_count, trigger["turns_elapsed"]):
        return False

    for character_id, room in (trigger.get("character_at") or {}).items():
        if state.locations.get(character_id) != room:
            return False

    for character_id, room in (trigger.get("character_not_at") or {}).items():
        if state.locations.get(character_id) == room:
            return False

    return True


def has_ended(end_condition: dict, state: SceneState, facts: dict[str, Fact]) -> bool:
    """Has the scene reached its declared end?

    Deliberately the same evaluator a pressure trigger uses. An author
    writing "this ends when the music box is finally said out loud"
    should not need a second condition language to say it, and the
    vocabulary is already covered by tests.

    An arc without an end condition is just a sandbox that escalates.
    """
    if not end_condition:
        return False
    return evaluate_trigger(end_condition, state, facts)


def is_eligible(pressure: Pressure, state: SceneState, facts: dict[str, Fact]) -> bool:
    fires = state.fires.get(pressure.id, [])
    if len(fires) >= pressure.max_fires:
        return False
    if fires and state.turn_count - fires[-1] < pressure.cooldown_turns:
        return False
    return evaluate_trigger(pressure.trigger, state, facts)


def pressure_desire(
    pressure: Pressure, state: SceneState, mode: str, top_character_bid: float
) -> float:
    """Same pressures, different objective (spec §9). Arc escalates toward
    its end condition, so a pressure competes harder the longer the scene
    runs. Sandbox maintains equilibrium, so it reseeds tension only once
    the characters have stopped generating their own."""
    if mode == "arc":
        ramp = min(1.0, state.turn_count / ARC_RAMP_TURNS)
        return min(1.0, pressure.weight + ramp * 0.5)

    if top_character_bid >= SANDBOX_LULL_THRESHOLD:
        return 0.0
    return pressure.weight


def select_pressure(
    pressures: list[Pressure],
    state: SceneState,
    facts: dict[str, Fact],
    mode: str,
    top_character_bid: float,
) -> tuple[Pressure, float] | None:
    """Highest desire among eligible pressures, ties broken by authored
    order so the same scene state always selects the same pressure."""
    scored = []
    for index, pressure in enumerate(pressures):
        if not is_eligible(pressure, state, facts):
            continue
        desire = pressure_desire(pressure, state, mode, top_character_bid)
        if desire > 0:
            scored.append((desire, -index, pressure))
    if not scored:
        return None
    desire, _, pressure = max(scored, key=lambda triple: (triple[0], triple[1]))
    return pressure, desire
