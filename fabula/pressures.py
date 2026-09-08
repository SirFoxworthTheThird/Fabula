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
from fabula.world import Fact, World, mentions_fact, resolve_perception

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
    # Needed to answer "did anyone actually hear that?", which is what
    # separates a subject being raised from a word appearing in the log.
    characters: dict[str, Character] = field(default_factory=dict)
    world: World | None = None


def scene_state(
    events: list[Event], characters: dict[str, Character], world: World | None = None
) -> SceneState:
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
    return SceneState(
        events=events,
        turn_count=turn_count,
        locations=locations,
        fires=fires,
        characters=characters,
        world=world,
    )


def _anyone_heard(event: Event, state: SceneState) -> bool:
    """Did anyone but the actor perceive this event in full?

    Full only. The degraded descriptor carries no words — "muffled voices
    from the kitchen" — so someone through a wall did not hear what the
    subject was.
    """
    if state.world is None:
        return True  # no topology to judge with; fall back to the log
    for character_id, character in state.characters.items():
        if character_id == event.actor_id:
            continue
        where = location_at_seq(character_id, character.location_id, state.events, event.seq)
        if resolve_perception(event, character_id, where, state.world) == "full":
            return True
    return False


def _first_spoken_seq(fact: Fact, state: SceneState) -> int | None:
    """When was this fact actually raised in front of somebody?

    Not merely "appears somewhere in the log". An event nobody perceived
    did not put the subject in the room, and the difference is not
    academic: Tomás's authored off-screen intention names the music box
    in its own action text, so checking the glue alone in an empty
    kitchen used to satisfy `fact_spoken` and end an arc that was waiting
    for someone to say it out loud. Found by playing it.
    """
    for event in state.events:
        if event.metadata.get("transmission"):
            # Somebody reporting that they told a person off-screen, once,
            # in private. The subject reached one pair of ears in the past
            # tense; it is not the same as it being said in the room, and
            # an arc waiting for somebody to say it out loud is still
            # waiting.
            continue
        if mentions_fact(fact, event.content) and _anyone_heard(event, state):
            return event.seq
    return None


def _named_facts(value, facts: dict[str, Fact]) -> list[Fact] | None:
    """Resolve one fact id or a list of them.

    A story with two people each holding something cannot say what it
    needs to with a single id — "over once both are in the room" is the
    whole shape of it. A bare string still means one fact, so nothing
    authored against the old vocabulary changes meaning.

    None means an id that does not exist, which is an authoring mistake
    and must not quietly satisfy anything.
    """
    ids = [value] if isinstance(value, str) else list(value)
    resolved = [facts.get(fact_id) for fact_id in ids]
    return None if not resolved or any(f is None for f in resolved) else resolved


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
        # None of the named facts has been said. A list is "all still
        # unsaid": one of them surfacing is enough to stop holding.
        unspoken = _named_facts(trigger["fact_unspoken"], facts)
        if unspoken is None:
            return False
        if any(_first_spoken_seq(fact, state) is not None for fact in unspoken):
            return False
        if for_turns is not None and not _compare(state.turn_count, for_turns):
            return False

    if "fact_spoken" in trigger:
        # All of the named facts have been said. `for_turns` counts from
        # the last of them to land, which is when the condition actually
        # became true.
        spoken = _named_facts(trigger["fact_spoken"], facts)
        if spoken is None:
            return False
        seqs = [_first_spoken_seq(fact, state) for fact in spoken]
        if any(seq is None for seq in seqs):
            return False
        if for_turns is not None and not _compare(state.turn_count - max(seqs), for_turns):
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
