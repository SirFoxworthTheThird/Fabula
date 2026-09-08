"""M3: pressures and the arc/sandbox director objectives (spec §5.5, §9)."""
import pytest

from fabula.director import Director
from fabula.llm import FakeLLM
from fabula.loader import load_pressures
from fabula.memory import project
from fabula.models import Event, Pressure
from fabula.narrator import Narrator
from fabula.pressures import (
    evaluate_trigger,
    is_eligible,
    pressure_desire,
    scene_state,
    select_pressure,
)

from tests.conftest import ASHGROVE


def kitchen_event(scene, seq, content="They talk about the soup.", metadata=None):
    return Event(
        id=seq,
        scene_id=scene.id,
        seq=seq,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="tomas",
        location_id="kitchen",
        content=content,
        audibility="room",
        metadata=metadata or {},
    )


def a_pressure(**overrides) -> Pressure:
    base = dict(
        id="test_pressure",
        intent="Something authored happens",
        weight=0.5,
        trigger={"turns_elapsed": "> 3"},
        effect={"kind": "narration", "location": "kitchen"},
        cooldown_turns=5,
        max_fires=2,
    )
    base.update(overrides)
    return Pressure(**base)


def test_the_shipped_world_loads_its_pressures():
    pressures = load_pressures(ASHGROVE)

    assert [p.id for p in pressures] == [
        "maria_comes_through",
        "the_mantel_draws_the_eye",
        "mother_calls_from_upstairs",
    ]


def test_trigger_is_not_satisfied_before_its_condition(scenario):
    _world, characters, scene = scenario
    pressure = a_pressure(trigger={"turns_elapsed": "> 3"})

    early = scene_state([kitchen_event(scene, 1)], characters)
    late = scene_state([kitchen_event(scene, seq) for seq in range(1, 6)], characters)

    assert not evaluate_trigger(pressure.trigger, early, {})
    assert evaluate_trigger(pressure.trigger, late, {})


def test_fact_unspoken_trigger_closes_once_the_fact_is_said(scenario):
    world, characters, scene = scenario
    pressure = a_pressure(trigger={"fact_unspoken": "music_box", "for_turns": "> 2"})

    quiet = scene_state([kitchen_event(scene, seq) for seq in range(1, 5)], characters)
    spoken = scene_state(
        [kitchen_event(scene, seq) for seq in range(1, 4)]
        + [kitchen_event(scene, 4, "You broke the music box, didn't you?")],
        characters,
    )

    assert evaluate_trigger(pressure.trigger, quiet, world.facts)
    assert not evaluate_trigger(pressure.trigger, spoken, world.facts)


def test_character_at_trigger_reads_location_from_the_log(scenario):
    world, characters, scene = scenario
    pressure = a_pressure(trigger={"character_at": {"maria": "study"}})

    before = scene_state([kitchen_event(scene, 1)], characters)
    arrival = Event(
        id=2,
        scene_id=scene.id,
        seq=2,
        story_time=scene.start_time,
        kind="arrival",
        actor_id="maria",
        location_id="kitchen",
        content="Maria comes through.",
        audibility="room",
    )
    after = scene_state([kitchen_event(scene, 1), arrival], characters)

    assert evaluate_trigger(pressure.trigger, before, world.facts)
    assert not evaluate_trigger(pressure.trigger, after, world.facts)


def test_max_fires_and_cooldown_are_derived_from_the_log(scenario):
    _world, characters, scene = scenario
    pressure = a_pressure(cooldown_turns=5, max_fires=2)

    fired_once = scene_state(
        [kitchen_event(scene, seq) for seq in range(1, 5)]
        + [kitchen_event(scene, 5, metadata={"pressure_id": pressure.id})],
        characters,
    )
    assert not is_eligible(pressure, fired_once, {})  # still cooling down

    cooled = scene_state(
        fired_once.events + [kitchen_event(scene, seq) for seq in range(6, 12)], characters
    )
    assert is_eligible(pressure, cooled, {})

    fired_twice = scene_state(
        cooled.events + [kitchen_event(scene, 12, metadata={"pressure_id": pressure.id})],
        characters,
    )
    exhausted = scene_state(
        fired_twice.events + [kitchen_event(scene, seq) for seq in range(13, 25)], characters
    )
    assert not is_eligible(pressure, exhausted, {})  # max_fires reached


def test_unknown_trigger_key_is_rejected_rather_than_always_eligible(scenario):
    _world, characters, scene = scenario
    state = scene_state([kitchen_event(scene, 1)], characters)

    with pytest.raises(ValueError, match="unknown pressure trigger key"):
        evaluate_trigger({"fact_whispered": "music_box"}, state, {})


def test_arc_escalates_while_sandbox_waits_for_a_lull(scenario):
    """Spec §9: the mode changes how pressures are selected, not what
    they are."""
    _world, characters, scene = scenario
    pressure = a_pressure(weight=0.5)
    late = scene_state([kitchen_event(scene, seq) for seq in range(1, 25)], characters)
    busy_scene, quiet_scene = 0.8, 0.1

    arc_busy = pressure_desire(pressure, late, "arc", busy_scene)
    sandbox_busy = pressure_desire(pressure, late, "sandbox", busy_scene)
    sandbox_quiet = pressure_desire(pressure, late, "sandbox", quiet_scene)

    assert arc_busy > pressure.weight       # arc escalates as the scene runs on
    assert sandbox_busy == 0.0              # sandbox leaves a busy scene alone
    assert sandbox_quiet == pressure.weight  # and reseeds tension once it goes quiet


def test_arc_pressure_grows_with_elapsed_turns(scenario):
    _world, characters, scene = scenario
    pressure = a_pressure(weight=0.4)

    early = scene_state([kitchen_event(scene, seq) for seq in range(1, 6)], characters)
    later = scene_state([kitchen_event(scene, seq) for seq in range(1, 26)], characters)

    assert pressure_desire(pressure, later, "arc", 0.0) > pressure_desire(
        pressure, early, "arc", 0.0
    )


def test_selection_is_deterministic_and_prefers_authored_order_on_ties(scenario):
    _world, characters, scene = scenario
    first = a_pressure(id="first", weight=0.5)
    second = a_pressure(id="second", weight=0.5)
    state = scene_state([kitchen_event(scene, seq) for seq in range(1, 6)], characters)

    chosen = select_pressure([first, second], state, {}, "sandbox", 0.0)

    assert chosen is not None
    assert chosen[0].id == "first"
    assert select_pressure([first, second], state, {}, "sandbox", 0.0)[0].id == "first"


def test_director_never_fires_a_pressure_that_was_not_authored(scenario, store):
    world, characters, scene = scenario
    director = Director(
        store, world, characters, scene, Narrator(FakeLLM()), FakeLLM(), pressures=[]
    )

    user_event = director.build_event("utterance", "elena", "kitchen", "Tomas, say something.")
    turn_events = director.run_turn(user_event)

    assert all(not event.metadata.get("pressure_id") for event in turn_events)


def test_a_fired_pressure_becomes_an_ordinary_perception_filtered_event(scenario, store):
    """A pressure's effect is an event like any other: Maria in the study
    does not perceive a room-scoped beat fired in the kitchen."""
    world, characters, scene = scenario
    pressure = a_pressure(
        id="kitchen_beat",
        trigger={"turns_elapsed": "> 0"},
        weight=0.9,
        effect={"kind": "narration", "location": "kitchen"},
    )
    director = Director(
        store, world, characters, scene, Narrator(FakeLLM()), FakeLLM(), pressures=[pressure]
    )

    user_event = director.build_event("utterance", "elena", "kitchen", "Well?")
    director.run_turn(user_event)

    all_events = store.get_events(scene.id)
    fired = [e for e in all_events if e.metadata.get("pressure_id") == "kitchen_beat"]
    assert fired, "the pressure should have fired"
    assert project(characters["maria"], fired, world) == []
    assert project(characters["tomas"], fired, world) != []


def test_pressure_intent_never_reaches_a_characters_context(scenario, store):
    """The authored intent is director-only, like a bid rationale. What
    characters get is the narrator's rendering of it."""
    world, characters, scene = scenario
    intent = "AUTHORED_INTENT_MARKER that must never be quoted verbatim"
    pressure = a_pressure(
        id="marked", intent=intent, trigger={"turns_elapsed": "> 0"}, weight=0.9
    )
    director = Director(
        store, world, characters, scene, Narrator(FakeLLM()), FakeLLM(), pressures=[pressure]
    )

    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Well?"))

    all_events = store.get_events(scene.id)
    assert all(intent not in event.content for event in all_events)
    for character in characters.values():
        assert intent not in director.contexts.for_character(character, all_events)


def test_a_trigger_can_name_several_facts_at_once(scenario):
    """A story with two people each holding something cannot say what it
    needs to with a single id — "over once both are in the room" is the
    whole shape of it. A bare string still means one fact, so nothing
    authored against the old vocabulary changes meaning."""
    from fabula.world import Fact

    world, characters, scene = scenario
    facts = {"a": Fact(id="a", keywords=["alpha"]), "b": Fact(id="b", keywords=["bravo"])}

    def state_after(*contents):
        events = [kitchen_event(scene, i + 1, c) for i, c in enumerate(contents)]
        return scene_state(events, characters)

    both = {"fact_spoken": ["a", "b"]}
    assert evaluate_trigger(both, state_after("alpha"), facts) is False
    assert evaluate_trigger(both, state_after("alpha", "bravo"), facts) is True

    # A bare string is still one fact.
    assert evaluate_trigger({"fact_spoken": "a"}, state_after("alpha"), facts) is True

    # `fact_unspoken` with a list is "all still unheld": one surfacing
    # is enough to stop holding.
    neither = {"fact_unspoken": ["a", "b"]}
    assert evaluate_trigger(neither, state_after("nothing"), facts) is True
    assert evaluate_trigger(neither, state_after("alpha"), facts) is False


def test_for_turns_counts_from_the_last_fact_to_land(scenario):
    """The condition became true when the second one was said, not the
    first."""
    from fabula.world import Fact

    world, characters, scene = scenario
    facts = {"a": Fact(id="a", keywords=["alpha"]), "b": Fact(id="b", keywords=["bravo"])}
    events = [
        kitchen_event(scene, 1, "alpha"),
        kitchen_event(scene, 2, "nothing"),
        kitchen_event(scene, 3, "bravo"),
        kitchen_event(scene, 4, "nothing"),
    ]
    state = scene_state(events, characters)

    assert evaluate_trigger({"fact_spoken": ["a", "b"], "for_turns": 1}, state, facts) is True
    assert evaluate_trigger({"fact_spoken": ["a", "b"], "for_turns": 3}, state, facts) is False


def test_an_unknown_fact_id_in_a_list_satisfies_nothing(scenario):
    """An authoring typo must not quietly make a pressure eligible."""
    from fabula.world import Fact

    world, characters, scene = scenario
    facts = {"a": Fact(id="a", keywords=["alpha"])}
    state = scene_state([kitchen_event(scene, 1, "alpha")], characters)

    assert evaluate_trigger({"fact_spoken": ["a", "typo"]}, state, facts) is False
    assert evaluate_trigger({"fact_unspoken": ["typo"]}, state, facts) is False
