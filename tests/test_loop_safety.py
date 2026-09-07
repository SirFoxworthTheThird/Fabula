"""Turn-loop guards (spec §7): agent-to-agent exchange always terminates,
and the user always regains the turn within N exchanges."""
from fabula.db import EventStore
from fabula.director import Director
from fabula.llm import FakeLLM
from fabula.narrator import Narrator


def _director(world, characters, scene):
    return Director(EventStore(), world, characters, scene, Narrator(FakeLLM()), FakeLLM())


def test_agent_loop_terminates_within_the_turn_budget(scenario):
    world, characters, scene = scenario
    director = _director(world, characters, scene)

    # Address everyone so bids are as high as possible — the adversarial
    # case for a loop that might not want to stop.
    user_event = director.build_event(
        "utterance", "elena", "kitchen", "Tomas, Maria, please, talk to me!"
    )
    turn_events = director.run_turn(user_event)

    generated = turn_events[1:]
    assert len(generated) <= scene.turn_budget


def test_user_regains_the_turn_within_n_exchanges(scenario):
    world, characters, scene = scenario
    director = _director(world, characters, scene)

    user_event = director.build_event(
        "utterance", "elena", "kitchen", "Tomas, Maria, please, talk to me!"
    )
    turn_events = director.run_turn(user_event)

    # No run of consecutive non-user turns may exceed the scene's cap —
    # that cap is exactly the guarantee that the user gets the turn back.
    assert len(turn_events) - 1 <= scene.max_consecutive_agent_turns
