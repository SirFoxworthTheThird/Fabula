"""Invariant 1: a character can never be given information their
character could not perceive. These are the highest-priority tests in
the suite (spec §13) — everything else is negotiable, this is not.
"""
from fabula.agents import generate_utterance
from fabula.llm import FakeLLM
from fabula.memory import assemble_context, project
from fabula.models import Event

SECRET = "music box"


def secret_event(scene, seq: int = 1) -> Event:
    return Event(
        scene_id=scene.id,
        seq=seq,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="tomas",
        location_id="kitchen",
        content="I broke Grandma's music box, Elena. It wasn't the cat.",
        audibility="room",
        addressed_to=["elena"],
    )


def test_maria_projection_contains_zero_tokens_of_the_secret(scenario):
    world, characters, scene = scenario
    event = secret_event(scene)
    maria = characters["maria"]

    projected = project(maria, [event], world)

    assert all(SECRET not in p.perceived_content.lower() for p in projected)
    assert all("cat" not in p.perceived_content.lower() for p in projected)
    context = assemble_context(maria, projected)
    assert SECRET not in context.lower()


def test_maria_reply_does_not_reference_the_secret(scenario):
    world, characters, scene = scenario
    event = secret_event(scene)
    maria = characters["maria"]
    llm = FakeLLM()

    reply = generate_utterance(maria, [event], world, llm)

    assert SECRET not in reply.lower()
    # FakeLLM's output is a pure function of its prompt and invents nothing,
    # so this also proves the prompt itself never carried the secret.
    system, prompt, _ = llm.calls[-1]
    assert SECRET not in system.lower()
    assert SECRET not in prompt.lower()


def test_tomas_in_the_same_room_does_perceive_it(scenario):
    world, characters, scene = scenario
    event = secret_event(scene)
    tomas = characters["tomas"]

    projected = project(tomas, [event], world)

    assert len(projected) == 1
    assert projected[0].perception == "full"
    assert SECRET in projected[0].perceived_content.lower()


def test_degraded_perception_never_carries_true_content(scenario):
    """A louder event (audibility 'adjacent') reaches the next room, but
    only as a degraded descriptor — never the real content."""
    world, characters, scene = scenario
    event = Event(
        scene_id=scene.id,
        seq=1,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="tomas",
        location_id="kitchen",
        content="I broke Grandma's music box!",
        audibility="adjacent",
    )
    maria = characters["maria"]

    projected = project(maria, [event], world)

    assert len(projected) == 1
    assert projected[0].perception == "degraded"
    assert SECRET not in projected[0].perceived_content.lower()


def test_narration_never_describes_what_an_offscreen_character_cannot_perceive(scenario):
    world, characters, scene = scenario
    trigger = secret_event(scene, seq=1)
    narration = Event(
        scene_id=scene.id,
        seq=2,
        story_time=scene.start_time,
        kind="narration",
        actor_id=None,
        location_id="kitchen",
        content="Tomás glances guiltily at the music box on the shelf.",
        audibility="room",
    )
    maria = characters["maria"]

    projected = project(maria, [trigger, narration], world)

    assert all(SECRET not in p.perceived_content.lower() for p in projected)


def test_bid_rationale_never_leaks_into_another_characters_context(scenario, monkeypatch):
    """The director sees every bid, but a rationale must never travel
    further than the director: it is never written into an Event, so it
    can never reach another character's projection."""
    from fabula.db import EventStore
    from fabula.director import Director
    import fabula.director as director_module
    from fabula.narrator import Narrator

    world, characters, scene = scenario
    store = EventStore()
    director = Director(store, world, characters, scene, Narrator(FakeLLM()), FakeLLM())

    marker = "RATIONALE_MARKER_SHOULD_NEVER_LEAK"
    original_get_bid = director_module.get_bid

    def spy_get_bid(character, event, level, events, world_, llm):
        bid = original_get_bid(character, event, level, events, world_, llm)
        return bid.model_copy(update={"one_line_reason": marker})

    monkeypatch.setattr(director_module, "get_bid", spy_get_bid)

    user_event = director.build_event(
        "utterance", "elena", "kitchen", "Tomas, tell me something, anything!"
    )
    turn_events = director.run_turn(user_event)

    for event in turn_events:
        assert marker not in event.content

    for character in characters.values():
        projected = project(character, store.get_events(scene.id), world)
        context = assemble_context(character, projected)
        assert marker not in context


def test_leak_survives_50_turns_and_summarization():
    import pytest

    pytest.skip("memory tiers/summarization land in M2; revisit once implemented")


def test_leak_survives_a_scene_boundary_with_persistence():
    import pytest

    pytest.skip("cross-scene persistence lands in M5; revisit once implemented")
