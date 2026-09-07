"""Invariant 5: projections are reproducible. Same log + same character
→ byte-identical projection."""
from fabula.memory import assemble_context, project
from fabula.models import Event


def _event(scene, seq=1, actor="tomas", content="Something happens.", audibility="room"):
    return Event(
        scene_id=scene.id,
        seq=seq,
        story_time=scene.start_time,
        kind="utterance",
        actor_id=actor,
        location_id="kitchen",
        content=content,
        audibility=audibility,
    )


def test_projection_is_deterministic_for_the_same_log(scenario):
    world, characters, scene = scenario
    events = [_event(scene, seq=1), _event(scene, seq=2, content="Something else happens.")]
    maria = characters["maria"]

    first = project(maria, events, world)
    second = project(maria, events, world)

    assert [p.perceived_content for p in first] == [p.perceived_content for p in second]
    assert [p.perception for p in first] == [p.perception for p in second]


def test_context_assembly_is_deterministic(scenario, store, fake_llm):
    world, characters, scene = scenario
    events = [_event(scene, seq=1, audibility="adjacent")]
    maria = characters["maria"]
    projected = project(maria, events, world)

    first = assemble_context(maria, projected, scene.id, store, fake_llm)
    second = assemble_context(maria, projected, scene.id, store, fake_llm)
    assert first == second
