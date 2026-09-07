"""Divergence tests: the feature working, not just invariant 1 holding.
Different characters must diverge in what they believe, not converge on
one shared truth."""
from fabula.memory import form_belief, project
from fabula.models import Event


def test_different_salience_bias_yields_different_belief_salience(scenario):
    world, characters, scene = scenario
    event = Event(
        scene_id=scene.id,
        seq=1,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="tomas",
        location_id="kitchen",
        content="An ambiguous glance is exchanged.",
        audibility="room",
        salience_base=0.5,
    )
    tomas = characters["tomas"]  # salience_bias.utterance = 1.0
    # Co-locate Maria so this isolates salience_bias, not perception level.
    maria_in_kitchen = characters["maria"].model_copy(update={"location_id": "kitchen"})

    tomas_projected = project(tomas, [event], world)[0]
    maria_projected = project(maria_in_kitchen, [event], world)[0]

    tomas_belief = form_belief(tomas, tomas_projected)
    maria_belief = form_belief(maria_in_kitchen, maria_projected)

    assert tomas_belief.content == maria_belief.content  # same words heard
    assert tomas_belief.salience != maria_belief.salience  # different weight given to them
    assert maria_belief.salience > tomas_belief.salience  # Maria's bias (1.3) > Tomás's (1.0)


def test_degraded_perception_forms_a_belief_from_the_descriptor_not_the_truth(scenario):
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

    projected = project(maria, [event], world)[0]
    belief = form_belief(maria, projected)

    assert projected.perception == "degraded"
    assert "music box" not in belief.content.lower()
    assert belief.confidence < 1.0
