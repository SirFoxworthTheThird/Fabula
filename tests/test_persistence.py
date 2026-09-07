"""M5: cross-scene persistence (spec §10)."""
from fabula.director import Director
from fabula.llm import FakeLLM
from fabula.loader import Scene
from fabula.memory import form_belief, project
from fabula.models import Event, Goal, Relationship
from fabula.narrator import Narrator
from fabula.persistence import (
    BELIEF_SALIENCE_FLOOR,
    age_beliefs,
    begin_scene,
    encode_belief,
    unresolved_goals,
)


def second_scene(scene: Scene) -> Scene:
    """A later scene in the same world, sharing the same store."""
    return scene.model_copy(update={"id": "the_morning_after"})


def a_director(scenario, store, scene=None):
    world, characters, loaded = scenario
    return Director(
        store, world, characters, scene or loaded, Narrator(FakeLLM()), FakeLLM()
    )


def test_beliefs_are_formed_during_play_and_persist_by_character(scenario, store):
    world, characters, scene = scenario
    director = a_director(scenario, store)

    director.run_turn(
        director.build_event(
            "utterance", "elena", "kitchen", "Tomas, you have been strange all evening."
        )
    )

    assert store.get_beliefs("tomas")
    # Maria is in the study and perceived none of it, so she remembers none of it.
    assert store.get_beliefs("maria") == []


def test_only_salient_moments_are_encoded(scenario, store):
    _world, characters, scene = scenario
    tomas = characters["tomas"]
    dull = Event(
        id=1,
        scene_id=scene.id,
        seq=1,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="elena",
        location_id="kitchen",
        content="The soup needs salt.",
        audibility="room",
        salience_base=0.1,
    )
    sharp = dull.model_copy(update={"id": 2, "seq": 2, "salience_base": 0.9})
    projected = project(tomas, [dull, sharp], world=_world)

    assert not encode_belief(store, tomas, form_belief(tomas, projected[0]))
    assert encode_belief(store, tomas, form_belief(tomas, projected[1]))
    assert [b.source_event_id for b in store.get_beliefs("tomas")] == [2]


def test_the_same_moment_is_never_encoded_twice(scenario, store):
    world, characters, scene = scenario
    tomas = characters["tomas"]
    event = Event(
        id=1,
        scene_id=scene.id,
        seq=1,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="elena",
        location_id="kitchen",
        content="You have been strange all evening.",
        audibility="room",
        salience_base=0.9,
    )
    projected = project(tomas, [event], world)[0]

    encode_belief(store, tomas, form_belief(tomas, projected))
    encode_belief(store, tomas, form_belief(tomas, projected))

    assert len(store.get_beliefs("tomas")) == 1


def test_beliefs_age_between_scenes_and_salient_ones_resist(scenario, store):
    world, characters, scene = scenario
    tomas = characters["tomas"]
    betrayal = Event(
        id=1,
        scene_id=scene.id,
        seq=1,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="maria",
        location_id="kitchen",
        content="I told mother it was you.",
        audibility="room",
        salience_base=1.0,
    )
    passing = betrayal.model_copy(
        update={"id": 2, "seq": 2, "content": "It might rain later.", "salience_base": 0.55}
    )
    for projected in project(tomas, [betrayal, passing], world):
        encode_belief(store, tomas, form_belief(tomas, projected))

    before = {b.source_event_id: b.salience for b in store.get_beliefs("tomas")}
    age_beliefs(store, tomas.id)
    after = {b.source_event_id: b.salience for b in store.get_beliefs("tomas")}

    assert after[2] < before[2]                       # the unremarkable fades
    assert after[1] == before[1]                      # the betrayal stays sharp
    assert (before[2] - after[2]) > (before[1] - after[1])


def test_aging_never_corrects_a_belief_against_the_truth(scenario, store):
    """Invariant 2: a belief store may hold things that are false, and no
    component reconciles it against world state."""
    world, characters, scene = scenario
    tomas = characters["tomas"]
    false_event = Event(
        id=1,
        scene_id=scene.id,
        seq=1,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="maria",
        location_id="kitchen",
        content="The cat knocked the music box off the mantel.",
        audibility="room",
        salience_base=0.9,
    )
    encode_belief(store, tomas, form_belief(tomas, project(tomas, [false_event], world)[0]))

    age_beliefs(store, tomas.id)

    kept = store.get_beliefs("tomas")[0]
    assert kept.content == false_event.content  # untouched, still false


def test_relationships_are_seeded_authored_and_carry_forward(scenario, store):
    _world, characters, _scene = scenario

    begin_scene(store, characters)
    seeded = store.get_relationships("tomas")

    assert seeded["elena"].trust == 0.8            # authored
    assert seeded["elena"].note                    # in his own words
    assert "maria" in seeded

    # Everyone holds a stance toward everyone, the user's character included.
    assert set(store.get_relationships("maria")) == {"tomas", "elena"}
    assert set(store.get_relationships("elena")) == {"tomas", "maria"}


def test_re_entering_a_scene_does_not_reset_a_relationship(scenario, store):
    _world, characters, _scene = scenario
    begin_scene(store, characters)

    store.bump_interaction("tomas", "elena")
    store.bump_interaction("tomas", "elena")
    begin_scene(store, characters)  # a later scene starts

    assert store.get_relationships("tomas")["elena"].interactions == 2


def test_interactions_accumulate_through_play(scenario, store):
    _world, characters, scene = scenario
    begin_scene(store, characters)
    director = a_director(scenario, store)

    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Tomas, talk to me."))

    assert store.get_relationships("tomas")["elena"].interactions > 0
    # Maria, a room away, has not interacted with anyone.
    assert store.get_relationships("maria")["elena"].interactions == 0


def test_unresolved_goals_carry_and_resolved_ones_drop(scenario, store):
    _world, characters, _scene = scenario
    tomas = characters["tomas"].model_copy(
        update={
            "goals": [
                Goal(id="protect_secret", description="Keep the music box quiet"),
                Goal(id="fix_the_fence", description="Mend the back fence"),
            ]
        }
    )

    assert {g.id for g in unresolved_goals(store, tomas)} == {"protect_secret", "fix_the_fence"}

    store.resolve_goal(tomas.id, "fix_the_fence")

    assert {g.id for g in unresolved_goals(store, tomas)} == {"protect_secret"}


def test_a_character_carries_beliefs_into_the_next_scene(scenario, store):
    """A character who remembers last week's betrayal is the point."""
    world, characters, scene = scenario
    director = a_director(scenario, store)
    begin_scene(store, characters)
    director.run_turn(
        director.build_event(
            "utterance", "elena", "kitchen", "Tomas, you have been strange all evening."
        )
    )
    remembered = {b.content for b in store.get_beliefs("tomas")}
    assert remembered

    # A new scene, same store: the events are gone, the character is not.
    later = second_scene(scene)
    begin_scene(store, characters)

    assert store.get_events(later.id) == []
    assert {b.content for b in store.get_beliefs("tomas")} == remembered


def test_relationship_state_includes_the_users_character(scenario, store):
    _world, characters, _scene = scenario
    begin_scene(store, characters)

    toward_user = store.get_relationships("tomas")["elena"]

    assert isinstance(toward_user, Relationship)
    assert toward_user.affinity == 0.7


def test_belief_floor_is_what_gates_encoding(scenario, store):
    _world, characters, _scene = scenario
    assert 0.0 < BELIEF_SALIENCE_FLOOR < 1.0
