"""M4: time skips and off-screen resolution (spec §8)."""
from datetime import timedelta

from fabula.chronology import (
    LARGE_SKIP_MINUTES,
    derive_skip_minutes,
    describe_duration,
    pending_intentions,
    render_time_skip,
    was_asleep,
)
from fabula.director import Director
from fabula.llm import FakeLLM
from fabula.memory import project
from fabula.models import Event, Intention
from fabula.narrator import Narrator


def a_director(scenario, store, characters=None):
    world, loaded, scene = scenario
    return Director(
        store, world, characters or loaded, scene, Narrator(FakeLLM()), FakeLLM()
    )


def with_intentions(characters):
    """Tomás has something to do in 20 minutes, Maria in 30."""
    updated = dict(characters)
    updated["tomas"] = characters["tomas"].model_copy(
        update={
            "intentions": [
                Intention(
                    id="check_the_glue",
                    description="checks the seam where he glued it",
                    location_id="kitchen",
                    ready_after_minutes=20,
                )
            ]
        }
    )
    updated["maria"] = characters["maria"].model_copy(
        update={
            "intentions": [
                Intention(
                    id="finish_the_letters",
                    description="stacks the letters in date order",
                    location_id="study",
                    ready_after_minutes=30,
                )
            ]
        }
    )
    return updated


def test_skip_is_derived_from_the_next_ready_intention(scenario):
    _world, characters, _scene = scenario
    cast = with_intentions(characters)

    assert derive_skip_minutes(cast, []) == 20  # not an arbitrary span


def test_no_pending_intentions_means_no_skip(scenario):
    """Nothing pending means nothing worth discovering, so there is no
    jump to derive — that is what stops an arbitrary span landing on an
    empty room."""
    _world, characters, _scene = scenario
    idle = {
        cid: character.model_copy(update={"intentions": []})
        for cid, character in characters.items()
    }

    assert derive_skip_minutes(idle, []) is None


def test_skip_is_committed_to_the_log_with_an_explicit_duration(scenario, store):
    _world, characters, scene = scenario
    director = a_director(scenario, store, with_intentions(characters))
    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Back in a bit."))
    before = store.get_events(scene.id)[-1].story_time

    appended = director.advance_time()

    skip = appended[0]
    assert skip.kind == "time_skip"
    assert skip.metadata["minutes"] == 20
    assert skip.story_time == before + timedelta(minutes=20)


def move(director, character, destination):
    """Put a character in a room by logging their arrival, the way the
    engine itself moves anyone."""
    director.store.append_event(
        director.build_event(
            "arrival", character.id, destination, f"{character.name} comes in."
        )
    )


def test_off_screen_time_produces_coarse_summary_events(scenario, store):
    _world, characters, scene = scenario
    director = a_director(scenario, store, with_intentions(characters))
    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Back in a bit."))

    appended = director.advance_time(minutes=30)

    resolved = [e for e in appended if e.detail_level == "summary"]
    # Elena is in the kitchen, so only Maria's study work happened off screen.
    assert {e.metadata["intention_id"] for e in resolved} == {"finish_the_letters"}
    assert all(e.kind == "action" for e in resolved)


def test_someone_in_the_room_with_the_user_is_not_resolved_off_screen(scenario, store):
    """Off-screen resolution is for what the user is not there to see.
    Tomás is standing in the kitchen with Elena, so his intention waits
    rather than being flattened into a coarse stub she'd 'watch'."""
    _world, characters, scene = scenario
    cast = with_intentions(characters)
    director = a_director(scenario, store, cast)
    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Back in a bit."))

    director.advance_time(minutes=60)

    events = store.get_events(scene.id)
    assert not any(e.metadata.get("intention_id") == "check_the_glue" for e in events)
    assert pending_intentions(cast["tomas"], events)

    # Once she leaves the room, the same intention resolves off screen.
    move(director, cast["elena"], "study")
    director.advance_time(minutes=60)

    assert any(
        e.metadata.get("intention_id") == "check_the_glue"
        for e in store.get_events(scene.id)
    )


def test_an_intention_resolves_only_once(scenario, store):
    _world, characters, scene = scenario
    director = a_director(scenario, store, with_intentions(characters))
    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Back in a bit."))

    cast = with_intentions(characters)
    director.advance_time(minutes=30)
    move(director, cast["elena"], "study")  # so the kitchen resolves too
    director.advance_time(minutes=30)
    director.advance_time(minutes=30)

    events = store.get_events(scene.id)
    resolved = [e.metadata["intention_id"] for e in events if e.metadata.get("intention_id")]
    assert sorted(resolved) == ["check_the_glue", "finish_the_letters"]  # each exactly once

    assert pending_intentions(cast["tomas"], events) == []
    # And with nothing left pending, a derived skip declines to move at all.
    assert director.advance_time() == []


def test_large_skips_require_consent(scenario, store):
    _world, characters, scene = scenario
    director = a_director(scenario, store, with_intentions(characters))
    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Back in a bit."))
    big = LARGE_SKIP_MINUTES + 60

    declined = director.advance_time(minutes=big, consent=lambda _minutes: False)
    assert declined == []
    assert not any(e.kind == "time_skip" for e in store.get_events(scene.id))

    accepted = director.advance_time(minutes=big, consent=lambda _minutes: True)
    assert any(e.kind == "time_skip" for e in accepted)


def test_small_skips_do_not_need_consent(scenario, store):
    _world, characters, _scene = scenario
    director = a_director(scenario, store, with_intentions(characters))
    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Back in a bit."))

    appended = director.advance_time(minutes=LARGE_SKIP_MINUTES)

    assert any(e.kind == "time_skip" for e in appended)


def test_elapsed_time_is_perceived_non_uniformly(scenario):
    """Someone asleep experiences the skip as a discontinuity; someone
    awake and waiting experienced every hour."""
    world, characters, scene = scenario
    sleeps = Event(
        id=1,
        scene_id=scene.id,
        seq=1,
        story_time=scene.start_time,
        kind="state_change",
        actor_id="tomas",
        location_id="kitchen",
        content="Tomás dozes off in the chair.",
        audibility="room",
        metadata={"character_id": "tomas", "state": "asleep"},
    )
    skip = Event(
        id=2,
        scene_id=scene.id,
        seq=2,
        story_time=scene.start_time,
        kind="time_skip",
        actor_id=None,
        location_id="kitchen",
        content="2 hours pass.",
        audibility="building",
        metadata={"minutes": 120},
    )
    events = [sleeps, skip]

    assert was_asleep("tomas", events, before_seq=2)
    assert not was_asleep("elena", events, before_seq=2)

    asleep_view = project(characters["tomas"], events, world)[-1].perceived_content
    awake_view = project(characters["elena"], events, world)[-1].perceived_content

    assert asleep_view != awake_view
    assert "unfelt" in asleep_view
    assert "feel every one" in awake_view


def test_a_skip_reaches_everyone_in_the_building(scenario):
    world, characters, scene = scenario
    skip = Event(
        id=1,
        scene_id=scene.id,
        seq=1,
        story_time=scene.start_time,
        kind="time_skip",
        actor_id=None,
        location_id="kitchen",
        content="An hour passes.",
        audibility="building",
        metadata={"minutes": 60},
    )

    # Maria is in the study and still lives through the same hour.
    assert len(project(characters["maria"], [skip], world)) == 1


def test_off_screen_detail_materializes_only_for_someone_in_the_room(scenario, store):
    """Spec §8: when the user finds the kitchen ransacked, that is when
    'Tomás searched the house' expands into specifics."""
    _world, characters, scene = scenario
    cast = with_intentions(characters)
    director = a_director(scenario, store, cast)
    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Back in a bit."))
    director.advance_time(minutes=30)  # Maria's study work happens unwatched

    elena, tomas = cast["elena"], cast["tomas"]
    study_summary = next(
        e
        for e in store.get_events(scene.id)
        if e.metadata.get("intention_id") == "finish_the_letters"
    )

    # Standing in the kitchen, there is nothing of the study to see.
    assert director.materialize(study_summary, elena) is None
    assert director.materialize(study_summary, tomas) is None

    move(director, elena, "study")
    revealed = director.materialize(study_summary, elena)
    assert revealed is not None
    assert revealed.metadata["materializes"] == study_summary.id
    assert director.materialize(study_summary, elena) is None  # never twice


def test_unmaterialized_here_only_lists_the_observers_own_room(scenario, store):
    _world, characters, _scene = scenario
    cast = with_intentions(characters)
    director = a_director(scenario, store, cast)
    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Back in a bit."))
    director.advance_time(minutes=30)  # Maria's study work, unwatched

    assert director.unmaterialized_here(cast["elena"]) == []  # she is in the kitchen

    move(director, cast["elena"], "study")
    waiting = director.unmaterialized_here(cast["elena"])

    assert [e.metadata["intention_id"] for e in waiting] == ["finish_the_letters"]


def test_off_screen_action_in_another_room_is_not_perceived(scenario, store):
    """The world moved on without the user; they only find out by looking."""
    world, characters, scene = scenario
    director = a_director(scenario, store, with_intentions(characters))
    director.run_turn(director.build_event("utterance", "elena", "kitchen", "Back in a bit."))
    director.advance_time(minutes=30)

    study_work = [
        e
        for e in store.get_events(scene.id)
        if e.metadata.get("intention_id") == "finish_the_letters"
    ]
    assert study_work

    assert project(characters["elena"], study_work, world) == []


def test_duration_phrasing():
    assert describe_duration(1) == "a minute"
    assert describe_duration(45) == "45 minutes"
    assert describe_duration(60) == "an hour"
    assert describe_duration(180) == "3 hours"


def test_render_time_skip_is_deterministic(scenario):
    _world, _characters, scene = scenario
    skip = Event(
        id=1,
        scene_id=scene.id,
        seq=1,
        story_time=scene.start_time,
        kind="time_skip",
        actor_id=None,
        location_id="kitchen",
        content="An hour passes.",
        audibility="building",
        metadata={"minutes": 60},
    )

    assert render_time_skip(skip, "elena", [skip]) == render_time_skip(skip, "elena", [skip])
