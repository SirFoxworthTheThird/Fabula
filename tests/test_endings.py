"""Scene endings and arc mode.

An `end_condition` is written in the same vocabulary as a pressure
trigger, and it is *advisory*: the engine reports that the thing the
author was building toward has happened. It does not lock the scene.
"""
import pytest

from fabula.commands import run_command
from fabula.loader import load_scenario
from fabula.pressures import has_ended, scene_state
from fabula.session import Session

from tests.conftest import ASHGROVE

SECRET = "I broke Grandma's music box, and I let them blame the cat."


@pytest.fixture
def dinner(fake_llm):
    return Session.open(ASHGROVE, "the_dinner", llm=fake_llm)


def speak(session, actor, content, location="kitchen"):
    session.store.append_event(
        session.director.build_event("utterance", actor, location, content)
    )


def test_a_scene_with_no_end_condition_never_ends(dinner):
    """The dinner is a sandbox: it has no shape to reach, so nothing that
    happens in it can end it — the secret included."""
    assert dinner.scene.end_condition == {}

    speak(dinner, "tomas", SECRET)

    assert dinner.ended() is False


def test_the_reckoning_is_an_arc_that_ends_on_the_secret(fake_llm):
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)

    assert session.scene.mode == "arc"
    assert session.scene.end_condition == {"fact_spoken": "music_box"}
    assert session.ended() is False

    speak(session, "tomas", "The soup needs salt.")
    assert session.ended() is False

    speak(session, "tomas", SECRET)
    assert session.ended() is True


def test_the_reckoning_puts_the_whole_cast_in_one_room(fake_llm):
    """Nobody can slip out of earshot, which is what makes it an arc and
    not another evening."""
    world, characters, scene = load_scenario(ASHGROVE, "the_reckoning")

    assert set(scene.starting_positions.values()) == {"kitchen"}
    assert set(scene.cast) == set(scene.starting_positions)


def test_has_ended_is_false_without_a_condition(dinner):
    events = dinner.store.get_events(dinner.scene.id)
    state = scene_state(events, dinner.characters, dinner.world)

    assert has_ended({}, state, dinner.world.facts) is False


def test_the_ending_is_reported_once_on_the_action_that_caused_it(fake_llm):
    """A scene that is already over does not keep announcing itself, or
    every later line would carry the banner."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)

    before = run_command(session, "Pass the bread.")
    assert before.ended is False

    # The player is the one who says it: an accusation is as good a way to
    # put the music box in the room as a confession.
    at_the_end = run_command(session, "You broke Grandma's music box, didn't you.")
    assert at_the_end.ended is True

    after = run_command(session, "Say something.")
    assert after.ended is False
    assert session.ended() is True


def test_play_continues_after_the_end(fake_llm):
    """Advisory, not enforced: the player can keep talking, and the scene
    keeps recording."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    speak(session, "tomas", SECRET)

    outcome = run_command(session, "How long have you known?")

    assert outcome.quit is False
    assert any(
        p.event.content == "How long have you known?"
        for p in outcome.perceived
    )


def test_the_reveal_still_works_after_the_end(fake_llm):
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    speak(session, "tomas", SECRET)

    outcome = run_command(session, "/reveal")

    assert outcome.message
    assert session.ended() is True


def test_the_scene_state_carries_the_ending_over_http(fake_llm):
    """A GUI cannot offer the reveal at the right moment unless the wire
    tells it the scene is over."""
    from fastapi.testclient import TestClient

    from fabula.api import create_app

    app = create_app(worlds_root=ASHGROVE.parent, llm=fake_llm)
    with TestClient(app) as client:
        opened = client.post(
            "/sessions", json={"world": "ashgrove", "scene": "the_reckoning"}
        ).json()
        assert opened["mode"] == "arc"
        assert opened["ended"] is False

        client.post(
            f"/sessions/{opened['session_id']}/say",
            json={"text": "You broke Grandma's music box, didn't you."},
        )
        state = client.get(f"/sessions/{opened['session_id']}").json()

        assert state["ended"] is True


def test_a_fact_nobody_heard_was_not_raised(fake_llm):
    """Found by playing it. Tomás's authored off-screen intention names
    the music box in its own action text, so checking the glue alone in
    an empty kitchen satisfied `fact_spoken` and ended an arc that was
    waiting for somebody to say it out loud.

    A word in the log is not a subject in the room.
    """
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    for other in ("maria", "elena"):
        session.store.append_event(
            session.director.build_event("arrival", other, "study", f"{other} goes through.")
        )

    speak(session, "tomas", "takes the music box down and checks the seam where he glued it")

    assert session.ended() is False


def test_the_same_words_end_it_once_somebody_is_there_to_hear(fake_llm):
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)

    speak(session, "tomas", SECRET)  # Maria and Elena are at the table

    assert session.ended() is True


def test_half_hearing_it_through_a_wall_is_not_hearing_it(fake_llm):
    """The degraded descriptor carries no words — "muffled voices from
    the kitchen" — so somebody through a wall did not catch the subject.
    """
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    for other in ("maria", "elena"):
        session.store.append_event(
            session.director.build_event("arrival", other, "study", f"{other} goes through.")
        )

    session.store.append_event(
        session.director.build_event(
            "utterance", "tomas", "kitchen", SECRET, audibility="adjacent"
        )
    )

    assert session.ended() is False
