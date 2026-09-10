"""Saying who is being asked, without saying what they answered.

A turn is thirty to ninety seconds of a single dot. Every other
application on this shelf fixes that by streaming the reply token by
token, and this one cannot — which is worth a test file rather than a
comment, because it will otherwise read as an omission.

Everything a character says is checked *after* it is written and before
it becomes an event: a line naming the secret its speaker is keeping is
refused, a narration that invents a fact or plays the player is dropped,
a line repeating one already said is asked for again. Streaming would put
the words on screen ahead of all of that — and the sharpest of those
guards exists precisely to stop "the music box" reaching a player who has
not earned it. Streaming and retracting tells them anyway.

So the channel carries a key and a job. The safety is structural: there
is nowhere in the frame to put a sentence.
"""
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from fabula.api import Working, create_app
from fabula.llm import FakeLLM, Watched, is_filing
from fabula.session import Session

from tests.conftest import ASHGROVE


def a_story():
    return Session.open(
        ASHGROVE, "the_reckoning", llm=FakeLLM(),
        db_path=str(Path(tempfile.mkdtemp()) / "story.sqlite"),
    )


# --- what the channel can carry -----------------------------------------

def test_a_working_frame_has_nowhere_to_put_a_sentence():
    """The whole safety argument, and it is about the type rather than
    about anybody remembering to be careful."""
    assert set(Working.model_fields) == {"who", "name"}


def test_it_reports_the_key_and_never_the_answer():
    seen = []
    watched = Watched(FakeLLM(), seen.append)

    answer = watched.complete(system="s", prompt="a secret prompt", key="tomas")

    assert seen == ["tomas"]
    assert answer not in seen
    assert "secret" not in "".join(seen)


def test_bookkeeping_is_not_somebody_taking_a_turn():
    """The readings and summaries are two thirds of a turn's calls and
    none of them is anybody speaking in the room."""
    seen = []
    watched = Watched(FakeLLM(), seen.append)
    for key in ("interpret:maria", "summary:tomas", "classify:tomas", "beat"):
        watched.complete(system="s", prompt="p", key=key)
        assert is_filing(key)

    assert seen == []

    watched.complete(system="s", prompt="p", key="__narrator__")
    watched.complete(system="s", prompt="p", key="maria")
    assert seen == ["__narrator__", "maria"]


# --- and how a session wears it -----------------------------------------

def test_a_session_reports_the_cast_taking_its_turn():
    session = a_story()
    seen = []
    session.watch(seen.append)

    session.say("Tomás, what is it?")

    assert seen, "somebody was asked something"
    assert all(isinstance(key, str) for key in seen)
    # Every line the scene actually produced is absent from the channel.
    said = " ".join(e.content for e in session.store.get_events(session.scene.id))
    for key in seen:
        assert key not in said or key in session.characters, key
    session.close()


def test_watching_twice_does_not_report_twice():
    session = a_story()
    once, twice = [], []
    session.watch(once.append)
    session.watch(twice.append)

    session.say("Tomás.")

    assert once == [], "the first watcher was replaced, not stacked under the second"
    assert twice
    session.close()


def test_it_can_be_taken_off_again():
    session = a_story()
    seen = []
    session.watch(seen.append)
    session.watch(None)

    session.say("Tomás.")

    assert seen == []
    assert not isinstance(session.llm, Watched)
    session.close()


def test_nothing_downstream_knows_it_is_being_watched():
    """`Watched` satisfies the one-method protocol, so the director, the
    agents and the narrator go on holding a single client."""
    session = a_story()
    session.watch(lambda key: None)

    assert isinstance(session.llm, Watched)
    assert session.director.llm is session.llm
    assert session.director.narrator.llm is session.llm
    session.close()


# --- over the wire -------------------------------------------------------

def test_the_frames_reach_a_listener_as_their_own_event_type(tmp_path):
    client = TestClient(create_app(
        ASHGROVE.parent, llm=FakeLLM(),
        library_root=tmp_path / "stories", settings_path=tmp_path / "settings.json",
    ))
    opened = client.post("/sessions", json={"world": "ashgrove", "scene": "the_reckoning"})
    session_id = opened.json()["session_id"]

    with client.stream("GET", f"/sessions/{session_id}/stream?follow=false") as caught_up:
        body = "".join(caught_up.iter_text())

    # The catch-up carries only events; `working` is a live signal and
    # there is nothing live about a scene nobody is playing.
    assert "event: working" not in body
    assert "caught up" in body
