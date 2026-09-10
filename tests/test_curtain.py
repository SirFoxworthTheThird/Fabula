"""Getting into a story should not cost what opening it costs.

A scene's opening is a room to describe, a bid from everybody standing in
it, and a line from whoever wanted to speak — eight model calls in a full
room, and every one of them was between clicking Begin and seeing
anything at all. Measured against the rest of the shelf that is the
worst moment in the app: the click that decides whether somebody plays.

So the two halves are separated. Opening a story is free and instant and
lands the player on the scene's own authored first words; the curtain
goes up afterwards, on the stream the client opens anyway, where the wait
is spent watching the room take its turn instead of watching a button.
"""
import tempfile
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fabula.api import Trouble, create_app
from fabula.llm import FakeLLM, ModelUnavailable
from fabula.player import Player
from fabula.session import Session

from tests.conftest import ASHGROVE


class Counted(FakeLLM):
    """Answers as usual, and says how often it was asked."""

    def __init__(self):
        super().__init__()
        self.asked = 0
        self._lock = threading.Lock()

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        with self._lock:
            self.asked += 1
        return super().complete(system, prompt, key)


class Refuses(FakeLLM):
    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        raise ModelUnavailable("Missing credentials")


def a_story(llm, curtain=True, player=None):
    return Session.open(
        ASHGROVE, "the_dinner", llm=llm, curtain=curtain, player=player,
        db_path=str(Path(tempfile.mkdtemp()) / "story.sqlite"),
    )


def lines(session):
    return [e.content for e in session.store.get_events(session.scene.id)]


# --- the click ----------------------------------------------------------

def test_opening_a_story_asks_the_model_nothing():
    """The whole point of the split, stated as the only number that
    matters: how many times somebody's provider is called before they
    are in the story."""
    llm = Counted()

    a_story(llm, curtain=False)

    assert llm.asked == 0


def test_and_still_lands_on_something_to_read():
    """Free does not mean empty. The scene's own first words are
    authored, so they cost nothing and can be handed over immediately —
    which is what makes the deferred half bearable."""
    session = a_story(Counted(), curtain=False)

    assert lines(session) == [session.scene.opening.strip()]


def test_the_curtain_does_the_same_thing_late_as_it_did_inline():
    """Deferring it is a change of *when*, and must not be a change of
    what: the same narration, the same greeting, in the same order."""
    inline = a_story(FakeLLM())
    deferred = a_story(FakeLLM(), curtain=False)

    deferred.raise_curtain()

    assert lines(deferred) == lines(inline)


def test_raising_it_twice_opens_nothing_twice():
    """It runs from a background task, and a client that reconnects must
    not be able to talk the room into greeting them again."""
    session = a_story(FakeLLM(), curtain=False)
    session.raise_curtain()
    was = lines(session)

    assert session.raise_curtain() == []
    assert lines(session) == was


def test_the_player_still_introduces_themselves_once():
    """The look is said at the story's first moment and never again, and
    it is in the deferred half because it belongs after the room is
    described rather than before it."""
    session = a_story(
        FakeLLM(), curtain=False,
        player=Player(name="Wren", look="A tall girl in a borrowed coat."),
    )

    session.raise_curtain()

    said = lines(session)
    assert "A tall girl in a borrowed coat." in said
    assert said.index("A tall girl in a borrowed coat.") > 0, "after the room, not before"


# --- when it fails ------------------------------------------------------

def test_a_curtain_that_fails_leaves_the_scene_at_its_first_words():
    """Half an opening is the worst outcome available: a room described
    to nobody, or two of four people having heard a greeting the others
    never will. It is one unit of work for the same reason a turn is."""
    session = a_story(Refuses(), curtain=False)

    with pytest.raises(ModelUnavailable):
        session.raise_curtain()

    assert lines(session) == [session.scene.opening.strip()]


def test_a_trouble_frame_is_the_service_talking_and_not_a_character():
    """It exists because the opening runs after its response has gone,
    so a failure there has no reply to travel back on."""
    assert set(Trouble.model_fields) == {"detail"}


# --- one thing at a time ------------------------------------------------

def test_a_first_line_typed_over_the_opening_waits_for_it():
    """The composer is live the instant somebody lands, which is the
    whole point — and it means two pieces of engine work can now be
    asked for at once for the first time. A turn is a savepoint, so two
    of them interleaved would settle each other's half-written work.
    """
    parked, release = threading.Event(), threading.Event()

    class Parks(FakeLLM):
        """Stops inside the curtain's first call, and writes down the
        order it was asked in."""

        def __init__(self):
            super().__init__()
            self.calls = []
            self.lock = threading.Lock()

        def complete(self, system, prompt, key=None):
            with self.lock:
                first = not self.calls
                self.calls.append(threading.current_thread().name)
            if first:
                parked.set()
                release.wait(5)
            return super().complete(system, prompt, key)

    llm = Parks()
    session = a_story(llm, curtain=False)

    curtain = threading.Thread(target=session.raise_curtain, name="curtain")
    curtain.start()
    assert parked.wait(5), "the opening never reached the model"

    said = threading.Thread(target=lambda: session.say("Hello."), name="player")
    said.start()
    # Long enough that an unguarded turn would have bid by now: it is the
    # curtain that is blocked, not the machine.
    said.join(timeout=0.5)
    asked_while_blocked = list(llm.calls)

    release.set()
    curtain.join(5)
    said.join(5)

    assert "player" not in asked_while_blocked, (
        "the player's line was being written while the room was still opening"
    )
    assert "player" in llm.calls, "and it did play, once the curtain was up"


# --- through the service ------------------------------------------------

def test_nothing_of_the_opening_is_lost_to_a_client_that_connects_late(tmp_path):
    """The curtain goes up whether or not anybody is streaming yet, so
    the catch-up every stream begins with is what makes that safe."""
    app = create_app(
        worlds_root=ASHGROVE.parent, llm=FakeLLM(), library_root=tmp_path / "stories"
    )
    with TestClient(app) as client:
        begun = client.post("/stories", json={"world": "ashgrove", "scene": "the_dinner"})
        session_id = begun.json()["session_id"]

        caught_up = client.get(f"/sessions/{session_id}/stream?follow=false").text

    # The room described itself and somebody spoke, and all of it was
    # waiting for a client that had not arrived when it happened.
    assert caught_up.count("data:") > 1
    assert "caught up" in caught_up


def test_the_client_listens_for_trouble():
    """A room that never says anything reads as the engine being slow.
    The frame is only worth having if the page acts on it."""
    page = (Path(__file__).parent.parent / "fabula" / "web" / "index.html").read_text(
        encoding="utf-8"
    )

    assert 'addEventListener("trouble"' in page
