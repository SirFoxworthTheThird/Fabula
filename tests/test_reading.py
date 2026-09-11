"""Something in a room that carries it, and who gets to see what.

An item is a second channel for the same asymmetry the engine already
turns on: whoever reads it knows, whoever does not, does not, and the
room can see you reading without seeing what you read. It is the one way
a world can put a secret *somewhere* rather than in somebody.

All of it was built and none of it was reachable from the app. `/read`
existed in the terminal and nowhere else — no endpoint, no button — so
ardenhall's grey folder, the one item in the shipped worlds, could not be
opened by anybody playing in a browser. The same shape of failure as the
off-screen machinery: a mechanism with no trigger.
"""
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fabula.api import create_app
from fabula.llm import FakeLLM
from fabula.session import Session
from fabula.shelf import SHIPPED

ARDENHALL = SHIPPED / "ardenhall"


@pytest.fixture
def playing(tmp_path):
    app = create_app(
        worlds_root=SHIPPED, llm=FakeLLM(), library_root=tmp_path / "stories"
    )
    with TestClient(app) as client:
        begun = client.post(
            "/stories", json={"world": "ardenhall", "scene": "arrival"}
        ).json()
        yield client, begun["session_id"]


def state_of(client, session_id):
    return client.get(f"/sessions/{session_id}").json()


# --- that it is reachable at all -----------------------------------------

def test_the_room_says_what_is_in_it_to_read(playing):
    client, session_id = playing
    where = state_of(client, session_id)["location"]

    # Walk to wherever the folder is, if it is not here.
    session = Session.open(ARDENHALL, "arrival", llm=FakeLLM(),
                           db_path=str(Path(tempfile.mkdtemp()) / "s.sqlite"))
    rooms = {item.location_id for item in session.world.items.values()}
    session.close()
    assert rooms, "ardenhall has something in it to find"

    if where not in rooms:
        client.post(f"/sessions/{session_id}/move", json={"room": sorted(rooms)[0]})
    things = state_of(client, session_id)["things"]

    assert things, "standing in the room with it and the state does not mention it"
    assert all(set(thing) == {"id", "name"} for thing in things)


def test_the_state_never_carries_what_it_says(playing):
    """Names only. A client handed the text with the room state would
    have been handed it before anybody opened anything."""
    client, session_id = playing
    session = Session.open(ARDENHALL, "arrival", llm=FakeLLM(),
                           db_path=str(Path(tempfile.mkdtemp()) / "s.sqlite"))
    texts = [item.text for item in session.world.items.values()]
    rooms = {item.location_id for item in session.world.items.values()}
    session.close()

    client.post(f"/sessions/{session_id}/move", json={"room": sorted(rooms)[0]})
    body = client.get(f"/sessions/{session_id}").text

    for text in texts:
        for sentence in text.split(".")[:3]:
            if len(sentence.strip()) > 20:
                assert sentence.strip() not in body


def test_reading_it_is_two_things(playing):
    """The room sees somebody open it; what it says is addressed to the
    reader alone. One event each, through the ordinary perception path."""
    client, session_id = playing
    session = Session.open(ARDENHALL, "arrival", llm=FakeLLM(),
                           db_path=str(Path(tempfile.mkdtemp()) / "s.sqlite"))
    item = next(iter(session.world.items.values()))
    session.close()

    client.post(f"/sessions/{session_id}/move", json={"room": item.location_id})
    played = client.post(
        f"/sessions/{session_id}/read", json={"name": item.name}
    ).json()

    assert played, "reading it produced nothing"
    said = " ".join(event["content"] or "" for event in played)
    assert item.text.split(".")[0].strip() in said, "the reader did not get the text"


def test_reading_something_that_is_not_here_is_a_404_and_not_a_traceback(playing):
    client, session_id = playing

    refused = client.post(
        f"/sessions/{session_id}/read", json={"name": "a door into summer"}
    )

    assert refused.status_code == 404
    assert "a door into summer" in refused.json()["detail"]


def test_you_cannot_read_something_in_another_room(playing):
    """A name, resolved against what is actually here. Not an id from
    anywhere else."""
    client, session_id = playing
    session = Session.open(ARDENHALL, "arrival", llm=FakeLLM(),
                           db_path=str(Path(tempfile.mkdtemp()) / "s.sqlite"))
    item = next(iter(session.world.items.values()))
    session.close()
    here = state_of(client, session_id)["location"]
    assert here != item.location_id, "the fixture needs them to start apart"

    refused = client.post(f"/sessions/{session_id}/read", json={"name": item.name})

    assert refused.status_code == 404


def test_the_client_offers_what_is_in_the_room(playing):
    """A mechanism with no trigger is a fixture. The endpoint is only
    worth having if the page draws the button."""
    page = (Path(__file__).parent.parent / "fabula" / "web" / "index.html").read_text(
        encoding="utf-8"
    )

    assert 'id="things"' in page
    assert 'act("read"' in page
