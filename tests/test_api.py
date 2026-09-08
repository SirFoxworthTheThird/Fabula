"""M6: the HTTP service and its POV protocol (spec §12).

The protocol is the product surface for every future client, so the leak
invariant has to hold at the wire, not just in the engine.
"""
import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from fabula.api import create_app

from tests.conftest import ASHGROVE

SECRET = "music box"


@pytest.fixture
def client(fake_llm):
    app = create_app(worlds_root=ASHGROVE.parent, llm=fake_llm)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_id(client):
    response = client.post("/sessions", json={"world": "ashgrove", "scene": "the_dinner"})
    assert response.status_code == 200
    return response.json()["session_id"]


def test_worlds_are_discoverable(client):
    worlds = client.get("/worlds").json()

    assert "ashgrove" in worlds
    assert "the_dinner" in worlds["ashgrove"]


def test_opening_a_session_reports_the_users_own_pov(client):
    state = client.post("/sessions", json={"world": "ashgrove", "scene": "the_dinner"}).json()

    assert state["character"] == "elena"
    assert state["character_name"] == "Elena"
    assert state["location"] == "kitchen"
    assert state["location_name"] == "the kitchen"
    assert set(state["cast"]) == {"tomas", "maria", "elena"}


def test_a_world_name_cannot_escape_the_worlds_root(client):
    for attempt in ("../..", "../worlds", "/etc", "..%2f..", "ashgrove/../.."):
        response = client.post("/sessions", json={"world": attempt, "scene": "the_dinner"})
        assert response.status_code == 404, attempt


def test_unknown_world_and_scene_are_404(client):
    assert (
        client.post("/sessions", json={"world": "nowhere", "scene": "the_dinner"}).status_code
        == 404
    )
    assert (
        client.post("/sessions", json={"world": "ashgrove", "scene": "no_such"}).status_code
        == 404
    )


def test_speaking_returns_structured_pov_events(client, session_id):
    events = client.post(
        f"/sessions/{session_id}/say", json={"text": "Tomas, you have been quiet."}
    ).json()

    assert events
    for event in events:
        assert event["visibility"] in ("full", "degraded")
        assert event["location"] in ("kitchen", "study")
        assert "content" in event and event["content"]
        assert set(event) >= {"seq", "kind", "speaker", "location", "visibility", "content"}


def test_the_protocol_never_carries_an_unperceived_event(client, session_id):
    """Elena steps out; Tomás says the secret in the kitchen. Nothing on
    the wire — response, history, or stream — may carry it."""
    client.post(f"/sessions/{session_id}/move", json={"room": "study"})

    # Reach past the API to make Tomás say it, exactly as the engine would.
    session = client.app.state.sessions[session_id].session
    session.store.append_event(
        session.director.build_event(
            "utterance",
            "tomas",
            "kitchen",
            "I broke Grandma's music box, and I let them blame the cat.",
        )
    )

    history = client.get(f"/sessions/{session_id}/events").json()
    assert all(SECRET not in event["content"].lower() for event in history)
    assert all(event["visibility"] != "none" for event in history)

    body = client.post(f"/sessions/{session_id}/say", json={"text": "Anyone there?"}).json()
    assert all(SECRET not in event["content"].lower() for event in body)

    frames = _read_stream(client, session_id)
    assert frames
    assert all(SECRET not in frame["content"].lower() for frame in frames)


def test_history_is_the_scene_as_this_character_experienced_it(client, session_id):
    client.post(f"/sessions/{session_id}/say", json={"text": "Tomas, hello."})

    history = client.get(f"/sessions/{session_id}/events").json()

    # The scene opens with a line of its own, so the player's is not the
    # first thing in the log — only the first thing they said.
    spoken = [event for event in history if event["speaker"]]
    assert spoken[0]["speaker"] == "elena"
    assert spoken[0]["content"] == "Tomas, hello."
    assert history[0]["speaker"] is None  # the scene establishing itself
    assert [event["seq"] for event in history] == sorted(event["seq"] for event in history)


def test_moving_changes_where_the_user_is(client, session_id):
    client.post(f"/sessions/{session_id}/move", json={"room": "study"})

    state = client.get(f"/sessions/{session_id}").json()

    assert state["location"] == "study"


def test_moving_to_a_room_that_does_not_exist_is_rejected(client, session_id):
    response = client.post(f"/sessions/{session_id}/move", json={"room": "cellar"})

    assert response.status_code == 400


def test_the_next_skip_is_reported_so_a_client_can_ask_first(client, session_id):
    state = client.get(f"/sessions/{session_id}").json()

    # Derived from intentions, not arbitrary — and only from those that
    # could actually resolve. Tomás's are private and he is in the room
    # with the player, so the next reachable moment is Maria's.
    assert state["pending_skip_minutes"] == 30


def test_a_large_skip_is_refused_without_consent_in_the_request(client, session_id):
    """The protocol makes consent explicit rather than hiding it: a client
    reads how long the skip would be, then says yes."""
    from fabula.chronology import LARGE_SKIP_MINUTES
    from fabula.models import Intention

    session = client.app.state.sessions[session_id].session
    far_off = Intention(
        id="much_later",
        description="finally goes up to bed",
        location_id="study",
        ready_after_minutes=LARGE_SKIP_MINUTES + 120,
    )
    session.characters["maria"] = session.characters["maria"].model_copy(
        update={"intentions": [far_off]}
    )
    session.characters["tomas"] = session.characters["tomas"].model_copy(
        update={"intentions": []}
    )

    assert client.get(f"/sessions/{session_id}").json()["pending_skip_minutes"] == 180

    declined = client.post(f"/sessions/{session_id}/wait", json={"consent": False}).json()
    assert declined == []
    assert client.get(f"/sessions/{session_id}").json()["pending_skip_minutes"] == 180

    client.post(f"/sessions/{session_id}/wait", json={"consent": True})
    assert client.get(f"/sessions/{session_id}").json()["pending_skip_minutes"] is None


def test_presence_names_only_who_is_actually_here(client, session_id):
    """A POV client needs to show who you are with. It must never be
    answered with where the rest of the cast is."""
    state = client.get(f"/sessions/{session_id}").json()

    # Elena starts in the kitchen with Tomás; Maria is a room away.
    assert [person["id"] for person in state["present"]] == ["tomas"]

    client.post(f"/sessions/{session_id}/move", json={"room": "study"})
    moved = client.get(f"/sessions/{session_id}").json()

    assert "tomas" not in [person["id"] for person in moved["present"]]
    assert all(person["id"] != "elena" for person in moved["present"])  # never yourself


def test_presence_follows_a_character_who_leaves(client, session_id):
    session = client.app.state.sessions[session_id].session
    director = session.director

    assert [p["id"] for p in client.get(f"/sessions/{session_id}").json()["present"]] == ["tomas"]

    # Tomás steps out to the study; Elena is alone whether or not she saw him go.
    session.store.append_event(
        director.build_event("arrival", "tomas", "study", "Tomás steps through.")
    )

    assert client.get(f"/sessions/{session_id}").json()["present"] == []


def test_scene_state_carries_the_map_and_the_clock(client, session_id):
    state = client.get(f"/sessions/{session_id}").json()

    assert {room["id"] for room in state["rooms"]} == {"kitchen", "study"}
    assert state["story_time"].startswith("2024-01-01T19:00")


def test_the_client_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "<title>Fabula</title>" in body
    # It is a POV client: it reads the per-character stream, and there is
    # no world-log endpoint for it to read instead.
    assert "/stream" in body
    assert "degraded" in body  # half-heard events are rendered as uncertain


def test_the_reveal_is_asked_for_and_never_pushed(client, session_id):
    """The one endpoint that steps outside the requesting character's POV.
    It is a POST a client has to make deliberately, and it appears in no
    scene state and no stream."""
    client.post(f"/sessions/{session_id}/move", json={"room": "study"})
    session = client.app.state.sessions[session_id].session
    session.store.append_event(
        session.director.build_event(
            "utterance", "tomas", "kitchen", "I broke Grandma's music box."
        )
    )

    # Nothing pushed at the player carries it.
    state = client.get(f"/sessions/{session_id}").json()
    assert SECRET not in json.dumps(state).lower()
    history = client.get(f"/sessions/{session_id}/events").json()
    assert all(SECRET not in event["content"].lower() for event in history)

    revealed = client.post(f"/sessions/{session_id}/reveal").json()

    assert revealed["character"] == "Elena"
    assert any(SECRET in m["content"].lower() for m in revealed["missed"])
    assert revealed["knowledge"]["music_box"]["Maria"] is False
    assert SECRET in revealed["text"].lower()


def test_revealing_does_not_disturb_the_scene(client, session_id):
    client.post(f"/sessions/{session_id}/say", json={"text": "Tomas?"})
    before = client.get(f"/sessions/{session_id}/events").json()

    client.post(f"/sessions/{session_id}/reveal")

    assert client.get(f"/sessions/{session_id}/events").json() == before


def test_the_client_offers_the_reveal_behind_a_confirmation(client):
    body = client.get("/").text

    assert "/reveal" in body
    assert "spoils the scene" in body  # asked for, never sprung on the player


def test_unknown_session_is_404(client):
    assert client.get("/sessions/nope").status_code == 404
    assert client.post("/sessions/nope/say", json={"text": "hi"}).status_code == 404


def test_stream_sends_sse_frames(client, session_id):
    client.post(f"/sessions/{session_id}/say", json={"text": "Tomas, hello."})

    frames = _read_stream(client, session_id)

    assert frames
    spoken = [frame for frame in frames if frame["speaker"]]
    assert spoken[0]["speaker"] == "elena"
    assert spoken[0]["visibility"] == "full"


def test_live_subscribers_receive_each_turn_as_it_happens(client, session_id):
    """The follow path: a turn run through the API fans out to whoever is
    already streaming."""
    import asyncio

    entry = client.app.state.sessions[session_id]
    queue: asyncio.Queue = asyncio.Queue()
    entry.subscribers.append(queue)

    client.post(f"/sessions/{session_id}/say", json={"text": "Tomas, hello."})

    delivered = []
    while not queue.empty():
        delivered.append(queue.get_nowait())

    assert delivered
    # Their own line first, then whatever it provoked — the same transcript
    # the catch-up stream and /events would give.
    assert delivered[0].speaker == "elena"
    assert delivered[0].content == "Tomas, hello."
    assert len(delivered) > 1
    assert all(event.visibility in ("full", "degraded") for event in delivered)


def _read_stream(client, session_id) -> list[dict]:
    """Read the catch-up portion of the SSE stream."""
    response = client.get(f"/sessions/{session_id}/stream?follow=false")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return [
        json.loads(line[len("data: ") :])
        for line in response.text.split("\n")
        if line.startswith("data: ")
    ]


def test_the_reveal_reports_regard_that_moved(client):
    """The GUI needs it on the wire, and it is one of the few things in
    the scene the player provably could not have seen."""
    opened = client.post(
        "/sessions", json={"world": "ashgrove", "scene": "the_reckoning"}
    ).json()
    session_id = opened["session_id"]
    client.post(
        f"/sessions/{session_id}/say",
        json={"text": "What happened to Grandma's music box?"},
    )

    reveal = client.post(f"/sessions/{session_id}/reveal").json()
    moved = {(r["who"], r["toward"]): r for r in reveal["regard"]}

    assert ("Maria", "Tomás") in moved
    assert moved[("Maria", "Tomás")]["trust"] < moved[("Maria", "Tomás")]["started"]
    assert not [r for r in reveal["regard"] if r["who"] == reveal["character"]]


def test_a_retake_reuses_the_sequence_numbers_over_http(client):
    """A streaming client renders by seq, so a retake that appended
    instead of replacing would show the discarded take and the new one
    side by side."""
    opened = client.post("/sessions", json={"world": "ashgrove", "scene": "the_reckoning"}).json()
    session_id = opened["session_id"]
    first = client.post(f"/sessions/{session_id}/say", json={"text": "Tomás?"}).json()

    again = client.post(f"/sessions/{session_id}/regenerate").json()

    assert [e["seq"] for e in again] == [e["seq"] for e in first]
    body = client.get(f"/sessions/{session_id}/stream?follow=false").text
    seqs = [json.loads(line[6:])["seq"] for line in body.split("\n") if line.startswith("data: ")]
    assert seqs == sorted(set(seqs))  # no duplicates, nothing orphaned


def test_the_stream_is_told_where_to_truncate(client):
    """A retake can legitimately perceive nothing, so the instruction to
    drop the tail is its own frame rather than a field on an event — a
    client watching only events would keep showing the discarded take."""
    from fabula.api import Retake

    opened = client.post("/sessions", json={"world": "ashgrove", "scene": "the_reckoning"}).json()
    session_id = opened["session_id"]
    client.post(f"/sessions/{session_id}/say", json={"text": "Tomás?"})

    entry = client.app.state.sessions[session_id]
    queue = asyncio.Queue()
    entry.subscribers.append(queue)
    try:
        client.post(f"/sessions/{session_id}/regenerate")
        published = []
        while not queue.empty():
            published.append(queue.get_nowait())
    finally:
        entry.subscribers.remove(queue)

    assert isinstance(published[0], Retake)
    # Back to where the turn began, which is after the scene's opening
    # line — a retake rewinds a turn, not the scene.
    assert published[0].from_seq == 2
    assert all(event.seq >= published[0].from_seq for event in published[1:])


def test_regenerating_before_playing_is_refused(client, session_id):
    assert client.post(f"/sessions/{session_id}/regenerate").status_code == 409
