"""The story library: your stories, on your machine (CLAUDE.md).

`--db` made the state durable; it did not make it *yours*. A player has
no business knowing what a SQLite path is, and "resume the story I was
playing" is the thing they actually want. These tests are about that:
that a story can be found again, picked up where it was left, and thrown
away — without an account, and without the player naming a file.
"""
from pathlib import Path

import pytest

from fabula.llm import FakeLLM
from fabula.library import Library, StoryCard

from tests.conftest import ASHGROVE

WORLDS = ASHGROVE.parent


@pytest.fixture
def library(tmp_path):
    return Library(root=tmp_path / "stories", worlds_root=WORLDS)


def test_a_started_story_is_in_the_library(library):
    with library.start("ashgrove", "the_dinner", llm=FakeLLM()) as session:
        assert session.scene.id == "the_dinner"

    card = library.list()[0]
    assert card.world == "ashgrove"
    assert card.scene == "the_dinner"
    assert card.title == "The dinner", "named after the scene, in the author's words"
    assert card.turns == 0
    assert card.unplayed


def test_a_title_is_the_players_if_they_gave_one(library):
    with library.start("ashgrove", "the_dinner", title="Tuesday", llm=FakeLLM()):
        pass

    assert library.list()[0].title == "Tuesday"


def test_playing_a_story_counts_the_turns(library):
    with library.start("ashgrove", "the_dinner", llm=FakeLLM()) as session:
        session.say("Pass the salt.")
        session.say("Thank you.")

    card = library.list()[0]
    assert card.turns == 2
    assert not card.unplayed


def test_a_retake_is_the_same_turn_played_again(library):
    """`/again` rewinds the take, so it must rewind the count with it.

    Otherwise the library reports turns nobody played — and the count is
    the only sign a card gives of how far a story got.
    """
    with library.start("ashgrove", "the_dinner", llm=FakeLLM()) as session:
        session.say("Pass the salt.")
        session.regenerate()
        session.regenerate()

    assert library.list()[0].turns == 1


def test_resuming_lands_on_the_scene_it_was_left_on(library):
    """A story is resumed, not restarted: the scene it stopped on."""
    session = library.start("ardenhall", "arrival", llm=FakeLLM())
    story_id = session.store.get_story()["story_id"]
    session.say("I'm here.")
    later = session.go_on()
    assert later is not None, "arrival leads somewhere"
    left_on = later.scene.id
    later.close()
    assert left_on != "arrival"

    with library.resume(story_id, llm=FakeLLM()) as session:
        assert session.scene.id == left_on
    assert library.list()[0].scene == left_on


def test_a_resumed_story_carries_on_counting(library):
    with library.start("ashgrove", "the_dinner", llm=FakeLLM()) as session:
        story_id = session.store.get_story()["story_id"]
        session.say("Pass the salt.")

    with library.resume(story_id, llm=FakeLLM()) as session:
        session.say("Where were we?")

    assert library.list()[0].turns == 2


def test_a_resumed_story_remembers_what_was_said_in_it(library):
    """The whole point of the file: everybody arrives holding what they
    held when it was put down."""
    with library.start("ashgrove", "the_dinner", llm=FakeLLM()) as session:
        story_id = session.store.get_story()["story_id"]
        session.say("The rain has not stopped all week.")

    with library.resume(story_id, llm=FakeLLM()) as session:
        said = [p.perceived_content for p in session.perceived_so_far()]

    assert any("rain has not stopped" in text for text in said)


def test_the_most_recently_played_story_is_first(library):
    with library.start("ashgrove", "the_dinner", title="First", llm=FakeLLM()):
        pass
    with library.start("ashgrove", "the_dinner", title="Second", llm=FakeLLM()) as session:
        story_id = session.store.get_story()["story_id"]
    with library.resume(story_id, llm=FakeLLM()) as session:
        session.say("Still here.")

    assert [card.title for card in library.list()] == ["Second", "First"]


def test_deleting_removes_the_story(library):
    with library.start("ashgrove", "the_dinner", llm=FakeLLM()) as session:
        story_id = session.store.get_story()["story_id"]

    assert library.delete(story_id) is True
    assert library.list() == []
    assert library.delete(story_id) is False
    with pytest.raises(FileNotFoundError):
        library.resume(story_id)


def test_a_story_id_never_becomes_a_path(library):
    """Ids are minted here, never taken from a client — but this is the
    one place a bad one would be joined onto a filesystem path, and an
    HTTP route hands it straight in."""
    for bad in ["../../etc/passwd", "", "abcdef", "abcdef012345.sqlite", "ABCDEF012345", "../a1b2c3d4e5f6"]:
        with pytest.raises(ValueError):
            library.resume(bad)
        with pytest.raises(ValueError):
            library.delete(bad)


def test_a_file_that_is_not_a_story_is_skipped(library):
    """A library that refuses to open because of one bad file is worse
    than one that shows the rest."""
    with library.start("ashgrove", "the_dinner", title="Real", llm=FakeLLM()):
        pass
    (library.root / "junk.sqlite").write_bytes(b"not a database")
    (library.root / "notes.txt").write_text("nothing to do with anything")

    assert [card.title for card in library.list()] == ["Real"]


def test_a_story_that_fails_to_open_leaves_no_card(library):
    """The file is written before the scene is opened, so a world that
    does not load would otherwise leave a story nobody can resume."""
    with pytest.raises(Exception):
        library.start("ashgrove", "no_such_scene", llm=FakeLLM())

    assert library.list() == []
    assert list(library.root.glob("*.sqlite")) == []


def test_one_story_is_one_file(library):
    """Copy the file and the story goes with it — which is only true if
    a story is exactly one file."""
    with library.start("ashgrove", "the_dinner", llm=FakeLLM()) as session:
        story_id = session.store.get_story()["story_id"]
        session.say("Pass the salt.")

    files = list(library.root.glob("*"))
    assert [path.name for path in files] == [f"{story_id}.sqlite"]

    moved = Library(root=library.root.parent / "elsewhere", worlds_root=WORLDS)
    moved.root.mkdir()
    files[0].rename(moved.root / files[0].name)

    with moved.resume(story_id, llm=FakeLLM()) as session:
        assert any(
            "Pass the salt" in projected.perceived_content
            for projected in session.perceived_so_far()
        )


# --- Over the wire ----------------------------------------------------
#
# The library is a product surface, not an internal one: the web client
# is how most people will meet it, so the lifecycle has to hold at the
# wire too.


@pytest.fixture
def client(tmp_path, fake_llm):
    from fastapi.testclient import TestClient

    from fabula.api import create_app

    app = create_app(worlds_root=WORLDS, llm=fake_llm, library_root=tmp_path / "stories")
    with TestClient(app) as test_client:
        yield test_client


def test_the_story_lifecycle_over_http(client):
    assert client.get("/stories").json() == []

    started = client.post(
        "/stories", json={"world": "ashgrove", "scene": "the_dinner", "title": "Tuesday"}
    )
    assert started.status_code == 200
    session_id = started.json()["session_id"]
    client.post(f"/sessions/{session_id}/say", json={"text": "Pass the salt."})

    cards = client.get("/stories").json()
    assert len(cards) == 1
    assert cards[0]["title"] == "Tuesday"
    assert cards[0]["turns"] == 1
    story_id = cards[0]["id"]

    resumed = client.post(f"/stories/{story_id}/resume")
    assert resumed.status_code == 200

    assert client.delete(f"/stories/{story_id}").status_code == 200
    assert client.get("/stories").json() == []
    assert client.post(f"/stories/{story_id}/resume").status_code == 404


def test_resuming_an_open_story_hands_back_the_same_session(client):
    """Two sessions over one file would disagree about what happened in
    it, and the second would sit waiting on the first's write lock."""
    started = client.post("/stories", json={"world": "ashgrove", "scene": "the_dinner"})
    session_id = started.json()["session_id"]
    story_id = client.get("/stories").json()[0]["id"]

    resumed = client.post(f"/stories/{story_id}/resume")

    assert resumed.json()["session_id"] == session_id


def test_deleting_lets_go_of_an_open_story(client):
    """Otherwise the session outlives the file it was reading from."""
    started = client.post("/stories", json={"world": "ashgrove", "scene": "the_dinner"})
    session_id = started.json()["session_id"]
    story_id = client.get("/stories").json()[0]["id"]

    assert client.delete(f"/stories/{story_id}").status_code == 200

    assert client.get(f"/sessions/{session_id}").status_code == 404


def test_a_bad_story_id_is_rejected_at_the_wire(client):
    """The one route that hands a client-supplied string toward a path."""
    assert client.delete("/stories/..%2F..%2Fetc%2Fpasswd").status_code in (400, 404)
    assert client.post("/stories/nonsense/resume").status_code == 404
    assert client.post("/stories/a1b2c3d4e5f6/resume").status_code == 404


# --- At the terminal --------------------------------------------------


def test_the_listing_is_what_you_see_in_the_terminal(library, capsys):
    """`fabula` on its own opens the app in a browser; the terminal path
    shows the same thing in words."""
    from fabula.cli import main as cli_main

    argv = ["--terminal", "--library", str(library.root), "--worlds", str(WORLDS)]
    cli_main(argv)
    assert "No stories yet" in capsys.readouterr().out

    with library.start("ashgrove", "the_dinner", title="Tuesday", llm=FakeLLM()) as session:
        session.say("Pass the salt.")

    cli_main(argv)
    listed = capsys.readouterr().out
    assert "Tuesday" in listed
    assert "1 turn" in listed and "1 turns" not in listed


def test_a_mistyped_story_id_is_a_typo_not_a_crash(library):
    """The ids are copied off a listing by hand."""
    from fabula.cli import main as cli_main

    for argv in (["--delete", "nonsense"], ["--resume", "nonsense"]):
        with pytest.raises(SystemExit):
            cli_main(["--library", str(library.root), "--worlds", str(WORLDS), *argv])
