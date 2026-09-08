"""The character the player brings.

Everything in a world is authored, and that is right for the parts a
story turns on. It is wrong for the one person the player is: being
handed Rook, or Elena, is being handed somebody else's character to
wear, in an app whose point is playing a story of your own.

Two fields, and each has to reach the fiction or it is decoration — a
name that only the interface uses is worse than no renaming at all,
because then your sister calls you Elena while the screen says Wren.
"""
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fabula.api import create_app
from fabula.library import Library
from fabula.llm import FakeLLM
from fabula.loader import load_scenario
from fabula.player import Player, rename
from fabula.session import Session

from tests.conftest import ASHGROVE

WORLDS = ASHGROVE.parent
ARDENHALL = WORLDS / "ardenhall"
WREN = Player(name="Wren Halloway", look="A tall girl in a coat two sizes too big.")


@pytest.fixture
def library(tmp_path):
    return Library(root=tmp_path / "stories", worlds_root=WORLDS)


@pytest.fixture
def client(tmp_path, fake_llm):
    app = create_app(
        worlds_root=WORLDS,
        llm=fake_llm,
        library_root=tmp_path / "stories",
        settings_path=tmp_path / "settings.json",
    )
    with TestClient(app) as test_client:
        yield test_client


# --- The name -------------------------------------------------------


def test_you_are_who_you_said_you_were():
    _, characters, _ = load_scenario(ARDENHALL, "arrival", WREN)

    assert characters["rook"].name == "Wren Halloway"


def test_everybody_else_calls_you_that_too():
    """The authored prose is where the old name actually lives: the
    Director's persona says he read Rook's file on Tuesday."""
    _, characters, _ = load_scenario(ARDENHALL, "arrival", WREN)

    assert "Rook" not in characters["vance"].persona
    assert "Wren" in characters["vance"].persona


def test_the_ids_underneath_are_untouched():
    """`rook:` in a relationship map is a key, not a name. Renaming it
    would break every reference the world makes to the character."""
    _, characters, scene = load_scenario(ARDENHALL, "arrival", WREN)

    assert "rook" in characters
    assert "rook" in characters["vance"].relationships
    assert "rook" in scene.cast


def test_a_bare_first_name_becomes_the_new_first_name():
    """Prose says "Ana Reyes" once and "Ana" after that."""
    said = rename("Ana Reyes came in. Ana looked cold.", "Ana Reyes", "Wren Halloway")

    assert said == "Wren Halloway came in. Wren looked cold."


def test_the_narrator_is_told_the_name_you_chose(fake_llm):
    """It is the one instruction that stops it playing your character
    for you, and it is by name."""
    session = Session.open(ARDENHALL, "arrival", llm=fake_llm, player=WREN)

    assert session.director.narrator.protagonist == "Wren Halloway"


def test_a_world_played_as_written_is_exactly_as_written():
    plain = load_scenario(ARDENHALL, "arrival")
    named = load_scenario(ARDENHALL, "arrival", Player())

    assert plain[1]["vance"].persona == named[1]["vance"].persona
    assert plain[1]["rook"].name == named[1]["rook"].name == "Rook"


# --- How you come across ---------------------------------------------


def test_the_room_sees_you(fake_llm):
    """It is an ordinary event, so it is perceived by the people standing
    there — which is what makes it something they can answer."""
    session = Session.open(ARDENHALL, "arrival", llm=fake_llm, player=WREN)
    events = session.store.get_events(session.scene.id)

    said = [e.content for e in events]
    assert WREN.look in said

    pell = session.director.contexts.project(session.characters["pell"], events)
    assert any(WREN.look == p.perceived_content for p in pell), "she is in the hall"


def test_and_nobody_who_is_not_in_the_room_does(fake_llm):
    session = Session.open(ARDENHALL, "arrival", llm=fake_llm, player=WREN)
    events = session.store.get_events(session.scene.id)

    vance = session.director.contexts.project(session.characters["vance"], events)

    assert not any(WREN.look in p.perceived_content for p in vance), "he is in the office"


def test_you_are_introduced_once_and_not_again(library):
    """It is how you arrived, not something you say at every door."""
    session = library.start("ardenhall", "arrival", llm=FakeLLM(), player=WREN)
    story_id = session.store.get_story()["story_id"]
    session.say("Where do I go?")
    later = session.go_on()
    assert later is not None
    assert not any(WREN.look in e.content for e in later.store.get_events(later.scene.id))
    later.close()

    resumed = library.resume(story_id, llm=FakeLLM())
    said = [e.content for e in resumed.store.get_events(resumed.scene.id)]
    assert said.count(WREN.look) == 0, "the second scene is not where they arrived"
    resumed.close()


def test_a_description_that_names_a_secret_is_not_appended(fake_llm):
    """It would hand the secret to everybody standing there before a word
    was spoken, and could end an arc on its first beat."""
    spoiler = Player(name="Kes", look="The one who broke the music box.")
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm, player=spoiler)

    said = [e.content for e in session.store.get_events(session.scene.id)]

    assert spoiler.look not in said


# --- What a name may be ----------------------------------------------


def test_a_name_cannot_reshape_the_yaml_it_goes_into():
    """It is substituted into authored YAML before it is parsed."""
    for bad in ['Kes"lin', "Kes: the Great", "Kes {x}", "Kes\nHalloway", "Kes #1"]:
        assert Player(name=bad).complaint(), bad


def test_a_name_in_any_script_is_a_name():
    """The engine speaks no language of its own, and neither does this."""
    for good in ("葉月", "Ana Reyes", "Wren Halloway", "Íris", "Ømer", "O'Hara-Smith"):
        assert Player(name=good).complaint() is None, good
    # An apostrophe is part of plenty of names and none of anybody's
    # business to refuse — but a straight one can close a quoted YAML
    # scalar, and the name is substituted into authored YAML.
    assert Player(name="O'Hara").called == "O\u2019Hara"


def test_a_description_is_one_line_and_not_an_essay():
    assert Player(look="a\nb").complaint()
    assert Player(look="x" * 500).complaint()
    assert Player(look="A tall girl in a wet coat.").complaint() is None


# --- It has to survive being put down --------------------------------


def test_a_resumed_story_is_still_your_character(library):
    session = library.start("ardenhall", "arrival", llm=FakeLLM(), player=WREN)
    story_id = session.store.get_story()["story_id"]
    session.say("Where do I go?")
    session.close()

    resumed = library.resume(story_id, llm=FakeLLM())

    assert resumed.user_character.name == "Wren Halloway"
    assert "Wren" in resumed.characters["vance"].persona
    assert library.list()[0].character == "Wren Halloway"
    resumed.close()


def test_an_older_story_file_still_opens(tmp_path):
    """The story table predates this. A file written before the columns
    existed must not fail to open."""
    from fabula.db import EventStore

    path = tmp_path / "old.sqlite"
    store = EventStore(str(path))
    store.conn.executescript(
        """DROP TABLE story;
           CREATE TABLE story (
               only_row INTEGER PRIMARY KEY CHECK (only_row = 1),
               story_id TEXT NOT NULL,
               title TEXT NOT NULL,
               world TEXT NOT NULL,
               scene TEXT NOT NULL,
               created_at TEXT NOT NULL,
               played_at TEXT NOT NULL,
               turns INTEGER NOT NULL DEFAULT 0
           );"""
    )
    store.conn.commit()
    store.conn.close()

    reopened = EventStore(str(path))
    reopened.start_story("a" * 12, "T", "ardenhall", "arrival", "Wren", "tall")

    assert reopened.get_story()["character_name"] == "Wren"
    reopened.close()


# --- Over the wire ---------------------------------------------------


def test_starting_a_story_as_somebody_of_your_own(client):
    opened = client.post(
        "/stories",
        json={
            "world": "ardenhall",
            "scene": "arrival",
            "character": {"name": "Wren Halloway", "look": "A tall girl in a wet coat."},
        },
    )

    assert opened.status_code == 200
    assert opened.json()["character_name"] == "Wren Halloway"
    assert client.get("/stories").json()[0]["character"] == "Wren Halloway"


def test_a_name_the_yaml_could_not_take_is_refused_with_a_reason(client):
    refused = client.post(
        "/stories",
        json={"world": "ardenhall", "scene": "arrival", "character": {"name": 'a"b: c'}},
    )

    assert refused.status_code == 400
    assert "quotes" in refused.json()["detail"]
    assert client.get("/stories").json() == [], "and no story was made"


def test_a_description_that_spoils_the_story_is_refused_with_a_reason(client):
    refused = client.post(
        "/stories",
        json={
            "world": "ashgrove",
            "scene": "the_dinner",
            "character": {"name": "Kes", "look": "The one who broke the music box."},
        },
    )

    assert refused.status_code == 400
    assert "turns on" in refused.json()["detail"]


def test_starting_without_one_is_still_one_click(client):
    opened = client.post("/stories", json={"world": "ashgrove", "scene": "the_dinner"})

    assert opened.status_code == 200
    assert opened.json()["character_name"] == "Elena"


# --- And in the terminal ---------------------------------------------


def test_the_terminal_can_do_it_too(tmp_path, monkeypatch):
    from fabula import cli

    played = {}
    monkeypatch.setattr(cli, "run", lambda session, **kwargs: played.update(session=session))
    cli.main([
        "ardenhall", "arrival",
        "--as", "Wren Halloway",
        "--look", "A tall girl in a wet coat.",
        "--worlds", str(WORLDS),
        "--library", str(tmp_path / "stories"),
    ])

    session = played["session"]
    assert session.user_character.name == "Wren Halloway"
    session.close()


def test_a_name_the_terminal_cannot_use_is_a_usage_error(tmp_path):
    from fabula import cli

    with pytest.raises(SystemExit):
        cli.main([
            "ardenhall", "arrival",
            "--as", 'a"b: c',
            "--worlds", str(WORLDS),
            "--library", str(tmp_path / "stories"),
        ])
