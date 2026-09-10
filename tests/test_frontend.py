"""The way in.

The app had a browser client from M7 and nothing that led to it: you
installed a roleplay app, typed its name, and got a REPL. What the shelf
offered was every scene in every world — including chapter three, which
nobody should start cold — with no word about what any of them was.

These are about the front door, and about the shelf being made of the
author's words rather than of directory names.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fabula.api import create_app, free_port
from fabula.llm import FakeLLM
from fabula.loader import catalogue

from tests.conftest import ASHGROVE

WORLDS = ASHGROVE.parent


@pytest.fixture
def client(fake_llm, tmp_path):
    app = create_app(worlds_root=WORLDS, llm=fake_llm, library_root=tmp_path / "stories")
    with TestClient(app) as test_client:
        yield test_client


# --- Typing the name of the app opens the app -------------------------


def test_bare_fabula_starts_the_app(monkeypatch, tmp_path):
    """Not a REPL. The terminal is still there for anyone who wants it,
    behind a flag that says so."""
    from fabula import cli

    started = {}
    monkeypatch.setattr(cli, "serve", lambda **kwargs: started.update(kwargs))

    cli.main(["--worlds", str(WORLDS), "--library", str(tmp_path / "stories")])

    assert started["open_browser"] is True
    assert started["library_root"] == tmp_path / "stories"
    assert started["worlds_root"] == WORLDS


def test_the_terminal_is_still_there(monkeypatch, tmp_path, capsys):
    from fabula import cli

    monkeypatch.setattr(cli, "serve", lambda **kwargs: pytest.fail("should not serve"))

    cli.main(["--terminal", "--worlds", str(WORLDS), "--library", str(tmp_path / "s")])

    assert "No stories yet" in capsys.readouterr().out


def test_naming_a_scene_still_plays_it_here(monkeypatch, tmp_path):
    """`fabula ashgrove the_dinner` is the developer's door and does not
    become a web server."""
    from fabula import cli

    monkeypatch.setattr(cli, "serve", lambda **kwargs: pytest.fail("should not serve"))
    monkeypatch.setattr(cli, "run", lambda session, **kwargs: session.close())

    cli.main([
        "ashgrove", "the_dinner",
        "--worlds", str(WORLDS), "--library", str(tmp_path / "stories"),
    ])


def test_a_second_copy_takes_another_port():
    """Somebody who left a story open in another window should not be
    told to go and find it and close it."""
    import socket

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        busy = taken.getsockname()[1]
        taken.listen()

        assert free_port("127.0.0.1", busy) != busy

    assert free_port("127.0.0.1", 0) > 0


# --- A shelf of stories, not a list of scenes -------------------------


def test_a_scene_that_continues_a_story_is_not_an_opening():
    """Starting cold in chapter three is how a menu of scenes reads."""
    shelf = {world["id"]: world for world in catalogue(WORLDS)}

    ardenhall = {scene["id"]: scene for scene in shelf["ardenhall"]["scenes"]}
    assert ardenhall["arrival"]["opens"] is True
    assert ardenhall["the_interview"]["opens"] is False  # arrival leads here


def test_the_shelf_says_what_a_scene_is_and_who_you_are_in_it():
    shelf = {world["id"]: world for world in catalogue(WORLDS)}
    dinner = next(
        scene for scene in shelf["ashgrove"]["scenes"] if scene["id"] == "the_dinner"
    )

    assert dinner["title"] == "The dinner"
    assert dinner["premise"]
    assert dinner["you"] == "Elena"
    assert set(dinner["with"]) == {"Tomás", "Maria"}


def test_every_shipped_world_introduces_itself():
    """A guard on the authoring, not on the engine: a world with no title
    and no blurb is a directory name on a shelf, which is what the start
    screen used to be made of."""
    for world in catalogue(WORLDS):
        assert world["title"], world["id"]
        assert world["blurb"], world["id"]
        for scene in world["scenes"]:
            assert scene["title"], f"{world['id']}/{scene['id']}"
            assert scene["premise"], f"{world['id']}/{scene['id']}"
            assert scene["you"], f"{world['id']}/{scene['id']} has nobody to play"


def test_a_world_that_does_not_load_is_left_out_rather_than_taking_the_shelf_down(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "world.yaml").write_text("id: broken\nrooms: {}\n[")
    good = tmp_path / "ashgrove"
    good.symlink_to(ASHGROVE)

    shelf = catalogue(tmp_path)

    assert [world["id"] for world in shelf] == ["ashgrove"]


def test_the_shelf_is_served(client):
    shelf = client.get("/catalogue").json()

    assert {world["id"] for world in shelf} >= {"ashgrove", "winterlight"}
    assert any(scene["opens"] for scene in shelf[0]["scenes"])


def test_a_story_is_named_after_its_scene(client):
    """"The dinner", not "Ashgrove — the_dinner": the card should read
    like a story, and the world is already on the same line."""
    client.post("/stories", json={"world": "ashgrove", "scene": "the_dinner"})

    assert client.get("/stories").json()[0]["title"] == "The dinner"


# --- The page itself --------------------------------------------------


def test_the_client_is_one_file_with_no_build_step():
    """The app installs with pip and runs. A page that needed compiling
    would put a node toolchain between somebody and their story."""
    page = (Path(__file__).parent.parent / "fabula" / "web" / "index.html").read_text(
        encoding="utf-8"
    )

    import re

    assert "<script" in page
    # Every script is inline, so there is nothing to fetch and nothing to
    # build.
    assert re.search(r"<script[^>]*\bsrc=", page) is None
    for outside in ("cdn.", "unpkg", "jsdelivr", "googleapis"):
        assert outside not in page, f"the client must not fetch {outside}"


def test_the_client_never_turns_model_output_into_markup():
    """Every line the client shows is model output — a character's words,
    the narrator's prose, a name the narrator invented mid-scene. So the
    page sets `textContent` and never `innerHTML`, and the one thing that
    builds elements from a story string (`emphasised`) makes text nodes
    and `<em>` and nothing else.

    Checked structurally because it is the sort of rule that survives
    right up until somebody adds a feature that needs "just a little"
    markup. Probed for real in a browser too — `*<b>x</b>*` renders as an
    `<em>` containing the literal angle brackets, and an `<img onerror>`
    creates no element.
    """
    page = (Path(__file__).parent.parent / "fabula" / "web" / "index.html").read_text(
        encoding="utf-8"
    )

    # The usage form, not the bare word: the file explains at length why
    # it never touches this, and the explanation should not fail its own
    # test.
    assert ".innerHTML" not in page
    assert "insertAdjacentHTML" not in page
    assert "document.write" not in page
    # The renderer's only element is <em>; everything else it emits is a
    # text node.
    body = page.split("function emphasised(")[1].split("\nfunction ")[0]
    assert body.count("createElement(") == 1
    assert 'createElement("em")' in body


def test_the_client_never_reads_the_world_log():
    """It renders one character's projection. The endpoints that would
    hand it anything else are not endpoints it calls."""
    page = (Path(__file__).parent.parent / "fabula" / "web" / "index.html").read_text(
        encoding="utf-8"
    )

    # `/reveal` is the one omniscient thing here, and it is asked for by
    # a person who has been warned.
    assert "confirm(" in page.split('el("reveal").onclick')[1][:200]
