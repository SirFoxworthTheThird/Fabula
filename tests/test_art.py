"""A cover for every story, and a face for everybody in it.

The client was prose on a page, which is what a story is and also what a
terminal is. The shelf this app sits on is one people browse with their
eyes, and being the one that is all text is not principled, it is bare.

The obvious version — an `image:` pointing at a file — fixes it only for
a world somebody drew art for, which is none of the generated ones. So
the authored file is the *override* and the fallback is drawn from the
id: deterministic, in SVG, with no files to ship, nothing fetched from
anywhere, and no world without a picture.
"""
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from fabula.api import create_app
from fabula.art import KINDS, authored, cover, portrait
from fabula.inspect import complaints
from fabula.llm import FakeLLM
from fabula.loader import load_characters, load_world

from tests.conftest import ASHGROVE

WORLDS = ASHGROVE.parent


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(
        WORLDS, llm=FakeLLM(), library_root=tmp_path / "stories",
        settings_path=tmp_path / "settings.json",
    ))


@pytest.fixture
def world(tmp_path):
    """A copy of ashgrove to point at art in, or at nothing."""
    copy = tmp_path / "worlds" / "ashgrove"
    shutil.copytree(ASHGROVE, copy)
    return copy


def edit(path: Path, change):
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(written)
    path.write_text(yaml.safe_dump(written, allow_unicode=True), encoding="utf-8")


# --- what gets drawn when nobody drew anything --------------------------

def test_the_same_id_is_always_the_same_picture():
    """Across processes, not just within one. Seeding from `hash()` would
    repaint everybody every time the service restarted, which is the one
    thing a face must not do."""
    assert cover("ashgrove") == cover("ashgrove")
    assert portrait("ashgrove", "tomas") == portrait("ashgrove", "tomas")


def test_two_worlds_do_not_look_alike():
    assert cover("ashgrove") != cover("winterlight")


def test_a_cast_is_told_apart_at_a_glance():
    """The plate's whole job in a scene is saying who just spoke, so the
    hues have to be spaced rather than hashed independently. Measured:
    hashing them gave `winterlight` five pinks."""
    cast = ["ana", "ilse", "nadia", "petra", "yusuf"]
    drawn = [portrait("winterlight", who, among=(i, len(cast))) for i, who in enumerate(cast)]

    assert len(set(drawn)) == len(cast)
    hues = sorted(_hue_of(one) for one in drawn)
    apart = [b - a for a, b in zip(hues, hues[1:])]
    assert min(apart) > 25, f"two of them are nearly the same colour: {hues}"


def test_but_a_cast_still_belongs_to_its_world():
    """Spaced from each other and anchored to the world, so Ashgrove's
    people look like Ashgrove's people."""
    here = [_hue_of(portrait("ashgrove", c, among=(i, 3))) for i, c in enumerate("abc")]
    there = [_hue_of(portrait("winterlight", c, among=(i, 3))) for i, c in enumerate("abc")]

    assert here != there


def test_a_name_cannot_put_markup_in_the_plate():
    """A character's name can be one the *player* typed, and this one goes
    into an attribute."""
    drawn = portrait("w", "c", name='" onload="alert(1)')

    assert 'onload="alert' not in drawn
    assert "&quot;" in drawn


# --- the authored override, and the path it is allowed to name ----------

def test_a_shipped_world_may_point_at_real_art():
    found = authored(ASHGROVE, load_world(ASHGROVE).image)

    assert found is not None
    path, kind = found
    assert path.is_file() and kind == "image/svg+xml"


def test_a_path_that_leaves_the_world_is_not_served(world):
    """A world directory is content, not code: generated here, copied off
    somebody's machine, downloaded. So the path inside it is treated the
    way any other path from outside would be."""
    (world.parent / "secret.png").write_bytes(b"not yours")

    assert authored(world, "../secret.png") is None
    assert authored(world, "/etc/passwd") is None
    assert authored(world, "art/../../secret.png") is None


def test_a_prefix_of_the_world_is_not_inside_it(tmp_path):
    """`/worlds/ash` is a string prefix of `/worlds/ashgrove_notes`, which
    is why this compares paths rather than strings."""
    (tmp_path / "ash").mkdir()
    (tmp_path / "ash_notes").mkdir()
    (tmp_path / "ash_notes" / "c.png").write_bytes(b"x")

    assert authored(tmp_path / "ash", "../ash_notes/c.png") is None


def test_only_things_a_browser_can_render(world):
    (world / "notes.txt").write_text("x", encoding="utf-8")
    (world / "keys.env").write_text("OPENAI_API_KEY=sk-x", encoding="utf-8")

    assert authored(world, "notes.txt") is None
    assert authored(world, "keys.env") is None
    assert set(KINDS) >= {".png", ".jpg", ".webp", ".svg"}


def test_art_that_was_pointed_at_and_never_shipped_is_a_complaint(world):
    """Silent otherwise: the service falls back to the drawn plate, so a
    broken path looks exactly like a world that never had a picture and
    the author is left wondering why theirs is not showing."""
    edit(world / "world.yaml", lambda w: w.update({"image": "art/nope.png"}))

    assert any("not an image inside this world" in c for c in complaints(world))


def test_and_so_is_one_that_reaches_outside(world):
    edit(world / "characters" / "tomas.yaml", lambda c: c.update({"image": "../../secret.png"}))

    assert any("tomas" in c and "not an image" in c for c in complaints(world))


def test_a_world_with_no_art_at_all_is_still_playable(world):
    """Which is every generated world, and three of the four here."""
    edit(world / "world.yaml", lambda w: w.pop("image", None))

    assert complaints(world) == []
    assert load_world(world).image == ""


# --- one URL per thing, always answering --------------------------------

def test_every_world_answers_with_a_cover(client):
    for world_id in ("ashgrove", "winterlight", "ardenhall", "vilamar"):
        answered = client.get(f"/worlds/{world_id}/cover")
        assert answered.status_code == 200, world_id
        assert answered.headers["content-type"].startswith("image/")


def test_everybody_answers_with_a_face(client):
    for character in load_characters(ASHGROVE):
        answered = client.get(f"/worlds/ashgrove/faces/{character}")
        assert answered.status_code == 200, character
        assert answered.headers["content-type"].startswith("image/")


def test_the_authored_file_wins_over_the_drawn_one(client):
    answered = client.get("/worlds/ashgrove/cover")

    assert answered.content == (ASHGROVE / "art" / "cover.svg").read_bytes()


def test_a_world_with_no_file_is_drawn_rather_than_missing(client):
    answered = client.get("/worlds/winterlight/cover")

    assert answered.status_code == 200
    assert answered.text == cover("winterlight", load_world(WORLDS / "winterlight").title)


def test_nobody_gets_a_face_who_is_not_in_the_cast(client):
    assert client.get("/worlds/ashgrove/faces/nobody").status_code == 404


def test_the_art_endpoints_take_a_name_and_never_a_path(client):
    """The same rule the rest of the service follows: a world is a name
    under the worlds root, so this can never become a file reader."""
    for reached in ("../../etc", "..%2f..%2fetc", "/etc", "ashgrove/../../etc"):
        assert client.get(f"/worlds/{reached}/cover").status_code == 404, reached


# --- and the client is told which world it is in ------------------------

def test_a_scene_says_which_world_it_is_so_its_art_can_be_asked_for(client):
    opened = client.post("/sessions", json={"world": "ashgrove", "scene": "the_dinner"})

    assert opened.json()["world"] == "ashgrove"


def _hue_of(svg: str) -> float:
    """The first hue in a plate. Not the figure's own — it is the rim
    light, which is a fixed turn from it — so this compares plates rather
    than reading a value out of one."""
    import re

    return float(re.search(r"hsl\((\d+)", svg).group(1))
