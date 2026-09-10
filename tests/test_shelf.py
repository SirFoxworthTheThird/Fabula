"""Where the worlds are when nobody said, and why it is not here.

`worlds_root` defaulted to the relative path `worlds` — a directory that
exists in a clone of this repository and nowhere else. So the app had a
shelf when you ran it from the checkout and an empty one everywhere
else, and `--invent` wrote a generated world into whatever directory you
happened to be standing in. Both are the same bug: an app that only
works from its own source tree is not installed, it is checked out.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fabula.api import create_app
from fabula.llm import FakeLLM
from fabula.loader import catalogue, load_world
from fabula.shelf import DELIVERED, SHIPPED, shipped_worlds, stocked

from tests.test_invent import Scripted


def a_world(root: Path, name: str) -> Path:
    """The smallest thing `shipped_worlds` counts as one."""
    (root / name).mkdir(parents=True)
    (root / name / "world.yaml").write_text(f"id: {name}\nname: {name}\n", encoding="utf-8")
    return root / name


# --- what ships -----------------------------------------------------------

def test_the_worlds_are_inside_the_package():
    """Not beside it. A wheel carries what is under `fabula/`, and the
    shelf has to be in the wheel or an installed copy opens on nothing."""
    import fabula

    assert SHIPPED.parent == Path(fabula.__file__).parent
    assert [world.name for world in shipped_worlds()], "nothing ships"


def test_every_shipped_world_loads():
    """A world that ships broken is worse than one that does not ship:
    `catalogue` leaves an unreadable world out silently, so it would be
    missing from the shelf with nothing said."""
    for world in shipped_worlds():
        assert load_world(world).id == world.name


def test_the_app_has_a_shelf_from_a_directory_that_is_not_a_clone(tmp_path, monkeypatch):
    monkeypatch.setattr("fabula.shelf.YOURS", tmp_path / "yours")
    monkeypatch.chdir(tmp_path)

    app = create_app(llm=FakeLLM(), library_root=tmp_path / "stories")
    with TestClient(app) as client:
        shelf = client.get("/catalogue").json()

    assert {world["id"] for world in shelf} == {world.name for world in shipped_worlds()}


# --- handing them over ----------------------------------------------------

def test_an_empty_home_gets_everything(tmp_path):
    yours = stocked(tmp_path / "yours", SHIPPED)

    assert sorted(p.name for p in yours.iterdir() if p.is_dir()) == [
        world.name for world in shipped_worlds()
    ]


def test_a_world_you_changed_is_yours_now(tmp_path):
    """The reason they are copied rather than read out of site-packages:
    a world is a directory of YAML somebody should be able to open and
    change. Which costs the other thing — we cannot fix one afterwards."""
    shipped = tmp_path / "shipped"
    a_world(shipped, "ashgrove")
    yours = stocked(tmp_path / "yours", shipped)
    (yours / "ashgrove" / "world.yaml").write_text("id: ashgrove\nname: Mine\n", encoding="utf-8")

    stocked(tmp_path / "yours", shipped)

    assert "Mine" in (yours / "ashgrove" / "world.yaml").read_text()


def test_a_world_you_deleted_stays_deleted(tmp_path):
    """What is written down is what has been *handed over*, not what is
    there — otherwise deleting a world means it is back on the next
    launch, which is the app arguing with you about your own directory."""
    shipped = tmp_path / "shipped"
    a_world(shipped, "ashgrove")
    a_world(shipped, "vilamar")
    yours = stocked(tmp_path / "yours", shipped)

    import shutil
    shutil.rmtree(yours / "vilamar")
    stocked(tmp_path / "yours", shipped)

    assert not (yours / "vilamar").exists()


def test_a_world_added_in_a_later_version_still_arrives(tmp_path):
    """The other half of the same record: never delivered means deliver."""
    shipped = tmp_path / "shipped"
    a_world(shipped, "ashgrove")
    stocked(tmp_path / "yours", shipped)

    a_world(shipped, "ardenhall")
    yours = stocked(tmp_path / "yours", shipped)

    assert (yours / "ardenhall" / "world.yaml").is_file()


def test_the_record_is_not_mistaken_for_a_world(tmp_path):
    yours = stocked(tmp_path / "yours", SHIPPED)

    assert (yours / DELIVERED).is_file()
    assert {world["id"] for world in catalogue(yours)} == {
        world.name for world in shipped_worlds()
    }


def test_no_default_anywhere_is_a_relative_path(tmp_path):
    """The bug, stated as the shape of the bug rather than as one of its
    symptoms: a default that is a relative path is a bet that the caller
    is standing in this repository. Two more were hiding behind the one
    that was found."""
    import inspect

    from fabula.api import create_app, serve
    from fabula.cli import main as cli_main
    from fabula.invent import invent
    from fabula.library import Library

    for func in (create_app, serve, invent, Library.__init__, cli_main):
        for name, parameter in inspect.signature(func).parameters.items():
            if "world" not in name and "root" not in name:
                continue
            given = parameter.default
            if given in (inspect.Parameter.empty, None):
                continue
            assert Path(given).is_absolute(), f"{func.__name__}({name}={given!r})"


def test_a_home_that_cannot_be_written_still_gets_a_shelf(tmp_path):
    """A read-only shelf is a working app. Refusing to start because
    somebody's home directory is odd is not."""
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding="utf-8")

    assert stocked(blocked / "yours", SHIPPED) == SHIPPED


def test_an_invented_world_lands_in_yours_rather_than_where_you_stood(tmp_path, monkeypatch):
    """The same bug from the writing side: a generated world used to be
    written into the current directory."""
    monkeypatch.setattr("fabula.shelf.YOURS", tmp_path / "yours")
    monkeypatch.chdir(tmp_path)

    app = create_app(llm=Scripted(), library_root=tmp_path / "stories")
    with TestClient(app) as client:
        made = client.post("/invent", json={"premise": "a heist in a hotel kitchen"})

    assert made.status_code == 200, made.text
    landed = {p.name for p in (tmp_path / "yours").iterdir() if p.is_dir()}
    assert landed - {world.name for world in shipped_worlds()}, "nothing new was written"
    assert not (tmp_path / "worlds").exists(), "it wrote into the current directory"
