"""Rooms the author did not write.

A school has corridors. Nobody wants to write them all, and a story that
answers "there is no library to go to" is answering with its own
scaffolding.

What makes this cheap is that almost none of a room carries weight. The
one part that does is its adjacency, and the **engine** decides that: a
found place hangs off exactly the room it was reached from, by one
symmetric edge. A model that got to choose edges could join the library
to the headmaster's office, which would be a leak rather than a bad
sentence. So the model writes a name and two lines of prose, and prose
is checked the way every other generated line is.
"""
from pathlib import Path

import pytest

from fabula.discovery import discover, invents_a_fact, restore, slug
from fabula.llm import FakeLLM
from fabula.loader import load_world
from fabula.session import Session
from fabula.world import Room, World, connect, find_room

from fabula.shelf import SHIPPED as WORLDS  # noqa: E402
ARDENHALL = WORLDS / "ardenhall"
ASHGROVE = WORLDS / "ashgrove"


@pytest.fixture
def school(fake_llm):
    return Session.open(ARDENHALL, "arrival", llm=fake_llm)


# --- the edge, which is the only part that matters ----------------------

def test_a_found_room_hangs_off_exactly_where_it_was_reached_from(school):
    school.move("the library")

    library = school.world.rooms["the_library"]
    assert set(library.adjacent) == {"hall"}
    assert "the_library" in school.world.rooms["hall"].adjacent


def test_the_doorway_carries_both_ways(school):
    """A doorway is symmetric. One-way stays something an author does on
    purpose, not something discovery does by accident."""
    from datetime import datetime

    from fabula.models import Event
    from fabula.world import resolve_perception

    school.move("the library")
    world = school.world
    shout = Event(
        scene_id="s", seq=1, story_time=datetime(2024, 9, 2), kind="utterance",
        actor_id="a", location_id="hall", content="oi", audibility="building",
    )

    assert resolve_perception(shout, "b", "the_library", world) == "degraded"
    back = shout.model_copy(update={"location_id": "the_library"})
    assert resolve_perception(back, "b", "hall", world) == "degraded"


def test_the_map_is_a_tree_not_a_map(school):
    """A real limit rather than a bug, and the price of the engine keeping
    the edges: two found wings never join up."""
    school.move("the library")
    school.move("hall")
    school.move("the east stair")

    assert school.world.distance("the_library", "the_east_stair") == 2  # via the hall
    assert "the_east_stair" not in school.world.rooms["the_library"].adjacent


# --- only where the author allowed it ------------------------------------

def test_a_world_that_never_opted_in_is_unchanged(fake_llm):
    """A two-room house has no corridors to find."""
    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)

    assert session.world.discover_rooms is False
    with pytest.raises(ValueError, match="no room"):
        session.move("the library")


def test_the_school_opted_in():
    assert load_world(ARDENHALL).discover_rooms is True
    assert load_world(ASHGROVE).discover_rooms is False


# --- finding what is already there ---------------------------------------

def test_the_same_place_is_not_found_twice(school):
    school.move("the library")
    rooms = dict(school.world.rooms)

    school.move("hall")
    school.move("The Library")

    assert set(school.world.rooms) == set(rooms)
    assert school.here() == "the_library"


def test_an_authored_room_is_reached_by_name_or_by_id(school):
    """"office", "the Director's office" and "Office" are one place."""
    world = school.world

    assert find_room(world, "office") == "office"
    assert find_room(world, "the Director's office") == "office"
    assert find_room(world, "OFFICE") == "office"
    assert find_room(world, "nowhere at all") is None


@pytest.mark.parametrize(
    "name, expected",
    [
        ("the library", "the_library"),
        ("A Sala de Música", "a_sala_de_musica"),
        ("the East Wing", "the_east_wing"),
        ("図書室", "図書室"),
    ],
)
def test_an_id_can_be_made_from_any_script(name, expected):
    assert slug(name) == expected


# --- prose is checked like any other generated line ----------------------

def test_a_description_naming_a_fact_is_thrown_away(fake_llm):
    """A room description reaches the log as narration, so a keyword in
    one lets the narrator raise the subject just by describing the room.
    Authored worlds are guarded in review; a generated one here."""

    class Leaky(FakeLLM):
        def complete(self, system, prompt, key=None):
            if key and key.startswith("furnish:"):
                return "Shelves, and a noticeboard listing everyone in Corvin House."
            return super().complete(system, prompt, key)

    session = Session.open(ARDENHALL, "arrival", llm=Leaky())
    session.move("the library")

    room = session.world.rooms["the_library"]
    assert room.description == ""      # the room stands, bare
    assert "the_library" in session.world.rooms  # and it still exists


def test_a_clean_description_is_kept(fake_llm):
    class Writer(FakeLLM):
        def complete(self, system, prompt, key=None):
            if key and key.startswith("furnish:"):
                return "Long tables under a skylight. It smells of dust and radiators."
            return super().complete(system, prompt, key)

    session = Session.open(ARDENHALL, "arrival", llm=Writer())
    session.move("the library")

    assert "skylight" in session.world.rooms["the_library"].description


def test_invents_a_fact_names_the_offender():
    world = load_world(ARDENHALL)

    assert invents_a_fact("A noticeboard for Thane House.", world) == "your_placement"
    assert invents_a_fact("Long tables under a skylight.", world) is None


# --- and it is still there tomorrow --------------------------------------

def test_a_corridor_found_once_is_there_in_a_later_scene(fake_llm, tmp_path):
    """Kept per world rather than per scene, which is the whole reason to
    store it rather than generate it again."""
    db = str(tmp_path / "school.sqlite")
    with Session.open(ARDENHALL, "arrival", db_path=db, llm=fake_llm) as first:
        first.move("the library")
        first.move("hall")
        first.move("the east stair")

    later = Session.open(ARDENHALL, "first_night", db_path=db, llm=fake_llm)

    assert {"the_library", "the_east_stair"} <= set(later.world.rooms)
    assert later.world.rooms["the_library"].adjacent == {"hall": "adjacent"}
    assert "the_library" in later.world.rooms["hall"].adjacent


def test_a_different_story_starts_with_the_authored_map(fake_llm, tmp_path):
    with Session.open(ARDENHALL, "arrival", db_path=str(tmp_path / "a.sqlite"), llm=fake_llm) as one:
        one.move("the library")

    other = Session.open(ARDENHALL, "arrival", db_path=str(tmp_path / "b.sqlite"), llm=fake_llm)

    assert sorted(other.world.rooms) == ["hall", "office"]


def test_restoring_skips_anything_hanging_off_a_room_that_is_gone(fake_llm, store):
    """An author who deletes a room should not break a saved story."""
    world = load_world(ARDENHALL)
    store.add_discovered_room("ardenhall", "cellar", "the cellar", "", "a_room_since_deleted")

    restore(world, store)

    assert "cellar" not in world.rooms


def test_connect_is_the_engine_deciding_not_a_model():
    """The whole safety argument in one assertion: the edge is not
    anything a model returned."""
    world = World(id="t", rooms={"hall": Room(id="hall", name="the hall")})

    connect(world, Room(id="library", name="the library"), "hall")

    assert world.rooms["library"].adjacent == {"hall": "adjacent"}
    assert world.rooms["hall"].adjacent == {"library": "adjacent"}
