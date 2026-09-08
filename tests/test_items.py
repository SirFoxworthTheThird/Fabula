"""Things that carry a fact.

The only kind of object worth modelling in this engine. A mug is scenery
and belongs in a room description; a key that opens a door is an
adventure game and a different product. A letter, a logbook, a sealed
file is a **second channel for the asymmetry the engine already turns
on** — whoever reads it knows, whoever does not, does not.

Reading is two things, so it is two events: the room sees you open it,
and what it says is private and addressed to you. Everything below falls
out of the perception rules that were already there.
"""
from pathlib import Path

import pytest

from fabula.commands import run_command
from fabula.llm import FakeLLM
from fabula.loader import load_world
from fabula.pressures import evaluate_trigger, scene_state
from fabula.session import Session

WORLDS = Path(__file__).parent.parent / "worlds"
ARDENHALL = WORLDS / "ardenhall"
ASHGROVE = WORLDS / "ashgrove"


@pytest.fixture
def office(fake_llm):
    """the_interview: Rook and Vance both in the office, with the file."""
    return Session.open(ARDENHALL, "the_interview", llm=fake_llm)


def perceived(session, character_id):
    events = session.store.get_events(session.scene.id)
    return [
        p.perceived_content
        for p in session.director.contexts.project(session.characters[character_id], events)
    ]


# --- the asymmetry, which is the whole point ----------------------------

def test_the_room_sees_you_read_and_does_not_see_what_you_read(office):
    assert "Director Vance" in [c.name for c in office.present()]

    office.read("grey folder")

    assert any("opens" in line for line in perceived(office, "vance"))
    assert not any("Marlow" in line for line in perceived(office, "vance"))
    assert any("Marlow" in line for line in perceived(office, "rook"))


def test_it_stays_out_of_the_watchers_context_and_memory(office):
    office.read("grey folder")
    events = office.store.get_events(office.scene.id)

    context = office.director.contexts.for_character(office.characters["vance"], events)

    assert "marlow" not in context.lower()
    assert not any("Marlow" in b.content for b in office.store.get_beliefs("vance"))


def test_the_reader_remembers_it(office):
    office.read("grey folder")

    assert any("Marlow" in b.content for b in office.store.get_beliefs("rook"))


def test_reading_is_not_saying(office):
    """Falls out of the existing rule rather than needing an exception:
    the private event's only perceiver is its own actor, and a fact is
    spoken when somebody *other than the actor* hears it in full."""
    office.read("grey folder")
    events = office.store.get_events(office.scene.id)
    state = scene_state(events, office.characters, office.world)

    assert evaluate_trigger({"fact_spoken": "the_seventh_file"}, state, office.world.facts) is False


def test_nobody_in_another_room_perceives_any_of_it(fake_llm):
    session = Session.open(ARDENHALL, "arrival", llm=fake_llm)  # Vance in the office
    session.read("the intake list")

    assert perceived(session, "vance") == []


# --- finding the thing ---------------------------------------------------

def test_looking_around_says_what_there_is_to_read(fake_llm):
    session = Session.open(ARDENHALL, "arrival", llm=fake_llm)

    outcome = run_command(session, "/look")

    assert "the intake list pinned by the stairs" in outcome.message


def test_an_item_is_found_by_any_reasonable_name(office):
    world = office.world

    for wanted in ("grey folder", "the grey folder on the desk", "GREY FOLDER", "the_sealed_file"):
        assert world.find_item("office", wanted) is not None
    assert world.find_item("office", "a sandwich") is None


def test_you_cannot_read_something_that_is_in_another_room(fake_llm):
    session = Session.open(ARDENHALL, "arrival", llm=fake_llm)  # Rook is in the hall

    outcome = run_command(session, "/read grey folder")

    assert outcome.perceived == []
    assert "no grey folder here" in outcome.message


def test_a_world_with_no_items_is_unaffected(fake_llm):
    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)

    assert session.world.items == {}
    assert session.items_here() == []
    assert "no music box here" in run_command(session, "/read music box").message


# --- how it sits with everything else -------------------------------------

def test_reading_is_a_turn_and_can_be_taken_again(office):
    office.read("grey folder")
    before = len(office.store.get_events(office.scene.id))

    office.regenerate()

    assert len(office.store.get_events(office.scene.id)) == before


def test_every_authored_item_points_at_things_that_exist():
    """A guard for every world: an item in a room that is not there can
    never be found, and one revealing a fact that is not there is an
    author's typo nobody would notice."""
    for world_dir in sorted(p.parent for p in WORLDS.glob("*/world.yaml")):
        world = load_world(world_dir)
        for item in world.items.values():
            assert item.location_id in world.rooms, f"{world.id}/{item.id}"
            assert item.reveals is None or item.reveals in world.facts, f"{world.id}/{item.id}"


def test_an_items_name_never_names_a_fact():
    """The name goes into the public "opens" beat, which the whole room
    perceives — so a fact keyword in a *name* would count as saying it
    out loud just by picking the thing up. The text is where the fact
    belongs, and the text is private."""
    from fabula.world import mentions_fact

    for world_dir in sorted(p.parent for p in WORLDS.glob("*/world.yaml")):
        world = load_world(world_dir)
        for item in world.items.values():
            named = [f for f in world.facts if mentions_fact(world.facts[f], item.name)]
            assert not named, f"{world.id}/{item.id} is named after {named}"
