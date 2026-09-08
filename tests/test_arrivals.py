"""Somebody who was not in the room when the scene opened.

The spec's own example pressure has always been an arrival with an
`actor`, and until now there was nowhere for that actor to come from:
the cast was the world, and then the cast was the scene. This is the
missing half — a character written for the world, declared as somebody
who *may* walk in, brought on when an authored pressure fires.

The narrator never does this. It writes prose, and a person who arrives
has to arrive as an event with an actor id, which only the director
assigns. What it invents stays a ghost in a sentence.

The part worth reading is what makes their memory correct: nothing
backfills it. They wait off-stage, which is not a room — unreachable in
both directions, so every moment before their arrival resolves to "none"
through the ordinary perception path. There is no decision to get wrong
about what they might have overheard.
"""
from pathlib import Path

import pytest

from fabula.llm import FakeLLM
from fabula.loader import load_scenario
from fabula.session import Session
from fabula.world import OFFSTAGE

WORLDS = Path(__file__).parent.parent / "worlds"
WINTERLIGHT = WORLDS / "winterlight"
FLIGHT = "The first flight has slipped to December."


@pytest.fixture
def station(fake_llm):
    return Session.open(WINTERLIGHT, "the_manifest", llm=fake_llm)


def test_someone_declared_but_not_cast_starts_outside_the_scene(station):
    assert "petra" not in station.characters
    assert "petra" in station.director.waiting
    assert station.director.waiting["petra"].location_id == OFFSTAGE


def test_off_stage_is_not_a_room(station):
    """Unreachable in both directions, so it needs no special case in the
    code the leak tests depend on."""
    world = station.world

    assert OFFSTAGE not in world.rooms
    assert world.distance(OFFSTAGE, "mess") == -1
    assert world.distance("mess", OFFSTAGE) == -1


def test_they_perceive_nothing_while_they_are_waiting(station):
    station.store.append_event(
        station.director.build_event("utterance", "ilse", "mess", FLIGHT, audibility="building")
    )

    waiting = station.director.waiting["petra"]
    events = station.store.get_events(station.scene.id)

    assert station.director.contexts.project(waiting, events) == []


def test_nobody_perceives_them_either(station):
    """Not visible, not audible, not counted as present."""
    assert "Petra Lindqvist" not in [c.name for c in station.present()]


def test_an_authored_pressure_brings_them_on(fake_llm):
    """End to end through the ordinary turn loop: the trigger comes due,
    the director fires the pressure, and she is in the room."""
    session = Session.open(WINTERLIGHT, "the_manifest", llm=fake_llm)
    for line in ("Yusuf, does the fuel log add up?", "Nobody has counted the drums.",
                 "Ilse, eleven days is a long time.", "Someone say something."):
        session.say(line)
        if "petra" in session.characters:
            break

    assert "petra" in session.characters
    assert session.director.waiting == {}
    fired = [
        e for e in session.store.get_events(session.scene.id)
        if e.metadata.get("pressure_id") == "petra_comes_in_off_the_line"
    ]
    assert fired and fired[0].actor_id == "petra"


def test_they_walk_in_knowing_only_what_they_walk_in_on(station):
    """The whole point. The secret was said in the room she is walking
    into, before she was anywhere — and it is not hers."""
    station.store.append_event(
        station.director.build_event("utterance", "ilse", "mess", FLIGHT)
    )
    station.director.admit("petra")
    station.store.append_event(
        station.director.build_event("arrival", "petra", "mess", "The porch door bangs.")
    )
    station.store.append_event(
        station.director.build_event("utterance", "nadia", "mess", "You picked a night for it.")
    )

    events = station.store.get_events(station.scene.id)
    petra = station.characters["petra"]
    perceived = station.director.contexts.project(petra, events)

    assert perceived[0].event.kind == "arrival"  # her own, and nothing before it
    assert "december" not in " ".join(p.perceived_content for p in perceived).lower()
    assert "december" not in station.director.contexts.for_character(petra, events).lower()


def test_from_the_arrival_on_they_are_an_agent_like_any_other(station):
    station.director.admit("petra")
    station.store.append_event(
        station.director.build_event("arrival", "petra", "mess", "The porch door bangs.")
    )

    assert station.director.current_location(station.characters["petra"]) == "mess"
    assert "Petra Lindqvist" in [c.name for c in station.present()]
    # Durable state exists for them from the moment they are in the scene.
    assert set(station.store.get_relationships("petra")) == {"ana", "ilse", "nadia", "yusuf"}


def test_admitting_them_reaches_every_holder_of_the_cast(station):
    """The session, the context builder and the director share one dict.
    Somebody who joined only the director's copy would bid without ever
    being seen to be in the room."""
    station.director.admit("petra")

    assert "petra" in station.characters
    assert "petra" in station.director.characters
    assert "petra" in station.director.contexts.characters


def test_a_scene_cannot_await_somebody_it_has_already_cast(tmp_path):
    import shutil

    import yaml

    world_dir = tmp_path / "winterlight"
    shutil.copytree(WINTERLIGHT, world_dir)
    path = world_dir / "scenes" / "the_manifest.yaml"
    scene = yaml.safe_load(path.read_text(encoding="utf-8"))
    scene["may_arrive"] = ["ilse"]
    path.write_text(yaml.safe_dump(scene), encoding="utf-8")

    with pytest.raises(ValueError, match="both casts and awaits"):
        Session.open(world_dir, "the_manifest", llm=FakeLLM())


def test_awaiting_a_stranger_is_refused(tmp_path):
    """Silently dropping an unknown id would make a typo look like
    somebody who simply never turns up."""
    import shutil

    import yaml

    world_dir = tmp_path / "winterlight"
    shutil.copytree(WINTERLIGHT, world_dir)
    path = world_dir / "scenes" / "the_manifest.yaml"
    scene = yaml.safe_load(path.read_text(encoding="utf-8"))
    scene["may_arrive"] = ["petraa"]
    path.write_text(yaml.safe_dump(scene), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown character"):
        Session.open(world_dir, "the_manifest", llm=FakeLLM())


def test_a_scene_that_awaits_nobody_is_unaffected(fake_llm):
    session = Session.open(WORLDS / "ashgrove", "the_dinner", llm=fake_llm)

    assert session.director.waiting == {}
    assert load_scenario(WORLDS / "ashgrove", "the_dinner")[2].may_arrive == []
