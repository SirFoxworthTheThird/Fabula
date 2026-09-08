"""Asleep in the room is not the same as being in the room.

Spec §8 makes elapsed time explicitly non-uniform: somebody asleep gets
a discontinuity, somebody awake and waiting lived every hour of it. Half
of that was unreachable — `was_asleep` read `state_change` events, and
nothing in the engine had ever written one, so the "asleep" branch could
not fire and every character felt every minute.

Sleeping through something said in front of you is the sharpest
asymmetry this engine has, and it costs nothing to allow: the filter
only ever removes perception, never adds any.
"""
from pathlib import Path

import pytest

from fabula.chronology import render_time_skip, was_asleep
from fabula.llm import FakeLLM
from fabula.session import Session

WORLDS = Path(__file__).parent.parent / "worlds"
ASHGROVE = WORLDS / "ashgrove"
SECRET = "I broke Grandma's music box, and I let them blame the cat."


@pytest.fixture
def table(fake_llm):
    return Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)


def sleeps(session, character_id, state="asleep", room="kitchen"):
    return session.store.append_event(
        session.director.build_event(
            "state_change", character_id, room, f"{character_id} is {state}.",
            metadata={"character_id": character_id, "state": state},
        )
    )


def perceived(session, character_id):
    events = session.store.get_events(session.scene.id)
    return session.director.contexts.project(session.characters[character_id], events)


def test_a_sleeper_misses_what_is_said_in_front_of_them(table):
    sleeps(table, "maria")
    table.store.append_event(
        table.director.build_event("utterance", "tomas", "kitchen", SECRET)
    )

    lines = [p.perceived_content for p in perceived(table, "maria")]

    assert "music box" not in " ".join(lines).lower()
    # Elena was awake in the same room and did hear it.
    assert "music box" in " ".join(
        p.perceived_content for p in perceived(table, "elena")
    ).lower()


def test_it_stays_out_of_their_context_and_their_memory(table):
    sleeps(table, "maria")
    table.store.append_event(
        table.director.build_event("utterance", "tomas", "kitchen", SECRET)
    )
    table.director._absorb()

    events = table.store.get_events(table.scene.id)
    context = table.director.contexts.for_character(table.characters["maria"], events)

    assert "music box" not in context.lower()
    assert not any("music box" in b.content.lower() for b in table.store.get_beliefs("maria"))


def test_their_own_state_is_theirs_to_know(table):
    """Going under and coming back up are both things they feel."""
    sleeps(table, "maria")
    table.store.append_event(
        table.director.build_event("utterance", "tomas", "kitchen", SECRET)
    )
    sleeps(table, "maria", state="awake")

    kinds = [p.event.kind for p in perceived(table, "maria")]

    assert kinds == ["state_change", "state_change"]


def test_they_are_awake_again_afterwards(table):
    sleeps(table, "maria")
    sleeps(table, "maria", state="awake")
    table.store.append_event(
        table.director.build_event("utterance", "elena", "kitchen", "You missed it.")
    )

    assert "You missed it." in [p.perceived_content for p in perceived(table, "maria")]


def test_the_same_hours_are_felt_and_not_felt(table):
    """The payoff §8 promised: one skip, two experiences of it."""
    sleeps(table, "maria")
    skip = table.store.append_event(
        table.director.build_event(
            "time_skip", None, "kitchen", "2 hours pass.",
            audibility="building", metadata={"minutes": 120},
        )
    )
    events = table.store.get_events(table.scene.id)

    hers = render_time_skip(skip, "maria", events, table.world.phrasing)
    elenas = render_time_skip(skip, "elena", events, table.world.phrasing)

    assert "gap" in hers and "unfelt" in hers
    assert "feel every one" in elenas


def test_a_skip_still_reaches_a_sleeper(table):
    """Dropping it would leave them with no sense of time having passed
    at all, which is the opposite of what sleeping through it means."""
    sleeps(table, "maria")
    table.store.append_event(
        table.director.build_event(
            "time_skip", None, "kitchen", "2 hours pass.",
            audibility="building", metadata={"minutes": 120},
        )
    )

    assert "time_skip" in [p.event.kind for p in perceived(table, "maria")]


def test_a_sleeper_does_not_bid(table):
    """Asleep is not quiet, it is absent."""
    from fabula.bidding import prefilter_candidates

    sleeps(table, "maria")
    event = table.store.append_event(
        table.director.build_event("utterance", "elena", "kitchen", "Maria? Maria.")
    )
    events = table.store.get_events(table.scene.id)

    candidates = prefilter_candidates(event, table.characters, events, table.world)

    assert "maria" not in [c.id for c in candidates]
    assert "tomas" in [c.id for c in candidates]


def test_the_room_can_see_that_they_are_asleep(table):
    """Otherwise a sleeper reads as somebody being rude."""
    sleeps(table, "maria")
    events = table.store.get_events(table.scene.id)

    situation = table.director.contexts.situation(table.characters["elena"], events)

    assert "Maria (asleep)" in situation
    assert "Tomás (asleep)" not in situation


def test_an_authored_intention_is_what_puts_them_under(fake_llm):
    """Turning in is an authored moment, not something the engine decides
    for somebody. End to end through the ordinary time machinery."""
    session = Session.open(WORLDS / "winterlight", "the_long_dark", llm=fake_llm)
    for _ in range(14):
        if not session.wait(consent=lambda _minutes: True):
            break

    events = session.store.get_events(session.scene.id)
    states = [
        (e.metadata.get("character_id"), e.metadata.get("state"))
        for e in events
        if e.kind == "state_change"
    ]

    assert ("nadia", "asleep") in states
    assert ("nadia", "awake") in states  # and an authored moment brings her back


def test_a_pressure_can_do_it_too(table):
    """`state_change` is an ordinary effect kind, so an authored pressure
    can put somebody under as readily as an intention can."""
    from fabula.models import Pressure

    pressure = Pressure(
        id="she_nods_off",
        intent="Maria stops fighting it and her head goes back against the chair.",
        effect={"kind": "state_change", "actor": "maria", "state": "asleep", "location": "kitchen"},
    )

    event = table.director._fire(pressure)

    assert event.kind == "state_change"
    assert event.metadata["character_id"] == "maria"
    assert event.metadata["state"] == "asleep"


def test_waiting_cannot_spin_forever_on_something_that_will_not_resolve(fake_llm):
    """Regression, and older than sleep. A private intention in a room the
    player is standing in can never resolve — but the skip derivation kept
    proposing it, so `/wait` in `winterlight` burned fifty minutes of story
    time a turn, forever, resolving nothing. The derivation now offers only
    what resolution would accept."""
    session = Session.open(WORLDS / "winterlight", "the_long_dark", llm=fake_llm)
    derived = []
    for _ in range(14):
        derived.append(session.pending_skip())
        if not session.wait(consent=lambda _minutes: True):
            break

    assert None in derived, "the scene has to run out of reachable moments"
    assert derived.count(50) < 5, f"stuck re-deriving the same skip: {derived}"
