"""Somebody doing something in another room while you are not there.

This is the one thing a cast of separate agents can do that one model
puppeting all of them cannot, and the spec built every piece of it:
`intentions` say what a character means to do and where, `advance_time`
resolves what came of them off screen, `materialize` turns a coarse stub
into what you walk into. Then none of it ever ran.

Two reasons, and the second was the real one. `advance_time` was
reachable from exactly one place — the player pressing "let thirty
minutes pass", a button whose consequence they had no reason to expect —
so story time never moved. And nothing anywhere sent a character out of
the room, so the cast converged on the player in the first turn and
stood there for the rest of the scene, which made every intention
unreachable: an intention happens somewhere, and they were all here.

Measured before: across fourteen player lines in `ashgrove/the_dinner`
and `winterlight/the_manifest`, zero time skips and zero off-screen
actions. Eleven authored intentions across five worlds, waiting for
somebody to guess.
"""
import tempfile
from pathlib import Path

import pytest

from fabula.chronology import LARGE_SKIP_MINUTES, UNASKED, pending_intentions
from fabula.director import Director
from fabula.llm import FakeLLM
from fabula.pressures import scene_state
from fabula.session import Session
from fabula.shelf import SHIPPED


def a_story(world="ashgrove", scene="the_dinner", **kwargs):
    return Session.open(
        SHIPPED / world, scene, llm=FakeLLM(),
        db_path=str(Path(tempfile.mkdtemp()) / "story.sqlite"), **kwargs
    )


def played(session, lines):
    for i in range(lines):
        session.say(f"Line {i}.")
    return session.store.get_events(session.scene.id)


def offscreen(events):
    return [e for e in events if e.detail_level == "summary"]


# --- that it happens at all ----------------------------------------------

@pytest.mark.parametrize(
    "world, scene", [("ashgrove", "the_dinner"), ("winterlight", "the_manifest")]
)
def test_the_house_carries_on_around_the_scene(world, scene):
    """The measurement this exists for, in both a two-hander and a room
    with four people in it."""
    session = a_story(world, scene)

    events = played(session, 14)

    assert offscreen(events), "nobody ever did anything anywhere else"
    assert [e for e in events if e.kind == "time_skip"], "story time never moved"
    session.close()


def test_somebody_gets_up_and_goes_where_the_thing_is():
    """An intention names a room, and the author wrote that room because
    it is not this one. Without a way to walk there it may as well not
    have been written."""
    session = a_story()

    events = played(session, 14)

    # An arrival the *house* produced, not one an authored pressure did:
    # `maria_comes_through` already walked her into the kitchen, which is
    # the story coming to the player rather than carrying on without them.
    went = [
        e for e in events
        if e.kind == "arrival" and e.metadata.get(UNASKED) and e.actor_id
    ]
    assert went, "the cast stood exactly where the scene put them"
    session.close()


def test_it_is_the_authored_intention_that_resolves():
    """Not something made up. What lands in the log is the line the
    author wrote in `intentions`, in the room they wrote it for."""
    session = a_story()

    events = played(session, 14)

    done = offscreen(events)
    assert done
    for event in done:
        assert event.metadata.get("intention_id")
        wrote = [
            i
            for c in session.characters.values()
            for i in c.intentions
            if i.id == event.metadata["intention_id"]
        ]
        assert wrote, event
        assert event.location_id == wrote[0].location_id
    session.close()


# --- and that the player does not see it ---------------------------------

def test_none_of_it_reaches_the_player_where_they_are_standing():
    """The whole value of the thing is that they find out afterwards. An
    off-screen action they watched happen is just an action."""
    session = a_story()

    events = played(session, 14)

    perceived = {p.event.seq for p in session.perceived_so_far()}
    for event in offscreen(events):
        assert event.seq not in perceived, event.content
    session.close()


def test_an_intention_is_never_resolved_in_the_room_the_player_is_in():
    """Found by playing it: resolution only compared where the character
    was against where the player was, never where the *intention* was —
    so Maria sorted the letters in a room she was not in, and with the
    player sitting in that room, in front of them."""
    session = a_story()
    director = session.director

    played(session, 6)
    session.move("study")
    played(session, 8)

    where = director.current_location(session.user_character)
    for event in offscreen(session.store.get_events(session.scene.id)):
        assert event.location_id != where or event.seq < session.turn_started_at
    session.close()


def test_a_character_cannot_do_a_thing_in_a_room_they_are_not_in():
    """The same rule from the other side, asked of the guard directly."""
    # the_reckoning puts Maria in the kitchen with the letters she means
    # to finish still in the study.
    session = a_story(scene="the_reckoning")
    director = session.director
    maria = session.characters["maria"]
    elsewhere = [
        i for i in pending_intentions(maria, session.store.get_events(session.scene.id))
        if i.location_id != director.current_location(maria)
    ]
    assert elsewhere, "the fixture needs an intention somewhere else"

    assert all(director.unreachable(maria, i) for i in elsewhere)
    session.close()


# --- what it costs the scene ---------------------------------------------

def test_the_house_never_takes_more_time_than_the_player_would_be_asked_about():
    """Spec §8: time is not something the player loses without noticing.
    A skip big enough to need consent stays a thing they ask for."""
    session = a_story("winterlight", "the_manifest")

    events = played(session, 14)

    for event in events:
        if event.kind == "time_skip" and event.metadata.get(UNASKED):
            assert event.metadata["minutes"] <= LARGE_SKIP_MINUTES
    session.close()


def test_the_player_is_never_left_alone_by_it():
    """A house that empties around somebody is not a world carrying on
    without them, it is a scene being taken away."""
    session = a_story("winterlight", "the_manifest")
    director = session.director

    for i in range(14):
        session.say(f"Line {i}.")
        here = director.current_location(session.user_character)
        assert director._company(here) >= 1, f"alone after line {i}"
    session.close()


def test_it_does_not_spend_the_author_s_pressures_faster():
    """`turns_elapsed` is counted in log positions, so anything that
    appends events quietly ramps every pressure in every world. The
    house moving is not the scene taking a turn."""
    session = a_story()

    events = played(session, 14)
    house = [e for e in events if e.metadata.get(UNASKED)]
    assert house, "nothing to discount"

    state = scene_state(events, session.characters, session.world)
    assert state.turn_count == events[-1].seq - len(house)
    session.close()


def test_the_room_still_gets_the_beat_it_was_going_to_get():
    """Measured twice, both wrong: inside the turn loop the house either
    costs the room the beat it was about to take or buys the director an
    extra round to fire a pressure in. It belongs after the turn."""
    quiet = a_story()
    still = a_story()
    still.director._meanwhile = lambda events, top_bid: []

    played(quiet, 8)
    played(still, 8)

    def beats(session):
        return len([
            e for e in session.store.get_events(session.scene.id)
            if e.kind == "narration" and not e.metadata.get("pressure_id")
        ])

    assert beats(quiet) >= beats(still)
    quiet.close()
    still.close()


# --- and then you walk in ------------------------------------------------

def test_walking_in_is_looking():
    """The payoff was behind a button. A coarse stub stays coarse until
    somebody is standing in the room, and the only thing that expanded
    one was the player thinking to press "look around" — in a room they
    had no reason to suspect anything about."""
    session = a_story()
    events = played(session, 14)
    done = offscreen(events)
    assert done, "nothing happened anywhere else to walk in on"
    where = done[0].location_id
    assert where != session.here(), "it happened in the room they are already in"

    seen = session.move(where)

    assert [p for p in seen if p.event.metadata.get("materializes")], (
        "walked into the room it happened in and saw nothing of it"
    )
    session.close()


def test_and_a_room_is_never_read_twice():
    """`materialize` marks what it expanded, so arriving and then looking
    again does not describe the same half hour a second time."""
    session = a_story()
    events = played(session, 14)
    where = offscreen(events)[0].location_id
    session.move(where)

    assert session.director.unmaterialized_here(session.user_character) == []
    assert session.look() == []
    session.close()
