"""Somebody getting up and going, where they are going *from*.

`departure` has been one of this engine's event kinds since the
beginning. It has a beat, a narrator instruction, and a degraded
perception template — "footsteps fading from {location}" — and nothing
anywhere ever built one. Four places handled a kind that could not exist.

Which made moving half an event. The room you walked *into* saw you
arrive; the room you walked out of perceived nothing at all, so somebody
sitting at the table with you did not see you stand up and leave. That
is not a perception rule, it is a hole in one — found by counting which
of the engine's mechanisms an ordinary player ever reaches.
"""
import tempfile
from pathlib import Path

from fabula.chronology import UNASKED
from fabula.llm import FakeLLM
from fabula.memory import location_at_seq
from fabula.session import Session
from fabula.shelf import SHIPPED


def a_story(world="ashgrove", scene="the_reckoning", **kwargs):
    return Session.open(
        SHIPPED / world, scene, llm=FakeLLM(),
        db_path=str(Path(tempfile.mkdtemp()) / "story.sqlite"), **kwargs
    )


def perceived_by(session, character_id):
    return session.director.contexts.project(
        session.characters[character_id], session.store.get_events(session.scene.id)
    )


# --- the hole -------------------------------------------------------------

def test_the_room_you_leave_sees_you_leave():
    session = a_story()
    session.say("I need a minute.")
    left_behind = [
        c.id for c in session.characters.values()
        if not c.is_user and session.director.current_location(c) == session.here()
    ]
    assert left_behind, "the fixture needs somebody to leave behind"

    session.move("study")

    seen = [p for p in perceived_by(session, left_behind[0]) if p.event.kind == "departure"]
    assert seen, "they did not see the player stand up and go"
    assert seen[-1].perception == "full", "they were in the room"
    assert "study" in seen[-1].perceived_content
    session.close()


def test_and_the_room_you_are_going_to_hears_the_door():
    """Both are true and both are perceived. You see her get up; the next
    room hears her come in."""
    session = a_story()
    session.say("Mm.")
    session.move("study")

    kinds = {
        p.event.kind: p.perception
        for p in perceived_by(session, "tomas")
        if p.event.kind in ("departure", "arrival")
    }

    assert kinds.get("departure") == "full"
    assert kinds.get("arrival") == "degraded"
    session.close()


def test_somebody_two_rooms_off_hears_footsteps_and_not_a_name():
    """The degraded template for this kind has existed since M1 and
    nothing could ever reach it."""
    session = a_story()
    session.say("Mm.")
    session.move("study")

    heard = [
        p for p in perceived_by(session, "maria")
        if p.event.kind == "departure" and p.perception == "degraded"
    ]
    for p in heard:
        assert "footsteps" in p.perceived_content
        assert "Elena" not in p.perceived_content
    session.close()


def test_it_moves_nobody():
    """Purely perceptual. Position is replayed from arrivals, and a
    departure that shifted it would put somebody in the room they had
    just left."""
    session = a_story()
    session.move("study")
    events = session.store.get_events(session.scene.id)
    assert [e for e in events if e.kind == "departure"]

    where = location_at_seq(
        session.user_character.id, session.user_character.location_id,
        events, events[-1].seq + 1,
    )

    assert where == "study"
    session.close()


# --- and when the house does it ------------------------------------------

def test_a_character_who_steps_out_is_seen_to_go():
    session = a_story("ashgrove", "the_dinner")
    for i in range(14):
        session.say(f"Line {i}.")

    events = session.store.get_events(session.scene.id)
    theirs = [
        e for e in events
        if e.kind == "departure" and e.actor_id != session.user_character.id
    ]
    assert theirs, "nobody was ever seen to leave"
    for event in theirs:
        # Marked like everything else the house does, or it would quietly
        # count as the scene taking a turn and ramp every pressure.
        assert event.metadata.get(UNASKED)
    session.close()


def test_the_player_sees_it_from_the_room_they_are_in():
    session = a_story("ashgrove", "the_dinner")
    for i in range(14):
        session.say(f"Line {i}.")

    seen = [
        p for p in session.perceived_so_far()
        if p.event.kind == "departure" and p.event.actor_id != session.user_character.id
    ]
    assert seen, "somebody left and the player perceived none of it"
    session.close()
