"""Taking the story back.

Every other app on this shelf has some version of it — swipe for another
answer, edit the line, delete back to a point — and it is the first
control people reach for, because it is how you survive a model having a
bad turn. This one had `/again`, which throws away the last take and
nothing else.

Three transcripts against a local 3B in one afternoon produced: a
character announcing the same intention eight times, another handing over
the secret he is keeping on the first line, and a third repeating
somebody else's line word for word. Deterministic guards caught some of
that and will never catch all of it. The player needs a way out.

What makes it more than a delete is that a scene is not only its log.
Trust moved, interactions counted, goals closed — none of which is a row
that can be deleted, all of which has to be put back.
"""
import tempfile
from pathlib import Path

from fabula.llm import FakeLLM
from fabula.persistence import unresolved_goals
from fabula.session import Session

from tests.conftest import ASHGROVE


def a_story(scene="the_dinner", **kwargs):
    return Session.open(
        ASHGROVE, scene, llm=FakeLLM(),
        db_path=str(Path(tempfile.mkdtemp()) / "story.sqlite"), **kwargs,
    )


def my_lines(session):
    return [
        event.seq
        for event in session.store.get_events(session.scene.id)
        if event.actor_id == session.user_character.id and event.kind == "utterance"
    ]


# --- what it takes back --------------------------------------------------

def test_the_scene_goes_back_to_where_it_was(fake_llm):
    session = a_story()
    for line in ("What is it?", "You have been odd all evening.", "Talk to me."):
        session.say(line)
    mark = my_lines(session)[0]
    was = [e.seq for e in session.store.get_events(session.scene.id) if e.seq <= mark]

    session.rewind_to(mark)

    assert [e.seq for e in session.store.get_events(session.scene.id)] == was
    assert session.turns_played == 1
    session.close()


def test_and_the_next_take_stands_where_the_old_one_did(fake_llm):
    """Not pasted after the withdrawn events: the scene occupies the slot
    it did before, so nothing downstream sees a gap in the sequence."""
    session = a_story()
    session.say("What is it?")
    session.say("Please.")
    session.rewind_to(my_lines(session)[0])

    session.say("Something else, then.")

    seqs = [e.seq for e in session.store.get_events(session.scene.id)]
    assert seqs == list(range(1, len(seqs) + 1))
    session.close()


def test_the_log_is_still_append_only(fake_llm):
    """The record of what was played is not rewritten. It stops being
    returned, which is what "never happened" has to mean when half the
    engine reads the log to decide what is true."""
    session = a_story()
    session.say("What is it?")
    session.say("Please.")
    before = session.store.conn.execute(
        "SELECT COUNT(*) FROM events WHERE scene_id = ?", (session.scene.id,)
    ).fetchone()[0]

    session.rewind_to(my_lines(session)[0])

    after = session.store.conn.execute(
        "SELECT COUNT(*) FROM events WHERE scene_id = ?", (session.scene.id,)
    ).fetchone()[0]
    assert after == before, "nothing was deleted from the log"
    assert len(session.store.get_events(session.scene.id)) < before, "and less is returned"
    session.close()


def test_what_was_made_of_it_goes_for_good(fake_llm):
    """A belief is not the record of a moment, it is somebody's
    impression of one — and the impression of a moment that has been
    taken back is nothing."""
    session = a_story()
    session.say("Tomás, what happened in March?")
    session.say("Answer me.")
    assert session.store.get_beliefs("tomas"), "he took something in"
    mark = my_lines(session)[0]
    kept = {
        belief.content
        for belief in session.store.get_beliefs("tomas")
        if belief.source_event_id and belief.source_event_id <= mark
    }

    session.rewind_to(mark)

    assert {b.content for b in session.store.get_beliefs("tomas")} == kept
    session.close()


# --- and what it puts back ----------------------------------------------

def test_a_goal_that_closed_is_open_again(fake_llm):
    """The part a delete cannot do. `resolved_goals` is a flag somebody
    set, not a row belonging to an event, so it has to be reset and
    replayed from what is left of the log."""
    session = a_story(scene="the_reckoning")
    tomas = session.characters["tomas"]
    assert unresolved_goals(session.store, tomas), "he starts out still keeping it"

    session.say("You broke Grandma's music box, didn't you.")
    assert not unresolved_goals(session.store, tomas), "and hears it raised"

    session.rewind_to(my_lines(session)[0] - 1)

    assert unresolved_goals(session.store, tomas), "and is keeping it again"
    session.close()


def test_trust_that_moved_moves_back(fake_llm):
    """Watching somebody refuse to answer drifts trust toward a floor.
    It is an increment, so nothing about deleting an event undoes it."""
    session = a_story(scene="the_reckoning")
    started = session.store.get_relationships("maria")["tomas"].trust
    build = session.director.build_event
    session.store.append_event(
        build("action", "tomas", "kitchen", "Tomás says nothing.", metadata={"withheld": True})
    )
    session.director._absorb()
    moved = session.store.get_relationships("maria")["tomas"].trust
    assert moved < started, "she saw him decline"

    session.rewind_to(0)

    assert session.store.get_relationships("maria")["tomas"].trust == started
    session.close()


def test_the_replay_costs_no_model_call(fake_llm):
    """The expensive half of taking an event in is the private reading,
    and the readings that survive are already written down."""
    session = a_story()
    session.say("What is it?")
    session.say("Please.")
    spent = len(session.llm.calls)

    session.rewind_to(my_lines(session)[0])

    assert len(session.llm.calls) == spent, "a rewind asks nothing of anybody"
    session.close()


# --- the edges -----------------------------------------------------------

def test_rewinding_to_the_beginning_leaves_the_scene_open(fake_llm):
    session = a_story()
    session.say("What is it?")

    left = session.rewind_to(0)

    assert session.store.get_events(session.scene.id) == []
    assert left == []
    assert session.turns_played == 0
    session.close()


def test_rewinding_past_the_end_takes_nothing(fake_llm):
    session = a_story()
    session.say("What is it?")
    stood_at = len(session.store.get_events(session.scene.id))

    session.rewind_to(9999)

    assert len(session.store.get_events(session.scene.id)) == stood_at
    session.close()


def test_there_is_nothing_left_to_say_again_to(fake_llm):
    """The take that was discardable belonged to a turn that no longer
    exists."""
    session = a_story()
    session.say("What is it?")
    session.say("Please.")
    assert session.can_regenerate()

    session.rewind_to(my_lines(session)[0])

    assert not session.can_regenerate()
    session.close()


def test_the_scene_can_be_played_on_from_where_it_was_left(fake_llm):
    """The whole point: not an undo button, a way back into the story."""
    session = a_story()
    for line in ("What is it?", "Please.", "Tomás."):
        session.say(line)

    session.rewind_to(my_lines(session)[0])
    perceived = session.say("Let me try that again.")

    assert perceived, "the room answered"
    assert session.turns_played == 2
    session.close()
