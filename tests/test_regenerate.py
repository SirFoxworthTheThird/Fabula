"""Taking the moment again.

The engine never stops to ask whether a change to the world is wanted —
a consent prompt in the middle of a scene is a dialogue box, not a
story. It acts, and this is how a take gets rejected: the player is a
director calling "again".

What makes it cheap is that everything durable lives in one SQLite
connection. A savepoint around a turn undoes the events, the beliefs
encoded from them, the readings, the rehearsals, the interaction counts
and any trust that moved — with no per-subsystem bookkeeping, and no
tombstones left in an append-only log.
"""
import itertools

import pytest

from fabula.commands import run_command
from fabula.session import Session

from tests.conftest import ASHGROVE

SECRET = "music box"


class VaryingLLM:
    """Different every call, so a retake is visibly a retake."""

    def __init__(self):
        self.n = itertools.count(1)

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        return f"take {next(self.n)}"


def state_of(session):
    store = session.store
    return (
        [e.content for e in store.get_events(session.scene.id)],
        len(store.get_beliefs("maria")),
        round(store.get_relationships("maria")["tomas"].trust, 4),
        len(store.get_rehearsals("maria")),
        store.next_seq(session.scene.id),
    )


def test_a_retake_replaces_the_take_rather_than_following_it(fake_llm):
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    session.say("Tomás, you have been quiet all evening.")
    after_one = state_of(session)

    session.regenerate()

    # Not appended: the same number of events, from the same sequence.
    assert state_of(session) == after_one


def test_the_retake_is_genuinely_a_new_take():
    session = Session.open(ASHGROVE, "the_reckoning", llm=VaryingLLM())
    session.say("Tomás, you have been quiet all evening.")
    first = [e.content for e in session.store.get_events(session.scene.id)]

    session.regenerate()

    again = [e.content for e in session.store.get_events(session.scene.id)]
    assert len(again) == len(first)
    assert again != first
    assert again[0] == first[0]  # the player's own line is not re-rolled


def test_everything_the_take_wrote_goes_with_it(fake_llm):
    """Not just events. A withheld beat moves trust, encodes beliefs and
    records rehearsals; all of it has to come back."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    session.say("Tomás, you have been quiet all evening.")
    before = state_of(session)

    session.say("What happened to Grandma's music box?")
    during = state_of(session)
    assert during[1] > before[1]      # beliefs were encoded
    assert during[2] < before[2]      # trust moved on the withheld beat

    session.regenerate()

    after = state_of(session)
    assert after[1] == during[1]      # the retake wrote its own, once
    assert after[2] == during[2]
    assert after[4] == during[4]      # and reused the sequence numbers


def test_only_the_last_take_can_be_taken_again(fake_llm):
    """Opening a turn settles the one before it, so what is still
    discardable is always the most recent — which is the one a player
    would want back."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=VaryingLLM())
    session.say("First.")
    first_line = session.store.get_events(session.scene.id)[0].content
    session.say("Second.")

    session.regenerate()

    contents = [e.content for e in session.store.get_events(session.scene.id)]
    assert contents[0] == first_line          # the settled turn is untouched
    assert "Second." in contents


def test_nothing_to_take_again_before_anything_is_played(fake_llm):
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)

    assert session.can_regenerate() is False
    with pytest.raises(ValueError, match="nothing has been played"):
        session.regenerate()


def test_a_move_or_a_look_can_be_taken_again_too(fake_llm):
    """Every player action is a turn, not just speaking."""
    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)
    session.move("study")
    after_move = state_of(session)

    session.regenerate()

    assert state_of(session) == after_move
    assert session.here() == "study"  # still where the move put her


def test_the_command_layer_flags_a_replacement(fake_llm):
    """A client rendering a transcript has to drop the tail before
    showing the new take, so it must be told."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    run_command(session, "Tomás, you have been quiet all evening.")

    outcome = run_command(session, "/again")

    assert outcome.replaced is True
    assert outcome.perceived
    assert run_command(session, "/look").replaced is False


def test_asking_for_a_retake_before_playing_says_so(fake_llm):
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)

    outcome = run_command(session, "/again")

    assert outcome.replaced is False
    assert "Nothing has been played" in outcome.message


def test_an_ending_is_reported_on_the_retake_that_causes_it(fake_llm):
    """`ended` compares before and after. A retake rewinds, so the
    comparison has to be against the state the discarded take started
    from — otherwise re-rolling the moment the scene ends never reports
    it, or reports it twice."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    run_command(session, "Pass the bread.")
    first = run_command(session, "You broke Grandma's music box, didn't you.")
    assert first.ended is True

    again = run_command(session, "/again")

    assert session.ended() is True
    assert again.ended is True


def test_a_retake_leaves_no_tombstone_in_the_log(fake_llm):
    """The log is append-only, and a discarded take is not history — it
    was never committed. Nothing is marked deleted, because nothing was
    ever there."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=VaryingLLM())
    session.say("Tomás?")
    session.regenerate()
    session.regenerate()

    events = session.store.get_events(session.scene.id)
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert not any("take 1" == e.content for e in events)  # the first roll is gone
