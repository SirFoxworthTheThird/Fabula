"""What a character is still trying to do.

Goals were authored in every world, stored, aged across scenes — and
read by nothing. `unresolved_goals` was called from no code in the
engine, so a character's stated want had no effect on anything they said
or did, and `resolve_goal` was never called at all, so nothing could
ever stop being open.

This is the third time the same shape has turned up here: durable state
written and never read (beliefs, then trust, now goals). A want that
changes nothing is not a want.
"""
from pathlib import Path

import pytest

from fabula.bidding import GOAL_STAKE, goal_at_stake, heuristic_bid
from fabula.llm import FakeLLM
from fabula.persistence import close_reached_goals, unresolved_goals, wants
from fabula.session import Session

from fabula.shelf import SHIPPED as WORLDS  # noqa: E402
ASHGROVE = WORLDS / "ashgrove"
SECRET = "You broke Grandma's music box, didn't you."


@pytest.fixture
def table(fake_llm):
    """the_reckoning: all three at the kitchen table."""
    return Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)


def open_goals(session, character_id):
    return [g.id for g in unresolved_goals(session.store, session.characters[character_id])]


# --- their own wants, in their own prompt -------------------------------

def test_a_character_is_told_what_they_want(table):
    text = wants(table.store, table.characters["tomas"])

    assert "Keep the broken music box a secret" in text


def test_it_reaches_the_prompt_they_are_actually_given(fake_llm):
    from fabula.agents import generate_utterance

    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    events = session.store.get_events(session.scene.id)

    generate_utterance(session.characters["tomas"], events, session.director.contexts, fake_llm)

    system, _, _ = fake_llm.calls[-1]
    assert "still trying to do" in system


def test_a_want_is_never_put_in_somebody_elses_prompt(table):
    """Authored inner life, like the persona. It describes the character
    rather than the room, so it is safe for them and nobody else."""
    his = wants(table.store, table.characters["tomas"])
    hers = wants(table.store, table.characters["maria"])

    assert "music box" in his
    assert "music box" not in hers


def test_the_narrator_is_never_given_them(table):
    """What somebody wants is not something the room can see."""
    import inspect

    from fabula import narrator

    assert "wants(" not in inspect.getsource(narrator)


def test_only_the_most_pressing_few(table):
    """A list is not a character."""
    from fabula.models import Goal
    from fabula.persistence import GOALS_IN_CONTEXT

    character = table.characters["maria"]
    character.goals = [
        Goal(id=f"g{i}", description=f"want number {i}", priority=i / 10) for i in range(8)
    ]

    text = wants(table.store, character)

    assert text.count(";") == GOALS_IN_CONTEXT - 1
    assert "want number 7" in text  # the highest priority survives
    assert "want number 0" not in text


def test_a_character_with_nothing_left_to_want_says_nothing(table):
    assert wants(table.store, table.characters["elena"]) == ""


# --- a goal that is touched raises the bid ------------------------------

def test_a_goal_at_stake_makes_them_press(table):
    """Mechanical, not only prose. Reactivity is a reflex; this is a
    standing want, and it stops applying once the goal is closed."""
    tomas = table.characters["tomas"]
    about_it = table.director.build_event(
        "utterance", "elena", "kitchen", "What happened to the music box?"
    )
    small_talk = table.director.build_event("utterance", "elena", "kitchen", "The soup needs salt.")

    assert goal_at_stake(tomas, about_it, "full", table.world, table.store) == 0.9
    assert goal_at_stake(tomas, small_talk, "full", table.world, table.store) == 0.0

    quiet, _ = heuristic_bid(tomas, small_talk, "full")
    pressed, reason = heuristic_bid(tomas, about_it, "full", stake=0.9)
    assert pressed - quiet == pytest.approx(0.9 * GOAL_STAKE)
    assert "want" in reason


def test_half_hearing_it_is_not_hearing_it(table):
    """Reading `event.content` is only safe at full fidelity, where the
    perceived content *is* the content."""
    tomas = table.characters["tomas"]
    event = table.director.build_event(
        "utterance", "elena", "kitchen", "What happened to the music box?"
    )

    assert goal_at_stake(tomas, event, "degraded", table.world, table.store) == 0.0


def test_a_goal_written_as_prose_alone_moves_nothing(table):
    """Nothing here can match "sort out grandmother's belongings fairly"
    against a line of dialogue, and pretending otherwise would be a
    keyword list masquerading as understanding."""
    maria = table.characters["maria"]
    assert [g.about for g in maria.goals] == [None]

    event = table.director.build_event(
        "utterance", "elena", "kitchen", "We should sort out her belongings."
    )

    assert goal_at_stake(maria, event, "full", table.world, table.store) == 0.0


# --- and it closes when the subject comes up ----------------------------

def test_a_secret_stops_being_one_once_it_is_out(table):
    assert open_goals(table, "tomas") == ["protect_secret"]

    table.say(SECRET)

    assert open_goals(table, "tomas") == []


def test_it_closes_only_for_those_who_heard_it(fake_llm):
    """The important one. A secret that came out in a room he was not in
    has not stopped being a secret *to him*, and he goes on guarding it."""
    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)
    session.store.append_event(
        session.director.build_event("arrival", "tomas", "study", "Tomás goes through.")
    )
    session.director._absorb()

    session.store.append_event(
        session.director.build_event(
            "utterance", "elena", "kitchen", "So the music box was already broken."
        )
    )
    session.director._absorb()

    assert open_goals(session, "tomas") == ["protect_secret"]


def test_a_closed_goal_stops_raising_the_bid(table):
    """Which is the difference between a want and a reflex."""
    tomas = table.characters["tomas"]
    event = table.director.build_event(
        "utterance", "elena", "kitchen", "What happened to the music box?"
    )
    assert goal_at_stake(tomas, event, "full", table.world, table.store) == 0.9

    table.say(SECRET)

    assert goal_at_stake(tomas, event, "full", table.world, table.store) == 0.0


def test_closing_survives_the_scene_it_happened_in(fake_llm, tmp_path):
    """Goals are durable state: a character does not walk into next week
    still guarding something everybody heard."""
    db = str(tmp_path / "scene.sqlite")
    first = Session.open(ASHGROVE, "the_reckoning", db_path=db, llm=fake_llm)
    first.say(SECRET)
    assert open_goals(first, "tomas") == []
    first.store.close()

    later = Session.open(ASHGROVE, "the_reckoning", db_path=db, llm=fake_llm)

    assert open_goals(later, "tomas") == []


def test_a_goal_naming_a_fact_that_does_not_exist_closes_nothing(table):
    """An authoring typo must not silently close a goal, or silently
    hold one open forever without anyone noticing."""
    from fabula.models import Goal

    character = table.characters["maria"]
    character.goals = [Goal(id="typo", description="something", about="no_such_fact")]
    table.store.append_event(
        table.director.build_event("utterance", "elena", "kitchen", "Anything at all.")
    )
    events = table.store.get_events(table.scene.id)
    projected = table.director.contexts.project(character, events)

    assert close_reached_goals(table.store, character, projected, table.world) == []
    assert open_goals(table, "maria") == ["typo"]
