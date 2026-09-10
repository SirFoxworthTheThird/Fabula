"""A story, rather than a menu of scenes.

Information asymmetry is what makes the fiction trustworthy. It is not
what anybody plays for, and for a long time this engine could play a
scene and not a story: scenes were a flat list, `end_condition` said one
was over without saying what followed, and the durable state we carried
across the seam had no seam to cross.

A scene now declares where the story goes from it, conditioned on what
happened in it — in the same vocabulary as endings and pressures,
because an author should not need a third language to say "if she was in
the room".
"""
from pathlib import Path

import pytest

from fabula.commands import run_command
from fabula.llm import FakeLLM
from fabula.loader import load_scenario
from fabula.persistence import unresolved_goals
from fabula.pressures import next_scene, scene_state
from fabula.session import Session

from fabula.shelf import SHIPPED as WORLDS  # noqa: E402
ASHGROVE = WORLDS / "ashgrove"
SECRET = "I broke Grandma's music box."


def a_reckoning(fake_llm, maria_stays=True):
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    if not maria_stays:
        session.store.append_event(
            session.director.build_event("arrival", "maria", "study", "Maria goes through.")
        )
    session.store.append_event(
        session.director.build_event("utterance", "tomas", "kitchen", SECRET)
    )
    session.director._absorb()
    return session


# --- choosing what follows ---------------------------------------------

def test_the_same_ending_leads_to_two_different_mornings(fake_llm):
    """Consequence, which is the whole point of the feature. What
    happened chooses what happens next."""
    with_her = a_reckoning(fake_llm, maria_stays=True)
    without_her = a_reckoning(fake_llm, maria_stays=False)

    assert with_her.ended() and without_her.ended()
    assert with_her.next_scene() == "the_morning_after"
    assert without_her.next_scene() == "nobody_said_a_word"


def test_branches_are_read_in_order_with_a_fallback_last(scenario):
    world, characters, scene = scenario
    state = scene_state([], characters, world)

    successors = [
        {"scene": "specific", "when": {"turns_elapsed": "> 100"}},
        {"scene": "fallback"},
    ]
    assert next_scene(successors, state, world.facts) == "fallback"
    assert next_scene([{"scene": "only", "when": {}}], state, world.facts) == "only"


def test_a_scene_with_nowhere_to_go_ends_the_story(fake_llm):
    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)

    assert session.scene.next == []
    assert session.next_scene() is None
    assert session.go_on() is None


# --- crossing the seam ---------------------------------------------------

def test_the_story_carries_what_the_last_scene_did(fake_llm):
    """One store for the whole story. Everybody arrives holding what the
    scene before did to them — which is what all that durable state was
    for, and until now it crossed a seam that did not exist."""
    first = a_reckoning(fake_llm, maria_stays=True)
    trust = first.store.get_relationships("maria")["tomas"].trust
    beliefs = len(first.store.get_beliefs("maria"))
    assert not unresolved_goals(first.store, first.characters["tomas"])

    second = first.go_on()

    assert second.scene.id == "the_morning_after"
    assert second.store.get_relationships("maria")["tomas"].trust == trust
    assert len(second.store.get_beliefs("maria")) == beliefs
    assert not unresolved_goals(second.store, second.characters["tomas"])


def test_they_walk_in_already_believing_something(fake_llm):
    second = a_reckoning(fake_llm, maria_stays=True).go_on()
    events = second.store.get_events(second.scene.id)

    context = second.director.contexts.for_character(second.characters["maria"], events)

    assert "What you already believed" in context


def test_the_new_scene_starts_with_a_log_of_its_own(fake_llm):
    """Events are per scene; only the characters carry. The new scene's
    log holds nothing but its own opening line."""
    second = a_reckoning(fake_llm, maria_stays=True).go_on()

    opened = second.store.get_events("the_morning_after")
    assert opened[0].kind == "narration"
    assert all(e.scene_id == "the_morning_after" for e in opened)
    assert second.store.get_events("the_reckoning")  # the old one is still there


def test_the_next_scene_puts_people_where_it_says(fake_llm):
    """Positions are the new scene's, not carried from the last."""
    second = a_reckoning(fake_llm, maria_stays=True).go_on()

    assert second.director.current_location(second.characters["maria"]) == "study"


def test_going_on_settles_the_turn_that_was_open(fake_llm, tmp_path):
    """A seam is not a take the player can ask to have again, and the
    scene before it has to reach disk."""
    from fabula.db import EventStore

    db = str(tmp_path / "story.sqlite")
    first = Session.open(ASHGROVE, "the_reckoning", db_path=db, llm=fake_llm)
    first.say("You broke Grandma's music box, didn't you.")
    written = len(first.store.get_events("the_reckoning"))

    second = first.go_on()
    second.close()

    assert len(EventStore(db).get_events("the_reckoning")) == written


# --- the clients ---------------------------------------------------------

def test_the_command_layer_goes_on_and_hands_back_the_new_scene(fake_llm):
    session = a_reckoning(fake_llm, maria_stays=True)

    outcome = run_command(session, "/next")

    assert outcome.went_on is not None
    assert outcome.went_on.scene.id == "the_morning_after"
    assert "morning after" in outcome.message


def test_asking_to_go_on_from_the_end_says_so(fake_llm):
    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)

    outcome = run_command(session, "/next")

    assert outcome.went_on is None
    assert outcome.message == "The story ends here."


def test_a_story_crosses_the_seam_over_http(fake_llm):
    from fastapi.testclient import TestClient

    from fabula.api import create_app

    app = create_app(worlds_root=WORLDS, llm=fake_llm)
    with TestClient(app) as client:
        opened = client.post(
            "/sessions", json={"world": "ashgrove", "scene": "the_reckoning"}
        ).json()
        session_id = opened["session_id"]
        client.post(
            f"/sessions/{session_id}/say",
            json={"text": "You broke Grandma's music box, didn't you."},
        )
        state = client.get(f"/sessions/{session_id}").json()
        assert state["ended"] is True and state["next_scene"] == "the_morning_after"

        went_on = client.post(f"/sessions/{session_id}/next").json()

        # A client holds a story, not a scene: the id survives the seam.
        assert went_on["session_id"] == session_id
        assert went_on["scene"] == "the_morning_after"
        assert client.post(f"/sessions/{session_id}/next").status_code == 409
