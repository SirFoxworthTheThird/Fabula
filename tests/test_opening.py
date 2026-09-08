"""How a scene opens.

A story that opens on a bare prompt is a text box: the room has a name
and nothing in it until somebody thinks to type /look. The narrator has
always had a 0.9 "establish the scene" bid for exactly this — and it
could never fire, because by the time anything is bid on the player has
already spoken and the log is no longer empty. So the opening is not bid
on at all now; it is the one narration a scene owes the player.
"""
from pathlib import Path

import pytest

from fabula.llm import FakeLLM
from fabula.session import Session

from tests.conftest import ASHGROVE

WORLDS = ASHGROVE.parent
SECRET_WORD = "music box"


class Blurts(FakeLLM):
    """A narrator that puts the scene's secret in the scenery."""

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        return "A music box sits open on the dresser, its lid split."


def test_a_scene_opens_with_a_line_of_its_own():
    session = Session.open(ASHGROVE, "the_dinner", llm=FakeLLM())

    opened = session.store.get_events(session.scene.id)

    assert opened[0].kind == "narration"
    assert opened[0].actor_id is None
    assert opened[0].location_id == session.here(), "the room the player is standing in"


def test_and_then_whoever_is_there_may_speak_first():
    """A story that waits for the player to speak first puts the whole
    burden of starting it on them: you arrive somewhere, nobody says
    anything, and the only way to find out you are not alone is to talk
    to the air. Tomás is in the kitchen; the kitchen can say so."""
    session = Session.open(ASHGROVE, "the_dinner", llm=FakeLLM())

    opened = session.store.get_events(session.scene.id)

    assert [e.kind for e in opened] == ["narration", "utterance"]
    assert opened[1].actor_id == "tomas"


def test_the_opening_is_a_hello_not_a_conversation():
    """One beat. Two characters talking to each other before the player
    has typed is a scene that started without them."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=FakeLLM())

    spoken = [e for e in session.store.get_events(session.scene.id) if e.actor_id]

    assert len(spoken) <= 1


def test_the_opening_beat_is_not_a_pressure():
    """A greeting is the room noticing you; a pressure is the director
    escalating, and a story whose first move is its own complication has
    started without you."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=FakeLLM())

    opened = session.store.get_events(session.scene.id)

    assert not any(e.metadata.get("pressure_id") for e in opened)


def test_the_player_is_shown_it_before_they_type():
    """It is the first thing in their own projection, which is what every
    client renders on connect."""
    session = Session.open(ASHGROVE, "the_dinner", llm=FakeLLM())

    perceived = session.perceived_so_far()

    assert perceived, "there is something on screen before they type"
    assert perceived[0].event.kind == "narration"


def test_only_the_room_it_describes_hears_it():
    """It is an ordinary event, so it is filtered like one: Maria is in
    the study and the kitchen is being described."""
    session = Session.open(ASHGROVE, "the_dinner", llm=FakeLLM())
    events = session.store.get_events(session.scene.id)

    maria = session.director.contexts.project(session.characters["maria"], events)

    assert maria == []


def test_the_next_scene_opens_too(fake_llm):
    """A seam in the story is still a curtain going up."""
    first = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    first.say("You broke Grandma's music box, didn't you.")
    second = first.go_on()

    assert second is not None
    assert second.store.get_events(second.scene.id)[0].kind == "narration"


def test_a_resumed_story_is_not_re_established(tmp_path, fake_llm):
    """Nobody wants the room described to them again in the middle of a
    conversation they are already having."""
    path = str(tmp_path / "story.sqlite")
    first = Session.open(ASHGROVE, "the_dinner", db_path=path, llm=fake_llm)
    first.say("Tomás?")
    before = len(first.store.get_events(first.scene.id))
    first.close()
    first.store.close()

    again = Session.open(ASHGROVE, "the_dinner", db_path=path, llm=fake_llm)

    assert len(again.store.get_events(again.scene.id)) == before


def test_the_opening_line_may_not_name_a_secret():
    """A room description reaches the log as narration, so a keyword in
    one hands the secret to everybody in the room before anyone has
    spoken — and can end an arc on its first beat. The authored
    description stands in instead."""
    session = Session.open(ASHGROVE, "the_dinner", llm=Blurts())

    opened = session.store.get_events(session.scene.id)

    assert SECRET_WORD not in opened[0].content.lower()
    assert opened[0].content == session.world.rooms["kitchen"].description


def test_opening_an_empty_room_costs_one_model_call():
    """Ana starts the long dark alone in the mess. The curtain is not
    absorbed as a memory — nobody needs a durable belief about what the
    room they are standing in looks like, and that would be a reading per
    character."""
    llm = FakeLLM()
    session = Session.open(WORLDS / "winterlight", "the_long_dark", llm=llm)

    assert session.present() == []
    assert len(llm.calls) == 1


def test_a_greeting_costs_the_people_in_the_room_and_nobody_else():
    """The bill for opening a scene is bounded by who is standing there:
    the line, a bid each, and at most one reply."""
    llm = FakeLLM()
    session = Session.open(ASHGROVE, "the_reckoning", llm=llm)

    here = len(session.present())
    assert len(llm.calls) <= 1 + here + 1


def test_a_player_alone_is_answered_by_the_room():
    """The transcript that prompted this: a player alone in a room spoke,
    and every line came back "No one answers." Ana starts the long dark
    alone in the mess."""
    session = Session.open(WORLDS / "winterlight", "the_long_dark", llm=FakeLLM())

    perceived = session.say("Is anyone here?")

    answered = [p for p in perceived if p.event.actor_id != session.user_character.id]
    assert answered, "somebody — something — responded"
    assert all(p.event.kind == "narration" for p in answered)


def test_the_room_does_not_answer_over_somebody_who_can():
    """The narrator answering for an empty room must not turn into the
    narrator answering instead of the person standing there."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=FakeLLM())

    perceived = session.say("Tomás, you have been quiet all evening.")

    spoken = [p for p in perceived if p.event.kind == "utterance"]
    assert any(p.event.actor_id != session.user_character.id for p in spoken)
