"""The line a scene says before the player has to.

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

    assert [e.kind for e in opened] == ["narration"]
    assert opened[0].actor_id is None
    assert opened[0].location_id == session.here(), "the room the player is standing in"


def test_the_player_is_shown_it_before_they_type():
    """It is the first thing in their own projection, which is what every
    client renders on connect."""
    session = Session.open(ASHGROVE, "the_dinner", llm=FakeLLM())

    perceived = session.perceived_so_far()

    assert len(perceived) == 1
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
    assert [e.kind for e in second.store.get_events(second.scene.id)] == ["narration"]


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


def test_it_costs_one_model_call():
    """Per scene, before the player types. The curtain is not absorbed as
    a memory — nobody needs a durable belief about what the room they are
    standing in looks like, and that would be a reading per character."""
    llm = FakeLLM()
    Session.open(ASHGROVE, "the_dinner", llm=llm)

    assert len(llm.calls) == 1


def test_a_player_alone_is_answered_by_the_room():
    """The transcript that prompted this: `arrival` puts Rook alone in
    the hall, and every line he spoke came back "No one answers." """
    session = Session.open(WORLDS / "ardenhall", "arrival", llm=FakeLLM())

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
