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

    assert opened[-1].kind == "utterance"
    assert opened[-1].actor_id == "tomas"


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

    # The generated one: the scene's own opening words come before it and
    # are the author's, not the model's.
    generated = [
        e for e in session.store.get_events(session.scene.id)
        if e.kind == "narration" and not e.metadata.get("opening")
    ]

    assert SECRET_WORD not in generated[0].content.lower()
    assert generated[0].content == session.world.rooms["kitchen"].description


def test_opening_an_empty_room_costs_one_model_call():
    """Ana starts the long dark alone in the mess. The curtain is not
    absorbed as a memory — nobody needs a durable belief about what the
    room they are standing in looks like, and that would be a reading per
    character."""
    llm = FakeLLM()
    session = Session.open(WORLDS / "winterlight", "the_long_dark", llm=llm)

    assert session.present() == []
    assert len(llm.calls) == 1, "the scene's own opening words cost nothing"


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


# --- The first thing you read -----------------------------------------


def test_a_scene_says_what_you_have_walked_into():
    """Every other platform on this shelf opens with one: a paragraph
    that says what the situation is, so the first thing asked of somebody
    is not "what do you say" to a room they know nothing about."""
    session = Session.open(ASHGROVE, "the_dinner", llm=FakeLLM())

    first = session.perceived_so_far()[0]

    assert first.event.metadata.get("opening") is True
    assert first.perceived_content == session.scene.opening
    assert "Sunday" in first.perceived_content


def test_it_is_addressed_to_the_player_and_perceived_by_nobody_else():
    """Which is what lets it be written in the second person, and lets it
    say what only this character would know coming in. Through the
    ordinary perception path — `audibility: private` — so it is the same
    choke point every leak test already covers."""
    session = Session.open(ASHGROVE, "the_dinner", llm=FakeLLM())
    events = session.store.get_events(session.scene.id)
    briefing = events[0]

    assert briefing.audibility == "private"
    assert briefing.addressed_to == [session.user_character.id]
    for other in ("tomas", "maria"):
        seen = session.director.contexts.project(session.characters[other], events)
        assert not any(p.event.seq == briefing.seq for p in seen), other


def test_it_costs_nothing():
    """Authored rather than generated: the same every time, free, and
    good prose instead of whatever a model made of a room name."""
    llm = FakeLLM()
    Session.open(WORLDS / "winterlight", "the_long_dark", llm=llm)

    # One call, and it is the room — not the opening.
    assert len(llm.calls) == 1
    assert all("place:" in str(key) for _, _, key in llm.calls)


def test_a_scene_without_one_still_opens(tmp_path):
    """Optional: a world written before this existed opens on the room
    and whoever is standing in it, exactly as it did."""
    import shutil

    import yaml

    world = tmp_path / "ashgrove"
    shutil.copytree(ASHGROVE, world)
    scene = world / "scenes" / "the_dinner.yaml"
    written = yaml.safe_load(scene.read_text(encoding="utf-8"))
    written.pop("opening")
    scene.write_text(yaml.safe_dump(written), encoding="utf-8")

    session = Session.open(world, "the_dinner", llm=FakeLLM())

    assert session.scene.opening == ""
    assert session.perceived_so_far(), "the room still gets described"
    assert not any(
        p.event.metadata.get("opening") for p in session.perceived_so_far()
    )


def test_it_is_said_once_per_scene_and_comes_back_on_resume(tmp_path, fake_llm):
    """It is the scene's first words, not something repeated at every
    turn — and a player picking the story back up should still be able to
    read what they walked into."""
    path = str(tmp_path / "story.sqlite")
    first = Session.open(ASHGROVE, "the_dinner", db_path=path, llm=fake_llm)
    first.say("Tomás?")
    first.close()
    first.store.close()

    again = Session.open(ASHGROVE, "the_dinner", db_path=path, llm=fake_llm)
    openings = [
        e for e in again.store.get_events(again.scene.id) if e.metadata.get("opening")
    ]

    assert len(openings) == 1
    assert again.perceived_so_far()[0].event.metadata.get("opening") is True


def test_every_shipped_scene_says_what_you_have_walked_into():
    """A guard on the authoring. A scene with no opening drops somebody
    into a room with a name and nothing else, which is the thing this
    fixes."""
    from fabula.loader import catalogue, load_scene

    for world in catalogue(WORLDS):
        for scene in world["scenes"]:
            opening = load_scene(WORLDS / world["id"], scene["id"]).opening
            assert opening.strip(), f"{world['id']}/{scene['id']}"
            assert len(opening) > 80, f"{world['id']}/{scene['id']} is barely a sentence"
