"""Reading a line for what it reports, not only for what it says.

"I told my brother last week" is a claim that a perception happened
off-screen. Ignore it and a character who walks in later arrives
ignorant of something the scene established — the objection that started
this: being mentioned is free only for somebody who is pure scenery.

Keyword lists cannot catch it. A *thing* has a name, so `mentions_fact`
works in any language; a *meaning* does not, and asking an author to
enumerate the ways of saying "I told him" is asking them to enumerate a
language. So it is a model's judgement — confined to picking from a
closed set, with every referent checked against authored data before
anything is appended.

Most of what follows tests the checking, because the checking is the
whole safety argument.
"""
from pathlib import Path

import pytest

from fabula.classify import Transmission, classify, knows, worth_classifying
from fabula.llm import FakeLLM
from fabula.session import Session

WORLDS = Path(__file__).parent.parent / "worlds"
ASHGROVE = WORLDS / "ashgrove"


class Judge(FakeLLM):
    """Answers the classifier the way a competent model would; everything
    else falls through to the deterministic placeholder."""

    def __init__(self, answer: str = '{"reports_telling": false}'):
        super().__init__()
        self.answer = answer

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        if key and key.startswith("classify:"):
            return self.answer
        return super().complete(system, prompt, key)


def a_scene(answer: str, scene: str = "the_dinner"):
    return Session.open(ASHGROVE, scene, llm=Judge(answer))


def says(session, actor, text, room="kitchen"):
    session.store.append_event(
        session.director.build_event("utterance", actor, room, text)
    )
    return session.director.record_transmission(
        session.store.get_events(session.scene.id)[-1]
    )


TOLD_MARIA = '{"reports_telling": true, "to": "maria", "fact": "music_box"}'


# --- the gate -----------------------------------------------------------

def test_a_line_naming_no_fact_never_reaches_the_model(scenario):
    """The cheap deterministic filter, and it skips almost every line.
    Keyword matching, so it costs nothing and works in any script."""
    world, _, scene = scenario
    from fabula.models import Event
    from datetime import datetime

    def line(text):
        return Event(
            scene_id="s", seq=1, story_time=datetime(2024, 1, 1), kind="utterance",
            actor_id="tomas", location_id="kitchen", content=text,
        )

    assert worth_classifying(line("I told her about the music box."), world) is True
    assert worth_classifying(line("The soup needs salt."), world) is False


def test_nothing_happens_without_a_model(scenario, store, fake_llm):
    """FakeLLM's placeholder is unparseable, so the feature is simply off
    rather than guessing."""
    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)

    assert says(session, "tomas", "I told Maria about the music box.") is None


# --- what it does when it fires ----------------------------------------

def test_a_reported_telling_becomes_the_perception_it_describes():
    """Maria is in the study and could not have heard a word of tonight.
    She knows anyway, because Tomás says he told her before it."""
    session = a_scene(TOLD_MARIA)
    events = session.store.get_events(session.scene.id)
    before = session.director.contexts.for_character(session.characters["maria"], events)
    assert "music box" not in before.lower()

    beat = says(session, "tomas", "She knows. I already told her about the music box.")
    session.director._absorb()

    assert beat is not None
    events = session.store.get_events(session.scene.id)
    after = session.director.contexts.for_character(session.characters["maria"], events)
    assert "music box" in after.lower()
    assert any("music box" in b.content.lower() for b in session.store.get_beliefs("maria"))


def test_the_telling_is_private_to_the_two_of_them():
    """The *report* was public — Elena heard him say it. The telling it
    reconstructs was not."""
    session = a_scene(TOLD_MARIA)
    beat = says(session, "tomas", "I already told her about the music box.")
    events = session.store.get_events(session.scene.id)

    def perceived(cid):
        return [
            p.event.seq
            for p in session.director.contexts.project(session.characters[cid], events)
        ]

    assert beat.seq in perceived("maria")
    assert beat.seq in perceived("tomas")   # you know what you did
    assert beat.seq not in perceived("elena")


def test_it_is_visible_rather_than_silent():
    """Nothing is hidden: the beat is an ordinary event in the log, so it
    reaches the transcript and the reveal like anything else."""
    session = a_scene(TOLD_MARIA)

    beat = says(session, "tomas", "I already told her about the music box.")

    assert "told Maria about music box" in beat.content
    assert beat.metadata["transmission"] == "music_box"


def test_the_words_are_the_worlds():
    """It lands in somebody's perception, so like everything else that
    does, an author writes it."""
    from fabula.loader import load_world

    phrasing = load_world(WORLDS / "vilamar").phrasing

    rendered = phrasing.transmission.format(speaker="A", recipient="B", subject="o barco")
    assert rendered == "A já tinha contado a B sobre o barco, antes desta noite."


# --- the checking, which is the safety argument -------------------------

def test_you_cannot_pass_on_what_you_do_not_know():
    """The load-bearing guard. Without it a player says "everyone already
    knows my secret" and the scene dissolves."""
    session = a_scene(TOLD_MARIA)

    # Elena has perceived nothing and holds no secret of her own.
    assert says(session, "elena", "I already told Maria about the music box.") is None


def test_a_fact_the_author_never_wrote_is_refused():
    session = a_scene('{"reports_telling": true, "to": "maria", "fact": "the_inheritance"}')

    assert says(session, "tomas", "I already told her about the music box.") is None


def test_somebody_who_is_not_in_the_story_is_refused():
    """The "my brother" case, and it is deliberately out of scope: this
    reconstructs a telling to a character the author wrote, not a person
    the model invented."""
    session = a_scene('{"reports_telling": true, "to": "my_brother", "fact": "music_box"}')

    assert says(session, "tomas", "I already told my brother about it.") is None


def test_the_speaker_is_whoever_spoke_not_whoever_the_model_picks():
    """`from` is never taken from the answer at all."""
    session = a_scene('{"reports_telling": true, "to": "maria", "fact": "music_box"}')

    beat = says(session, "tomas", "I already told her about the music box.")

    assert beat.actor_id == "tomas"


def test_nobody_can_be_told_by_themselves():
    session = a_scene('{"reports_telling": true, "to": "tomas", "fact": "music_box"}')

    assert says(session, "tomas", "I told myself about the music box.") is None


@pytest.mark.parametrize(
    "answer",
    ["not json at all", "", "{", '{"reports_telling": "maybe"', "[1, 2, 3]", '{"nope": 1}'],
)
def test_a_malformed_answer_is_a_no_rather_than_a_crash(answer):
    """A classifier that killed the turn would be worse than one that
    missed a beat."""
    session = a_scene(answer)

    assert says(session, "tomas", "I already told her about the music box.") is None


def test_wanting_to_tell_is_not_having_told():
    """The prompt draws this line, and this is the test that will catch it
    drifting on a real model — it is the classification most likely to go
    wrong, and the one with a consequence."""
    session = a_scene('{"reports_telling": false}')

    assert says(session, "tomas", "I should tell Maria about the music box.") is None


# --- how it sits with the rest of the engine ----------------------------

def test_a_reported_telling_does_not_end_an_arc():
    """`the_reckoning` is over when the music box is finally said out
    loud. One pair of ears in the past tense is not that.

    Said alone, so the line itself reaches nobody — otherwise the
    utterance names the fact in front of the room and *that* ends the
    scene, quite correctly, before the beat is even reached.
    """
    session = a_scene(TOLD_MARIA, scene="the_reckoning")
    for other in ("maria", "elena"):
        session.store.append_event(
            session.director.build_event("arrival", other, "study", f"{other} goes through.")
        )
    session.director._absorb()

    beat = says(session, "tomas", "I already told her about the music box.")

    assert beat is not None                       # the telling was recorded
    assert beat.metadata["transmission"] == "music_box"
    assert session.ended() is False               # and it did not end the scene


def test_the_turn_it_landed_in_can_be_taken_again():
    """Reversible is half of what makes an inferred world change
    acceptable; visible is the other half."""
    session = a_scene(TOLD_MARIA)
    session.say("Tomás, what did you do?")
    told = len([
        e for e in session.store.get_events(session.scene.id)
        if e.metadata.get("transmission")
    ])

    session.regenerate()

    again = [
        e for e in session.store.get_events(session.scene.id)
        if e.metadata.get("transmission")
    ]
    assert len(again) == told  # not doubled, not orphaned


def test_knows_accepts_an_authored_secret_and_a_memory(scenario, store):
    world, characters, _ = scenario

    assert knows(characters["tomas"], world.facts["music_box"], [], store) is True
    assert knows(characters["maria"], world.facts["music_box"], [], store) is False
