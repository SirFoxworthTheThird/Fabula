"""Beliefs as readings, not echoes.

A memory used to be the perceived line stored verbatim. Nobody remembers
a conversation as a transcript — they remember what they took it to
mean, and two people in the same room take it to mean different things.

The verbatim echo stays in `content`; the reading goes in
`interpretation`, alongside it rather than instead of it. That is the
whole safety argument: the provable half survives with no model at all,
and a reading that misbehaves costs a nicety, never the record.
"""
import pytest

from fabula.db import EventStore
from fabula.director import Director
from fabula.interpret import INTERPRETATION_WINDOW, interpret, invented_fact
from fabula.llm import FakeLLM
from fabula.memory import project
from fabula.models import Event
from fabula.narrator import Narrator
from fabula.persistence import begin_scene
from fabula.summaries import span_key

SECRET = "music box"


def said(scene, seq, content, actor="elena", location="study", audibility="room"):
    return Event(
        id=seq,
        scene_id=scene.id,
        seq=seq,
        story_time=scene.start_time,
        kind="utterance",
        actor_id=actor,
        location_id=location,
        content=content,
        audibility=audibility,
    )


def a_director(scenario, store, llm):
    world, characters, scene = scenario
    begin_scene(store, characters)
    return Director(store, world, characters, scene, Narrator(llm), llm)


def test_a_belief_keeps_the_echo_and_gains_a_reading(scenario, store):
    llm = FakeLLM(canned={"interpret:maria": "Maria thinks her sister is stalling."})
    director = a_director(scenario, store, llm)

    director.run_turn(director.build_event("utterance", "elena", "study", "Maria, you're quiet."))

    beliefs = store.get_beliefs("maria")
    assert beliefs
    first = beliefs[0]
    assert first.content == "Maria, you're quiet."          # the projection, untouched
    assert first.interpretation == "Maria thinks her sister is stalling."


def test_without_a_model_a_belief_is_still_a_belief(scenario, store):
    """The verbatim half is what makes the store work offline. A reading
    is a nicety; losing it must cost nothing that was provable."""
    world, characters, scene = scenario
    window = project(characters["maria"], [said(scene, 1, "You're quiet.")], world)

    assert interpret(characters["maria"], window, world, store, llm=None) == ""


def test_a_reading_is_written_once_and_never_recomputed(scenario, store):
    """A memory rewritten every turn drifts underneath the character who
    holds it — the same rule summaries follow."""
    world, characters, scene = scenario
    llm = FakeLLM(canned={"interpret:maria": "Maria thinks something is off."})
    window = project(characters["maria"], [said(scene, 1, "You're quiet.")], world)

    first = interpret(characters["maria"], window, world, store, llm)
    calls = len(llm.calls)
    second = interpret(characters["maria"], window, world, store, llm)

    assert first == second == "Maria thinks something is off."
    assert len(llm.calls) == calls  # the second read cost nothing


def test_a_reading_that_invents_a_secret_is_thrown_away(scenario, store):
    """The important one. A memory is durable and crosses scenes, so a
    hallucination here is not a bad line — it is a false memory a
    character carries for good. Maria never perceived the music box, so a
    reading that names it is refused and she keeps her echo."""
    world, characters, scene = scenario
    leaky = FakeLLM(
        canned={"interpret:maria": "Maria is certain Tomás broke the music box."}
    )
    window = project(characters["maria"], [said(scene, 1, "You're quiet.")], world)

    reading = interpret(characters["maria"], window, world, store, leaky)

    assert reading == ""
    assert leaky.calls  # it really was asked, and really was refused


def test_the_refusal_is_remembered_so_a_bad_reading_is_not_paid_for_twice(scenario, store):
    world, characters, scene = scenario
    leaky = FakeLLM(canned={"interpret:maria": f"Maria knows about the {SECRET}."})
    window = project(characters["maria"], [said(scene, 1, "You're quiet.")], world)

    interpret(characters["maria"], window, world, store, leaky)
    calls = len(leaky.calls)
    interpret(characters["maria"], window, world, store, leaky)

    assert len(leaky.calls) == calls


def test_a_character_may_think_about_their_own_secret(scenario, store):
    """Tomás protects the music box, so naming it is not an invention —
    it is the thing he cannot stop thinking about."""
    world, characters, scene = scenario
    llm = FakeLLM(
        canned={"interpret:tomas": "Tomás is bracing for someone to ask about the music box."}
    )
    window = project(characters["tomas"], [said(scene, 1, "You're quiet.", location="kitchen")], world)

    reading = interpret(characters["tomas"], window, world, store, llm)

    assert SECRET in reading.lower()


def test_a_fact_they_already_perceived_is_not_an_invention(scenario, store):
    """Elena hears it said out loud; a reading of hers may name it."""
    world, characters, scene = scenario
    heard = said(scene, 1, "I broke Grandma's music box.", actor="tomas", location="kitchen")
    window = project(characters["elena"], [heard], world)

    assert window[0].perception == "full"
    assert invented_fact("Elena is reeling about the music box.", window,
                         characters["elena"], world, store) is None


def test_half_hearing_it_does_not_license_naming_it(scenario, store):
    """Maria in the study gets the degraded descriptor, which carries no
    words — so a reading of hers that names the secret invented it."""
    world, characters, scene = scenario
    shouted = said(
        scene, 1, "I broke Grandma's music box!", actor="tomas",
        location="kitchen", audibility="adjacent",
    )
    window = project(characters["maria"], [shouted], world)

    assert window[0].perception == "degraded"
    assert SECRET not in window[0].perceived_content.lower()
    assert invented_fact("Maria thinks it was about the music box.", window,
                         characters["maria"], world, store) == "music_box"


def test_the_reading_only_ever_sees_perceived_lines(scenario, store):
    """Structural, not prompted: whatever the prompt says, the only story
    text in it came through the projection."""
    world, characters, scene = scenario
    llm = FakeLLM()
    shouted = said(
        scene, 1, "I broke Grandma's music box!", actor="tomas",
        location="kitchen", audibility="adjacent",
    )
    window = project(characters["maria"], [shouted], world)

    interpret(characters["maria"], window, world, store, llm)

    system, prompt, _ = llm.calls[-1]
    assert SECRET not in f"{system}\n{prompt}".lower()


def test_the_window_is_bounded(scenario, store):
    """One model call per remembered moment per character is the cost, so
    the run-up it reads has to stay small."""
    world, characters, scene = scenario
    llm = FakeLLM()
    events = [said(scene, i + 1, f"line {i}") for i in range(20)]
    window = project(characters["maria"], events, world)[-INTERPRETATION_WINDOW:]

    interpret(characters["maria"], window, world, store, llm)

    _, prompt, _ = llm.calls[-1]
    assert "line 19" in prompt
    assert "line 15" not in prompt  # only the last four


def test_a_reading_is_not_paid_for_on_a_moment_too_dull_to_keep(scenario, store):
    """Encoding has a salience floor. Reading a moment about to be
    discarded is money spent on the weather."""
    from fabula.models import Belief
    from fabula.persistence import encode_belief

    world, characters, scene = scenario
    called = []
    forgettable = Belief(
        character_id="maria",
        subject_id="elena",
        content="nothing much",
        confidence=1.0,
        source_event_id=1,
        formed_at=scene.start_time,
        last_rehearsed=scene.start_time,
        salience=0.1,
    )

    stored = encode_belief(store, characters["maria"], forgettable,
                           interpret=lambda: called.append(1) or "read")

    assert stored is False
    assert called == []


def test_what_a_character_carries_in_reaches_the_next_scene(scenario, store):
    """A belief store nothing reads back is a write-only diary. This is
    the half that makes durability visible in play."""
    world, characters, scene = scenario
    llm = FakeLLM(canned={"interpret:maria": "Maria thinks her brother is avoiding her."})
    director = a_director(scenario, store, llm)
    director.run_turn(director.build_event("utterance", "elena", "study", "Maria, you're quiet."))

    later = scene.model_copy(update={"id": "the_morning_after"})
    next_director = Director(store, world, characters, later, Narrator(llm), llm)
    context = next_director.contexts.for_character(characters["maria"], [])

    assert "What you already believed" in context
    # The echo she formed it from, annotated with what she made of it —
    # never one replacing the other.
    assert "Maria, you're quiet." in context
    assert "you took it as: Maria thinks her brother is avoiding her." in context


def test_tonights_own_beliefs_are_not_replayed_as_memory(scenario, store):
    """They are already in the transcript above. Listing them again would
    have a character remember the last ten minutes twice."""
    llm = FakeLLM(canned={"interpret:maria": "Maria thinks her sister is stalling."})
    director = a_director(scenario, store, llm)
    events = director.run_turn(
        director.build_event("utterance", "elena", "study", "Maria, you're quiet.")
    )

    context = director.contexts.for_character(
        director.characters["maria"], director.store.get_events(director.scene.id)
    )

    assert store.get_beliefs("maria")             # she did form some
    assert "What you already believed" not in context


def test_the_same_impression_twice_is_listed_once(scenario, store):
    world, characters, scene = scenario
    llm = FakeLLM(canned={"interpret:maria": "Maria thinks something is being kept from her."})
    director = a_director(scenario, store, llm)
    for line in ("Maria, you're quiet.", "Maria, are you listening?"):
        director.run_turn(director.build_event("utterance", "elena", "study", line))

    later = scene.model_copy(update={"id": "the_morning_after"})
    next_director = Director(store, world, characters, later, Narrator(llm), llm)
    context = next_director.contexts.for_character(characters["maria"], [])

    assert len(store.get_beliefs("maria")) > 1
    assert context.count("Maria, you're quiet.") == 1


def test_carrying_memory_in_cannot_carry_the_secret_in(scenario, store):
    """The leak test, applied to the new path. Everything read back was
    encoded from Maria's own projection, so there is nothing in it for
    her to arrive knowing."""
    world, characters, scene = scenario
    llm = FakeLLM()
    director = a_director(scenario, store, llm)
    director.run_turn(
        director.build_event(
            "utterance", "tomas", "kitchen",
            "I broke Grandma's music box, and I let them blame the cat.",
        )
    )

    later = scene.model_copy(update={"id": "the_morning_after"})
    next_director = Director(store, world, characters, later, Narrator(llm), llm)
    context = next_director.contexts.for_character(characters["maria"], [])

    assert SECRET not in context.lower()
    # And Tomás, who said it, does arrive still carrying it.
    his = next_director.contexts.for_character(characters["tomas"], [])
    assert SECRET in his.lower()


def test_a_weak_reading_cannot_delete_the_memory_it_annotates(scenario, store):
    """Regression, and the reason the reading is additive. A model that
    returns something useless — a placeholder, a refusal, one bland
    word — must colour a memory, never stand in for it. Substituting
    would let the softest component in the engine quietly erase the
    record it was given."""
    world, characters, scene = scenario
    useless = FakeLLM(canned={"interpret:tomas": "Something happened."})
    director = a_director(scenario, store, useless)
    director.run_turn(
        director.build_event(
            "utterance", "tomas", "kitchen",
            "I broke Grandma's music box, and I let them blame the cat.",
        )
    )

    later = scene.model_copy(update={"id": "the_morning_after"})
    next_director = Director(store, world, characters, later, Narrator(useless), useless)
    his = next_director.contexts.for_character(characters["tomas"], [])

    assert SECRET in his.lower()             # the record survived
    assert "you took it as: Something happened." in his  # the reading is only an annotation


def test_the_player_is_never_read(fake_llm):
    """Nothing reads Elena's memory back into a prompt — she is holding
    it — so writing down what she privately thinks would be the engine
    deciding her inner life, and paying a model call to do it."""
    from fabula.session import Session

    from tests.conftest import ASHGROVE

    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)
    session.say("Tomás, you have been quiet all evening.")

    assert not [key for _, _, key in fake_llm.calls if key == "interpret:elena"]
    assert all(b.interpretation == "" for b in session.store.get_beliefs("elena"))
    assert session.store.get_beliefs("elena")  # she still remembers


def test_readings_can_be_switched_off_entirely(fake_llm):
    """One call per remembered moment per character dominates the bill
    for a scene. Turning it off costs the annotations and nothing that
    was provable."""
    from fabula.session import Session

    from tests.conftest import ASHGROVE

    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm, interpret_beliefs=False)
    session.say("Tomás, you have been quiet all evening.")

    assert not [key for _, _, key in fake_llm.calls if key and key.startswith("interpret:")]
    beliefs = session.store.get_beliefs("tomas")
    assert beliefs
    assert all(b.interpretation == "" for b in beliefs)
    assert any("quiet all evening" in b.content for b in beliefs)
