"""Withholding: a character visibly declining to answer.

Without this, a reticent character pressed on their secret simply loses
the argmax, which reads exactly like their not being in the room.
"""
from fabula.bidding import WITHHOLD_RETICENCE, get_bid, recently_withheld, withholding_bid
from fabula.director import Director
from fabula.llm import FakeLLM
from fabula.memory import ContextBuilder, project
from fabula.models import Event
from fabula.narrator import Narrator

SECRET = "music box"


def asked(scene, seq=1, content="Tomas, what happened to the music box?", to=("tomas",)):
    return Event(
        id=seq,
        scene_id=scene.id,
        seq=seq,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="elena",
        location_id="kitchen",
        content=content,
        audibility="room",
        addressed_to=list(to),
    )


def test_a_protected_fact_asked_directly_produces_a_withhold_bid(scenario):
    world, characters, scene = scenario
    tomas = characters["tomas"]
    assert "music_box" in tomas.protects
    assert tomas.traits.reticence >= WITHHOLD_RETICENCE

    bid = withholding_bid(tomas, asked(scene), "full", [], world)

    assert bid is not None
    desire, reason = bid
    assert desire > 0.6
    # The rationale is director-only, and still must not name the thing.
    assert SECRET not in reason.lower()


def test_no_withholding_when_the_subject_did_not_come_up(scenario):
    world, characters, scene = scenario

    bid = withholding_bid(
        characters["tomas"], asked(scene, content="Tomas, how was the drive?"), "full", [], world
    )

    assert bid is None


def test_no_withholding_when_someone_else_was_asked(scenario):
    world, characters, scene = scenario

    bid = withholding_bid(characters["tomas"], asked(scene, to=("maria",)), "full", [], world)

    assert bid is None


def test_a_question_to_the_room_still_lands_on_the_one_hiding_it(scenario):
    """Nobody has to say his name for the question to be his to dodge."""
    world, characters, scene = scenario

    bid = withholding_bid(characters["tomas"], asked(scene, to=()), "full", [], world)

    assert bid is not None


def test_no_withholding_from_a_character_who_protects_nothing(scenario):
    world, characters, scene = scenario
    maria = characters["maria"].model_copy(update={"protects": []})

    bid = withholding_bid(maria, asked(scene, to=("maria",)), "full", [], world)

    assert bid is None


def test_no_withholding_from_a_forthcoming_character(scenario):
    """Reticence is what makes deflection in character."""
    world, characters, scene = scenario
    open_book = characters["tomas"].model_copy(
        update={"traits": characters["tomas"].traits.model_copy(update={"reticence": 0.1})}
    )

    assert withholding_bid(open_book, asked(scene), "full", [], world) is None


def test_no_withholding_on_a_half_heard_question(scenario):
    """Only at full perception — someone who caught the question through a
    wall did not hear it clearly enough to dodge it, and their perceived
    content would not contain the subject anyway."""
    world, characters, scene = scenario

    assert withholding_bid(characters["tomas"], asked(scene), "degraded", [], world) is None


def test_withholding_does_not_repeat_immediately(scenario, store):
    world, characters, scene = scenario
    tomas = characters["tomas"]
    deflection = Event(
        id=2,
        scene_id=scene.id,
        seq=2,
        story_time=scene.start_time,
        kind="action",
        actor_id="tomas",
        location_id="kitchen",
        content="Tomás turns to the sink.",
        audibility="room",
        metadata={"withheld": True},
    )
    log = [asked(scene), deflection]

    assert recently_withheld("tomas", log)
    assert withholding_bid(tomas, asked(scene, seq=3), "full", log, world) is None


def test_get_bid_returns_a_withhold_kind(scenario, store, fake_llm):
    world, characters, scene = scenario
    contexts = ContextBuilder(world, scene.id, store, fake_llm)

    bid = get_bid(characters["tomas"], asked(scene), "full", [], contexts, fake_llm)

    assert bid.kind == "withhold"


def test_the_director_turns_a_withhold_into_a_visible_beat(scenario, store, fake_llm):
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)

    director.run_turn(
        director.build_event(
            "utterance", "elena", "kitchen", "Tomas, what happened to the music box?"
        )
    )

    beats = [e for e in store.get_events(scene.id) if e.metadata.get("withheld")]
    assert beats, "Tomás should have deflected"
    beat = beats[0]
    assert beat.kind == "action"
    assert beat.actor_id == "tomas"  # not answering is visible and attributable
    assert beat.location_id == "kitchen"


def test_the_narrator_is_never_told_what_is_being_withheld(scenario, store, fake_llm):
    """The beat is perceived by everyone in the room. A narrator that knew
    the secret could hand it over in the act of describing it being kept."""
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)

    director.run_turn(
        director.build_event(
            "utterance", "elena", "kitchen", "Tomas, what happened to the music box?"
        )
    )

    withhold_calls = [
        call for call in fake_llm.calls if call[2] and call[2].startswith("withhold:")
    ]
    assert withhold_calls
    for system, prompt, _key in withhold_calls:
        assert SECRET not in system.lower()
        assert SECRET not in prompt.lower()

    # And the beat itself names nothing.
    beat = next(e for e in store.get_events(scene.id) if e.metadata.get("withheld"))
    assert SECRET not in beat.content.lower()


def test_a_deflection_is_perceived_by_the_room_but_not_next_door(scenario, store, fake_llm):
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    director.run_turn(
        director.build_event(
            "utterance", "elena", "kitchen", "Tomas, what happened to the music box?"
        )
    )
    beat = [e for e in store.get_events(scene.id) if e.metadata.get("withheld")]

    assert project(characters["elena"], beat, world)  # she sees him dodge
    assert project(characters["maria"], beat, world) == []  # the study does not
