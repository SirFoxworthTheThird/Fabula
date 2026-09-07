"""Characters knowing who is in the room with them.

Found by playtesting against a real model: Maria, standing in the
kitchen with Elena, asked "Where's Elena?". Context was assembled purely
from perceived events, so nobody was ever told who was present — an
engine gap no model can compensate for.
"""
from fabula.agents import strip_name_prefix
from fabula.director import Director
from fabula.memory import co_present
from fabula.narrator import Narrator

SECRET = "music box"


def test_co_present_names_only_the_same_room(scenario, store, fake_llm):
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    events = store.get_events(scene.id)

    with_elena = co_present(characters["elena"], characters, events)

    # Elena starts in the kitchen with Tomás; Maria is in the study.
    assert [c.id for c in with_elena] == ["tomas"]


def test_presence_never_names_someone_a_room_away(scenario, store, fake_llm):
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)

    situation = director.contexts.situation(
        characters["elena"], store.get_events(scene.id)
    )

    assert "Maria" not in situation  # she is in the study
    assert "Tomás" in situation
    assert "kitchen" in situation


def test_presence_follows_people_as_they_move(scenario, store, fake_llm):
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    store.append_event(
        director.build_event("arrival", "maria", "kitchen", "Maria comes through.")
    )

    events = store.get_events(scene.id)
    situation = director.contexts.situation(characters["elena"], events)

    assert "Maria" in situation
    assert {c.id for c in co_present(characters["elena"], characters, events)} == {
        "tomas",
        "maria",
    }


def test_a_character_alone_is_told_so(scenario, store, fake_llm):
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)

    situation = director.contexts.situation(characters["maria"], store.get_events(scene.id))

    assert "alone" in situation
    assert "study" in situation


def test_the_situation_line_reaches_the_character_prompt(scenario, store, fake_llm):
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)

    context = director.contexts.for_character(characters["elena"], store.get_events(scene.id))

    assert context.startswith("(You are in the kitchen. With you: Tomás.)")


def test_presence_does_not_leak_what_happens_elsewhere(scenario, store, fake_llm):
    """Knowing someone is in the room is the plainest perception there is.
    Knowing what they said in another room is not."""
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    store.append_event(
        director.build_event(
            "utterance", "tomas", "kitchen", "I broke Grandma's music box."
        )
    )

    context = director.contexts.for_character(characters["maria"], store.get_events(scene.id))

    assert SECRET not in context.lower()
    assert "alone" in context


def test_a_generated_line_does_not_keep_its_own_name_prefix():
    """Models answer "Tomás: I'm fine" when asked to speak as Tomás, and
    clients render the speaker themselves."""
    assert strip_name_prefix("Tomás: I'm fine.", "Tomás") == "I'm fine."
    assert strip_name_prefix("Tomás: Tomás: I'm fine.", "Tomás") == "I'm fine."
    assert strip_name_prefix("Tomas: I'm fine.", "Tomás") == "I'm fine."  # accent-insensitive
    assert strip_name_prefix("I'm fine.", "Tomás") == "I'm fine."
    # A colon that is not a name prefix survives untouched.
    assert strip_name_prefix("Here's the thing: I'm fine.", "Tomás") == "Here's the thing: I'm fine."


def test_the_narrator_is_told_whose_character_not_to_play(scenario, fake_llm):
    from fabula.session import Session
    from tests.conftest import ASHGROVE

    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)
    assert session.director.narrator.protagonist == "Elena"

    session.say("Anyone there?")

    narration = [call for call in fake_llm.calls if call[2] == "__narrator__"]
    for system, _prompt, _key in narration:
        assert "Elena is played by someone else" in system


def test_the_narrator_never_narrates_the_players_own_action(scenario, store, fake_llm):
    """A prompt asking it not to was not enough: a weak model answered a
    move with "Elena's light blue dress caught the dim light", inventing a
    dress and playing the one character the player controls. Not bidding
    is the only version of the rule a model cannot ignore."""
    world, characters, scene = scenario
    narrator = Narrator(fake_llm, protagonist="Elena", protagonist_id="elena")
    director = Director(store, world, characters, scene, narrator, fake_llm)
    hers = director.build_event("arrival", "elena", "study", "Elena comes in.")
    theirs = director.build_event("arrival", "maria", "study", "Maria comes in.")

    assert narrator.bid(hers, [hers], world) is None
    assert narrator.bid(theirs, [theirs], world) is not None


def test_a_character_is_told_not_to_volunteer_what_they_guard(scenario, store, fake_llm):
    """`protects` made Tomás deflect when asked, but nothing stopped him
    raising it himself — in playtesting he opened by announcing he had
    been repairing the music box."""
    from fabula.agents import generate_utterance

    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)

    generate_utterance(characters["tomas"], [], director.contexts, fake_llm)
    system, _prompt, _key = fake_llm.calls[-1]

    assert "never raise it yourself" in system
    assert SECRET in system  # his own secret, in his own prompt: no leak

    # Someone who guards nothing gets no such line.
    fake_llm.calls.clear()
    generate_utterance(characters["maria"], [], director.contexts, fake_llm)
    assert "never raise it yourself" not in fake_llm.calls[-1][0]


def test_the_narrator_does_not_narrate_a_narration(scenario, store, fake_llm):
    """Two near-identical narrations in a row was the observed failure."""
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    narration = director.build_event("narration", None, "kitchen", "The kettle ticks.")
    pressure_beat = director.build_event(
        "arrival", "maria", "kitchen", "Maria comes through.", metadata={"pressure_id": "x"}
    )

    assert director.narrator.bid(narration, [narration], world) is None
    assert director.narrator.bid(pressure_beat, [pressure_beat], world) is None
