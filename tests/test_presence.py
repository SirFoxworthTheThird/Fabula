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


def test_the_situation_line_sits_last_next_to_the_question(scenario, store, fake_llm):
    """It is current state, not history. Buried above a grown scene it
    gets contradicted — a character claimed the player was alone in a
    room she was standing in with two other people."""
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    for line in ("Tomas?", "Are you listening?", "Well?"):
        director.run_turn(director.build_event("utterance", "elena", "kitchen", line))

    context = director.contexts.for_character(characters["elena"], store.get_events(scene.id))

    assert context.endswith("(Right now: you are in the kitchen. With you: Tomás.)")
    assert len(context.splitlines()) > 3  # it really is after the history


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
    her_line = director.build_event("utterance", "elena", "study", "Anyone here?")
    theirs = director.build_event("arrival", "maria", "study", "Maria comes in.")

    assert narrator.bid(her_line, [her_line], world) is None
    assert narrator.bid(theirs, [theirs], world) is not None


def test_the_player_walking_in_gets_the_room_described_not_herself(scenario, store, fake_llm):
    """Refusing to narrate anything the player does left the most natural
    moment for scene-setting silent. The room is not the person."""
    world, characters, scene = scenario
    narrator = Narrator(fake_llm, protagonist="Elena", protagonist_id="elena")
    director = Director(store, world, characters, scene, narrator, fake_llm)
    arrival = director.build_event("arrival", "elena", "study", "Elena comes in.")

    bid = narrator.bid(arrival, [arrival], world)
    assert bid is not None
    assert "room" in bid.one_line_reason

    narrator.generate(arrival, [arrival], world)
    system, prompt, key = fake_llm.calls[-1]

    assert key == "place:study"
    assert "Elena" not in prompt          # she is not what is being described
    assert "Do not mention Elena" in system
    assert "Letters in date order" in prompt  # grounded in the authored room


def test_a_room_with_no_authored_description_still_works(scenario, store, fake_llm):
    world, characters, scene = scenario
    bare = world.model_copy(deep=True)
    bare.rooms["study"].description = ""
    narrator = Narrator(fake_llm, protagonist="Elena", protagonist_id="elena")

    narrator.describe_place("study", bare)

    assert "What is here:" not in fake_llm.calls[-1][1]


def test_the_narrator_speaks_up_when_a_scene_is_all_talk(scenario, store, fake_llm):
    """Spec §7 lists a lull as one of its triggers; without it a scene is
    a wall of dialogue with no room around it."""
    from fabula.narrator import LULL_WINDOW

    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    talk = [
        director.build_event("utterance", "tomas", "kitchen", f"Line {i}.")
        for i in range(LULL_WINDOW)
    ]

    assert director.narrator.bid(talk[-1], talk, world) is not None
    assert director.narrator.bid(talk[-1], talk[:-1], world) is None  # not yet a lull


def test_a_guarded_subject_is_a_reminder_not_a_rule(scenario, store, fake_llm):
    """`protects` names what a character would rather not discuss and who
    can currently hear them. It is deliberately not enforced: a character
    who cannot slip is less believable, and a secret that cannot escape by
    accident can only come out by authorial fiat."""
    from fabula.agents import generate_utterance

    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)

    generate_utterance(characters["tomas"], [], director.contexts, fake_llm)
    system, _prompt, _key = fake_llm.calls[-1]

    assert SECRET in system  # his own secret, in his own prompt: no leak
    assert "do not raise it lightly" in system
    assert "Weigh who can hear you" in system
    # Tomás starts in the kitchen with Elena, a room away from Maria.
    assert "In the room with you: Elena." in system

    # Someone who guards nothing gets no such line.
    fake_llm.calls.clear()
    generate_utterance(characters["maria"], [], director.contexts, fake_llm)
    assert "do not raise it lightly" not in fake_llm.calls[-1][0]


def test_the_guard_line_names_whoever_walked_in(scenario, store, fake_llm):
    from fabula.agents import generate_utterance

    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    store.append_event(
        director.build_event("arrival", "maria", "kitchen", "Maria comes through.")
    )

    events = store.get_events(scene.id)
    generate_utterance(characters["tomas"], events, director.contexts, fake_llm)

    assert "In the room with you: Elena, Maria." in fake_llm.calls[-1][0]


def test_a_character_alone_is_told_the_room_is_empty(scenario, store, fake_llm):
    from fabula.agents import generate_utterance

    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    store.append_event(
        director.build_event("arrival", "tomas", "study", "Tomás steps through.")
    )
    store.append_event(
        director.build_event("arrival", "maria", "kitchen", "Maria comes through.")
    )

    events = store.get_events(scene.id)
    generate_utterance(characters["tomas"], events, director.contexts, fake_llm)

    assert "There is no one else here." in fake_llm.calls[-1][0]


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
