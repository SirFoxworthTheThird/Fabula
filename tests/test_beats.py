"""What the room could use, and who decides.

The narrator was handed a triggering event and asked for "scene-setting
narration". It knew what had just happened and nothing about what the
scene needed, so it fired rarely and wrote whatever the last line
suggested. A scene played that way is people talking in a white room.

The director says what to narrate now. It is the only omniscient
component, which is the whole design problem: an instruction written
freely from what it knows would be a channel from the world log into
prose everybody in the room perceives. So a beat is one id from a closed
set plus something anybody standing there can already see.
"""
from datetime import datetime
from pathlib import Path

import pytest

from fabula.beats import COOLDOWN, LULL_WINDOW, NOBODY_SPEAKS, QUIET_FOR, Beat, choose
from fabula.director import Director
from fabula.llm import FakeLLM
from fabula.models import Event
from fabula.narrator import BEATS, Narrator, as_bid
from fabula.session import Session

from tests.conftest import ASHGROVE

WORLDS = ASHGROVE.parent


def line(seq, actor, room="kitchen", content="Something.", kind="utterance", **metadata):
    return Event(
        scene_id="the_dinner",
        seq=seq,
        story_time=datetime(2024, 1, 1, 19, 0),
        kind=kind,
        actor_id=actor,
        location_id=room,
        content=content,
        metadata=metadata,
    )


@pytest.fixture
def world(scenario):
    return scenario[0]


@pytest.fixture
def cast(scenario):
    return scenario[1]


# --- When the room gets a beat ---------------------------------------


def test_a_run_of_talk_wants_the_room(world):
    talk = [line(i, "tomas") for i in range(1, LULL_WINDOW + 1)]

    beat = choose(talk[-1], talk, world)

    assert beat.id == "lull"


def test_a_deflection_gets_held_on(world):
    events = [line(1, "elena"), line(2, "tomas", kind="action", withheld=True)]

    beat = choose(events[-1], events, world)

    assert beat.id == "after_deflection"


def test_somebody_who_has_stopped_talking_gets_put_in_the_frame(world, cast):
    """That they are quiet is observable by everyone in the room. What
    they are quiet *about* would not be, and a beat never carries it."""
    events = [line(i, "tomas") for i in range(1, QUIET_FOR + 3)]
    standing = {cid: c for cid, c in cast.items() if cid in ("tomas", "maria", "elena")}
    standing["maria"] = standing["maria"].model_copy(update={"location_id": "kitchen"})

    beat = choose(events[-1], events, world, standing)

    assert beat.id == "held_back"
    assert beat.subject == "Maria"


def test_the_player_is_never_the_subject_of_one(world, cast):
    """A model told not to narrate the protagonist still answered a move
    with "Elena's light blue dress caught the dim light". Not offering
    her as a subject is the version of that rule a model cannot ignore."""
    events = [line(i, "tomas") for i in range(1, QUIET_FOR + 3)]

    beat = choose(events[-1], events, world, cast)

    assert beat.subject != "Elena"


def test_something_in_the_room_nobody_has_looked_at(fake_llm):
    """Items are placed by the author and visible to anybody standing
    there."""
    session = Session.open(WORLDS / "ardenhall", "arrival", llm=fake_llm)
    here = {item.name for item in session.world.items_in("hall")}
    assert here, "the hall has something in it to look at"
    # A room where nothing else is going on: no run of talk, nobody long
    # silent, and enough distance from the scene's own opening prose.
    events = [
        line(seq, "pell", room="hall", kind="action", content="waits by the stairs.")
        for seq in range(1, COOLDOWN + 3)
    ]

    beat = choose(events[-1], events, session.world)

    assert beat is not None and beat.id == "object"
    assert beat.subject in here


# --- When it does not ------------------------------------------------


def test_prose_is_never_gilded_with_more_prose(world):
    narration = line(1, None, kind="narration", content="The kettle ticks.")
    pressure = line(2, None, kind="narration", content="A door.", pressure_id="p")

    assert choose(narration, [narration], world) is None
    assert choose(pressure, [pressure], world) is None


def test_atmosphere_waits_after_a_narration(world):
    """Otherwise a scene turns into paragraphs of weather with dialogue
    in between."""
    events = [line(1, None, kind="narration", content="The kettle ticks.")]
    events += [line(i + 2, "tomas", kind="action", content="pours.") for i in range(COOLDOWN - 1)]

    assert choose(events[-1], events, world) is None, "too soon after the last one"

    events += [line(len(events) + 1 + i, "tomas") for i in range(LULL_WINDOW)]
    assert choose(events[-1], events, world).id == "lull", "and then the room gets one"


def test_prose_in_another_room_does_not_quiet_this_one(world):
    """A paragraph in the study is not a paragraph the kitchen just
    read."""
    elsewhere = [line(1, None, "study", kind="narration", content="The lamp hums.")]
    talk = [line(i + 2, "tomas") for i in range(LULL_WINDOW)]

    assert choose(talk[-1], elsewhere + talk, world).id == "lull"


def test_but_an_arrival_never_waits(world):
    """The cooldown is for atmosphere. Somebody walking in is the story
    moving, and losing it is worse than one paragraph too many."""
    events = [line(1, None, kind="narration", content="The kettle ticks.")]
    walk_in = line(2, "maria", kind="arrival", content="Maria comes in.")

    assert choose(walk_in, events + [walk_in], world).id == "arrival"


def test_the_player_speaking_is_not_a_reason_to_describe_the_player(world, cast):
    """Their own line back at them, dressed up, is the worst thing the
    narrator can do."""
    said = line(1, "elena", content="Tomás?")

    beat = choose(said, [said], world, cast, protagonist_id="elena")

    assert beat is None or beat.subject != "Elena"


# --- What a beat may carry -------------------------------------------


def test_a_beat_carries_nothing_but_a_name_or_a_thing(fake_llm):
    """The director is the only omniscient component and its output has
    to stay narrow. Nothing from beliefs, trust, goals or what anybody
    protects can be in here — a beat is an id and something visible."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    for said in ("Tomás?", "You have been quiet.", "Say something.", "Please."):
        session.say(said)
    events = session.store.get_events(session.scene.id)

    visible = {c.name for c in session.characters.values()}
    visible |= {item.name for item in session.world.items.values()}
    for index in range(2, len(events)):
        beat = choose(
            events[index], events[: index + 1], session.world, session.characters,
            protagonist_id="elena",
        )
        if beat and beat.subject:
            assert beat.subject in visible, beat
    session.close()


def test_every_beat_has_something_to_say(world):
    """A beat id the narrator has no instruction for would silently
    become "write some narration", which is what this replaced."""
    for beat_id in ("lull", "after_deflection", "held_back", "object", "alone",
                    "arrival", "departure", "time_skip"):
        assert beat_id in BEATS


def test_the_reason_a_beat_was_chosen_never_reaches_an_event(fake_llm, scenario, store):
    """Bid rationales stop at the director. A reason like "somebody in
    the room has not spoken" is director-only reasoning about a
    character, and it is not what gets written down."""
    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    talk = [
        director.build_event("utterance", "tomas", "kitchen", f"Line {i}.")
        for i in range(LULL_WINDOW)
    ]
    for event in talk:
        store.append_event(event)

    beat = choose(talk[-1], store.get_events(scene.id), world, characters)
    bid = as_bid(beat)

    assert bid.one_line_reason == beat.reason
    assert all(beat.reason not in e.content for e in store.get_events(scene.id))


# --- What the narrator does with one ----------------------------------


def test_the_beat_is_what_the_narrator_is_asked_for(fake_llm, scenario):
    world, characters, scene = scenario
    narrator = Narrator(fake_llm, protagonist="Elena", protagonist_id="elena")
    said = line(1, "tomas", content="It's nothing.")

    narrator.generate(said, [said], world, beat=Beat("held_back", 0.45, "r", "Maria"),
                      present=["Tomás", "Maria"])
    _, prompt, _ = fake_llm.calls[-1]

    assert "Maria is standing there" in prompt
    assert "Who is here: Tomás, Maria" in prompt
    assert "It's nothing." in prompt  # the last line, which the room heard


def test_a_narration_that_names_a_fact_is_dropped(scenario, store):
    """It would raise the subject in front of everybody in the room —
    satisfying `fact_spoken` — and end an arc that was waiting for
    somebody to say it out loud. Nobody said it."""
    world, characters, scene = scenario

    class Blurts(FakeLLM):
        def complete(self, system, prompt, key=None):
            if key == "__narrator__":
                return "The music box sits open on the mantel."
            return super().complete(system, prompt, key)

    llm = Blurts()
    session = Session.open(ASHGROVE, "the_reckoning", llm=llm)
    for said in ("Tomás?", "You have been quiet.", "Say something.", "Please."):
        session.say(said)

    events = session.store.get_events(session.scene.id)

    assert not any("music box" in e.content.lower() for e in events)
    assert not session.ended(), "and the arc is still waiting for somebody to say it"
    session.close()


# --- A turn always answers -------------------------------------------


def test_every_turn_gives_the_player_something(fake_llm):
    """The complaint that started this: you speak into a room with
    somebody standing in it and the app says "No one answers." Nobody
    bid and no beat was due, so the turn produced nothing — or produced
    it two rooms away, which is the same silence where you are."""
    session = Session.open(WORLDS / "ardenhall", "arrival", llm=fake_llm)

    for said in ("Are you waiting for me?", "Which house am I in?",
                 "Nobody has told me anything.", "Shall we go up, then?"):
        answered = [
            p for p in session.say(said)
            if p.event.actor_id != session.user_character.id
        ]
        assert answered, f"nothing came back from {said!r}"
    session.close()


def test_the_room_that_answers_is_the_one_you_are_standing_in(fake_llm):
    """A pressure firing in the office is the story moving. It is not an
    answer to somebody waiting in the hall."""
    session = Session.open(WORLDS / "ardenhall", "arrival", llm=fake_llm)

    for said in ("Are you waiting for me?", "Which house am I in?"):
        for projected in session.say(said):
            assert projected.event.location_id == session.here()
    session.close()


def test_the_last_word_it_hangs_a_beat_on_is_one_you_heard(fake_llm):
    """`last_event` may be two rooms away, and putting it in the prompt
    would narrate the player's room out of words they never perceived —
    a leak dressed as scenery."""
    session = Session.open(WORLDS / "ardenhall", "arrival", llm=fake_llm)
    session.say("Are you waiting for me?")
    elsewhere = session.store.append_event(
        session.director.build_event(
            "utterance", "vance", "office", "The seventh file stays in the drawer."
        )
    )

    heard, beat = session.director._heard_last(session.store.get_events(session.scene.id))

    assert heard.seq != elsewhere.seq
    assert heard.location_id == session.here()
    assert beat.id == NOBODY_SPEAKS
    session.close()


# --- Who picks between them -------------------------------------------


class Picks(FakeLLM):
    """A director with an opinion about which beat this moment wants."""

    def __init__(self, wants: str):
        super().__init__()
        self.wants = wants
        self.asked: list[str] = []
        self.narration: list[str] = []

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        if key == "beat":
            self.asked.append(prompt)
            return self.wants
        if key == "__narrator__":
            self.narration.append(prompt)
        return super().complete(system, prompt, key)


def a_crowded_pause(llm) -> Session:
    """the_reckoning: three people in one room, so a moment is usually
    more than one thing at once."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=llm)
    for said in ("Tomás?", "You have been quiet.", "Say something.", "Please."):
        session.say(said)
    return session


def test_the_model_chooses_which_beat_and_the_narrator_writes_that_one():
    """The room has gone quiet *and* somebody has stopped talking. Which
    of those a scene wants is a judgement, not a rule."""
    lull, held = Picks("lull"), Picks("held_back")
    a_crowded_pause(lull).close()
    a_crowded_pause(held).close()

    assert lull.asked, "it was asked at all"
    assert "They have been talking" in lull.narration[-1]
    assert "has not said anything for a while" in held.narration[-1]


def test_it_can_only_answer_with_a_beat_that_was_offered():
    """The model proposes and the engine disposes, as everywhere else it
    is asked anything. It cannot invent a beat, write an instruction, or
    reach past the closed vocabulary."""
    invented = Picks("narrate that Tomás is hiding something")
    session = a_crowded_pause(invented)

    assert invented.asked
    assert "Tomás is hiding something" not in "".join(invented.narration)
    # And what it wrote was the deterministic first choice instead.
    assert "has not said anything for a while" in invented.narration[-1]
    session.close()


def test_it_is_only_asked_when_there_is_a_choice_to_make():
    """One beat is not a decision, and a call that changes nothing is
    somebody's money."""
    llm = Picks("lull")
    session = Session.open(WORLDS / "winterlight", "the_long_dark", llm=llm)
    session.say("Is anyone else awake?")

    assert llm.asked == []
    session.close()


def test_it_is_only_asked_once_the_narration_is_going_to_be_written():
    """The beat with the highest desire is what the narrator bids with.
    Choosing between them before knowing whether it won would be paying
    for a decision nobody uses."""
    llm = Picks("lull")
    session = a_crowded_pause(llm)
    written = [
        e for e in session.store.get_events(session.scene.id)
        if e.kind == "narration" and not e.metadata.get("pressure_id")
        and not e.metadata.get("opening")
    ]

    assert len(llm.asked) <= len(written)
    session.close()


def test_the_choice_is_made_on_what_the_room_can_see():
    """The director is omniscient. There is no reason to hand any of that
    to something whose whole job is picking between three labels."""
    llm = Picks("lull")
    session = a_crowded_pause(llm)
    asked = "\n".join(llm.asked)

    assert "music box" not in asked.lower()
    for character in session.characters.values():
        assert character.persona[:40] not in asked
    session.close()


def test_it_can_be_turned_off():
    """One call per narration is a real bill, and somebody paying it
    should be able to stop."""
    llm = Picks("lull")
    session = Session.open(ASHGROVE, "the_reckoning", llm=llm, direct_beats=False)
    for said in ("Tomás?", "You have been quiet.", "Say something.", "Please."):
        session.say(said)

    assert llm.asked == []
    session.close()
