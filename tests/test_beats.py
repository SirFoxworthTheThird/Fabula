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


# --- Saying it twice --------------------------------------------------


class Loops(FakeLLM):
    """A model that answers every character prompt with the same line,
    which is what a small one does when a scene runs out of road."""

    def __init__(self, line: str = "I'm going to check on grandmother's belongings."):
        super().__init__()
        self.line = line
        self.tries = 0

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        if key in ("tomas", "maria"):
            self.tries += 1
            return self.line
        return super().complete(system, prompt, key)


def test_nobody_says_the_same_line_twice_in_a_row():
    """Their own lines are in the context they were given, and a small
    model repeats them anyway: measured on a 1.5B, Maria said one
    sentence twice inside four lines. That reads as the app being broken
    rather than as a character insisting."""
    llm = Loops()
    session = Session.open(ASHGROVE, "the_reckoning", llm=llm)
    for said in ("Tomás?", "Say something.", "Please."):
        session.say(said)

    for character_id in ("tomas", "maria"):
        theirs = [
            e.content for e in session.store.get_events(session.scene.id)
            if e.actor_id == character_id
        ]
        assert all(
            first != second for first, second in zip(theirs, theirs[1:])
        ), f"{character_id} repeated themselves word for word"
    session.close()


def test_the_same_line_again_is_caught_however_it_is_reworded():
    """Word for word was the whole rule and it was not enough. Measured
    against a 3B over fourteen player lines, Maria announced the same
    intention eight times and no two were identical — paraphrase, not
    repetition, is how a small model loops."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=FakeLLM())
    said = session.store.append_event(
        session.director.build_event("utterance", "tomas", "kitchen", "It was nothing.")
    )
    events = session.store.get_events(session.scene.id)
    assert events[-1].seq == said.seq
    caught = lambda who, line: session.director._already_said(who, line, events)

    assert caught("tomas", "It was nothing.")
    assert caught("tomas", "  it was NOTHING!  ")
    assert caught("tomas", "It was nothing much."), "the same sentence, started again"
    # And parroting whoever just spoke, which a small model does as
    # readily as it repeats itself.
    assert caught("maria", "It was nothing.")
    # A different answer is still a different answer.
    assert not caught("tomas", "I broke it in March.")
    assert not caught("tomas", "Ask Maria.")
    session.close()


# Maria's actual lines, in order, from a real run against Qwen2.5-3B on
# `ashgrove/the_dinner`. The first four are a conversation. Everything
# after is one intention announced over and over, and the word-for-word
# rule caught none of it, because no two are identical.
THE_LOOP = [
    ("I'll take care of these letters, Elena. You're right, I've been neglecting you.", False),
    ("Elena, I promise to make time for you from now on.", False),
    ("I've been meaning to talk to you about something important.", False),
    ("I understand, Elena. I've just... needed some time to myself lately.", False),
    ("I'll start the letter sorting then.", False),
    ("I'll begin with the letters then.", True),
    ("I'll start with the ones that seem urgent.", True),
    ("I'll start with the oldest letters first.", True),
    ("I'll start with the ones addressed to you, Elena.", True),
    ("I'll start by checking the oldest ones, Elena.", True),
    ("I'll start by checking the oldest letters, Elena.", True),
    ("I'll start with the oldest letters, Elena.", True),
]


def test_the_loop_the_guard_was_rebuilt_for():
    """Replayed line by line, as it happened. Not every paraphrase is
    caught — the first one is not yet a loop, and two of these share
    almost no vocabulary with anything before them — but the run is
    broken, which is the difference between a character insisting and
    the app looking broken."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=FakeLLM())
    build = session.director.build_event
    caught = []
    for line, looping in THE_LOOP:
        events = session.store.get_events(session.scene.id)
        flagged = session.director._already_said("maria", line, events)
        assert not (flagged and not looping), f"a real line was blocked: {line}"
        caught.append(flagged)
        session.store.append_event(build("utterance", "maria", "kitchen", line))

    repeats = [flagged for flagged, (_, looping) in zip(caught, THE_LOOP) if looping]
    assert sum(repeats) >= 5, f"only {sum(repeats)} of {len(repeats)} caught"
    session.close()


def test_but_a_conversation_about_one_thing_is_not_a_loop():
    """The false positive worth avoiding: a scene about letters has
    everybody saying "letters", and that is the scene rather than a
    model stuck in a groove."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=FakeLLM())
    build = session.director.build_event
    session.store.append_event(
        build("utterance", "maria", "kitchen", "The letters are in the study.")
    )
    events = session.store.get_events(session.scene.id)

    for different in (
        "Start with the letters from March.",
        "You never sort anything.",
        "I'll get my coat.",
    ):
        assert not session.director._already_said("maria", different, events), different
    session.close()


def test_a_whole_sentence_does_not_come_round_twice_in_one_scene():
    """A short line can honestly repeat — "No." twice is a person. A
    sentence about the coffee and the herbs, word for word, twice, is a
    loop, and a 1.5B produced exactly that."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=FakeLLM())
    build = session.director.build_event
    long_line = "The aroma of coffee wafts through it, mingling with herbs from the garden."
    session.store.append_event(build("utterance", "tomas", "kitchen", long_line))
    session.store.append_event(build("utterance", "tomas", "kitchen", "No."))
    session.store.append_event(build("utterance", "tomas", "kitchen", "Because of the rain."))
    session.store.append_event(build("utterance", "maria", "kitchen", "Sort them out."))
    events = session.store.get_events(session.scene.id)

    assert session.director._already_said("tomas", long_line, events)
    assert not session.director._already_said("tomas", "No.", events)
    session.close()


def test_a_narration_that_plays_the_player_is_dropped():
    """The prompt has said never to describe them since M0, and a small
    model does it anyway — measured on a 1.5B: "Elena's finger brushed
    against the dusty glass of a photo album", which Elena never did.
    Playing the one character somebody else is holding is the worst
    thing the narrator can do, so the deterministic version of the rule
    is the one that counts."""
    class Plays(FakeLLM):
        def complete(self, system, prompt, key=None):
            if key == "__narrator__":
                return "Elena's finger brushes the dusty glass of the photo album."
            return super().complete(system, prompt, key)

    session = Session.open(ASHGROVE, "the_reckoning", llm=Plays())
    for said in ("Tomás?", "You have been quiet.", "Say something.", "Please."):
        session.say(said)

    written = [e.content for e in session.store.get_events(session.scene.id)]

    assert not any("Elena's finger" in line for line in written)
    session.close()
