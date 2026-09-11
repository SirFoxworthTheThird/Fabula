"""Living a story rather than finishing one.

This app is not meant to feel like a game, and scenes, end conditions and
"onward" buttons are game furniture. Some stories want them — Ashgrove is
better for having a reckoning that ends and two mornings that follow it.
Plenty do not. Meeting a childhood friend after twenty years is not a
thing you complete; a school year is somewhere you stay.

The measurement that made this necessary, on `ashgrove/the_dinner` over
twenty-four player lines: all three authored pressures are spent by the
seventh, and every turn from the eighth on is identically two people
talking with a narration between them. Nothing reports it. The story does
not end, it goes slack — and no amount of authoring fixes that, because
the twentieth complication is one nobody wrote.

So a story can be played open-ended: no ending is reported, no seam is
crossed, and when the authored complications run out the engine writes
the next thing that happens.
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

from fabula.llm import FakeLLM
from fabula.models import Event, Pressure
from fabula.session import Session
from fabula.situations import MADE, QUIET, RESPITE, compose, drifting, is_invented

from tests.conftest import ASHGROVE


class Weather(FakeLLM):
    """A model that answers the situation call with something usable."""

    def __init__(self, what="The back door bangs once in the wind.", where="the kitchen"):
        super().__init__()
        self.what, self.where, self.asked = what, where, 0

    def complete(self, system, prompt, key=None):
        if key == "situation":
            self.asked += 1
            self.last_prompt = prompt
            return json.dumps({"what": self.what, "where": self.where})
        return super().complete(system, prompt, key)


def a_story(llm=None, open_ended=True, scene="the_dinner"):
    return Session.open(
        ASHGROVE, scene, llm=llm or Weather(), open_ended=open_ended,
        db_path=str(Path(tempfile.mkdtemp()) / "story.sqlite"),
    )


def played(session, lines=20):
    """What happened each turn, in the order it happened."""
    shapes = []
    for i in range(lines):
        before = len(session.store.get_events(session.scene.id))
        session.say(f"Line {i}: something a person would say.")
        shapes.append(session.store.get_events(session.scene.id)[before:])
    return shapes


# --- the failure this exists for ----------------------------------------

def test_a_world_runs_out_of_things_that_can_happen(fake_llm):
    """The baseline, kept as a test so it cannot quietly stop being true.
    Pressures have `max_fires`; once they are spent the story is over
    without saying so."""
    session = a_story(open_ended=False)
    turns = played(session, 20)

    happened = [
        i for i, events in enumerate(turns)
        if any(e.metadata.get("pressure_id") for e in events)
    ]
    assert happened, "the authored pressures do fire"
    assert max(happened) < 10, "and they are all spent early"
    assert not any(
        e.metadata.get("pressure_id") for events in turns[12:] for e in events
    ), "after which nothing can ever happen again"
    session.close()


def test_and_open_ended_it_keeps_happening(fake_llm):
    llm = Weather()
    session = a_story(llm)
    turns = played(session, 20)

    invented = [
        i for i, events in enumerate(turns)
        if any(e.metadata.get("invented") for e in events)
    ]
    assert len(invented) >= 2, "the story keeps having things happen in it"
    assert min(invented) > 5, "but not until the authored ones are spent"
    session.close()


def test_what_it_costs(fake_llm):
    """One call, only on a drift, never inside the respite. A feature that
    spends somebody's money every turn has not been paid for."""
    llm = Weather()
    session = a_story(llm)
    played(session, 20)

    assert llm.asked <= 20 // RESPITE + 1, f"{llm.asked} calls across 20 lines"
    session.close()


# --- when it decides the story has gone slack ---------------------------

WHEN = datetime(2024, 1, 1, 19, 0, 0)


def an_event(seq, kind="utterance", actor="elena", **metadata):
    return Event(
        id=seq, scene_id="s", seq=seq, story_time=WHEN, kind=kind, actor_id=actor,
        location_id="kitchen", content="Something.", audibility="room",
        metadata=metadata,
    )


def _talk(seq, actor="elena", **metadata):
    return an_event(seq, actor=actor, **metadata)


def test_a_room_still_making_its_own_tension_is_left_alone():
    """The same threshold the authored sandbox pressures use, because it
    is the same judgement."""
    events = [_talk(i) for i in range(1, 12)]

    assert not drifting(events, "elena", top_bid=QUIET)
    assert not drifting(events, "elena", top_bid=0.9)
    assert drifting(events, "elena", top_bid=QUIET - 0.01)


def test_it_counts_what_the_player_said_rather_than_log_positions():
    """A scene with four rooms of people racks up events fast and still
    feels like nothing is happening."""
    chatter = [_talk(i, actor="tomas") for i in range(1, 30)]

    assert not drifting(chatter, "elena", top_bid=0.0), "nobody but the player counts"

    mine = chatter + [_talk(30 + i, actor="elena") for i in range(RESPITE)]
    assert drifting(mine, "elena", top_bid=0.0)


def test_narration_is_not_something_happening():
    """The measured trap. Atmosphere beats never run out, so a story can
    produce one every turn for ever while nothing changes — which is
    exactly the state this has to detect, not be fooled by."""
    events = []
    for i in range(1, 4 * RESPITE):
        events.append(_talk(i, actor="elena"))
        events.append(an_event(100 + i, kind="narration", actor=None))

    assert drifting(events, "elena", top_bid=0.0)


def test_but_a_pressure_that_fired_resets_the_clock():
    events = [_talk(i, actor="elena") for i in range(1, 20)]
    events.append(an_event(99, kind="narration", actor=None,
                           pressure_id="maria_comes_through"))

    assert not drifting(events, "elena", top_bid=0.0)


# --- what it is allowed to write ----------------------------------------

def test_an_invented_situation_is_an_ordinary_pressure(scenario):
    world, characters, _ = scenario
    made = compose(Weather(), world, [], characters, "kitchen")

    assert isinstance(made, Pressure) and is_invented(made)
    assert made.id.startswith(MADE)
    # Narration only. An arrival needs somebody the scene said might turn
    # up and a state change needs a scene written around it; neither is a
    # thing to decide mid-turn on a model's say-so.
    assert made.effect == {"kind": "narration", "location": "kitchen"}
    assert made.effect["location"] in world.rooms


def test_it_lands_somewhere_real(scenario):
    world, characters, _ = scenario
    lost = compose(Weather(where="the roof garden"), world, [], characters, "study")

    assert lost.effect["location"] == "study", "where the player is, not somewhere invented"


def test_it_may_not_name_a_secret(scenario):
    """The same guard the generated room descriptions get. A situation
    naming a fact's own words would raise the subject as scenery,
    satisfying `fact_spoken` before anybody in the story had said it."""
    world, characters, _ = scenario
    leaked = compose(
        Weather(what="The music box is sitting there on the mantel."),
        world, [], characters, "kitchen",
    )

    assert leaked is None


def test_the_writer_is_never_told_the_secrets(scenario):
    """Narrower than the director, on purpose. The deterministic guard can
    only catch a fact's actual keywords, and a writer that had been told
    the secret could paraphrase around them — "he looks at the empty space
    on the mantel" names nothing and gives everything away. A writer that
    was never told cannot allude to it."""
    world, characters, _ = scenario
    llm = Weather()
    compose(llm, world, [], characters, "kitchen")

    for fact in world.facts.values():
        for keyword in fact.keywords:
            assert keyword.lower() not in llm.last_prompt.lower()
    assert "protect" not in llm.last_prompt.lower()


def test_a_model_that_answers_with_nothing_costs_a_beat_and_not_the_story(scenario):
    world, characters, _ = scenario

    assert compose(FakeLLM(), world, [], characters, "kitchen") is None


# --- the seams, hidden ---------------------------------------------------

def test_a_story_played_this_way_never_ends(fake_llm):
    """`the_reckoning` is over the moment the music box is said out loud,
    and leads to one of two mornings. Played open-ended it does neither."""
    session = a_story(scene="the_reckoning")
    session.say("You broke Grandma's music box, didn't you.")

    assert not session.ended()
    assert session.next_scene() is None
    assert session.go_on() is None
    session.close()


def test_and_played_the_other_way_it_still_does(fake_llm):
    session = a_story(fake_llm, open_ended=False, scene="the_reckoning")
    session.say("You broke Grandma's music box, didn't you.")

    assert session.ended()
    assert session.next_scene() is not None
    session.close()


def test_an_arc_does_not_ramp_toward_an_ending_it_will_never_reach(fake_llm):
    """An arc pushes its pressures harder the longer it runs, because it is
    climbing toward something. With nothing to climb toward that is just a
    scene that gets louder for ever.

    Measured against what the author allowed rather than against a number
    somebody once observed. The count moved when the engine learned to
    let people leave the room — `maria_comes_through` triggers on Maria
    being in the study, which in this scene could never happen before,
    so a pressure that had been dead became reachable. More of the
    author's material, which is not the thing this guards against.
    """
    session = a_story(scene="the_reckoning")
    played(session, 8)

    events = session.store.get_events(session.scene.id)
    fired = [
        e for e in events
        if e.metadata.get("pressure_id") and not e.metadata.get("invented")
    ]
    budget = sum(p.max_fires for p in session.director.pressures if p.max_fires)
    assert len(fired) <= budget, "no pressure fires more often than it was written to"

    # And the shape of it: an arc that ramps spends more of itself the
    # longer it runs. This one does not get louder.
    halfway = events[len(events) // 2].seq
    early = [e for e in fired if e.seq <= halfway]
    late = [e for e in fired if e.seq > halfway]
    assert len(late) <= len(early), "equilibrium, not escalation"
    session.close()


# --- and it is the story's, not the caller's ----------------------------

def test_how_a_story_is_played_survives_being_put_down(tmp_path):
    from fabula.library import Library

    library = Library(root=tmp_path, worlds_root=ASHGROVE.parent)
    started = library.start("ashgrove", "the_dinner", llm=FakeLLM(), open_ended=True)
    story_id = library.list()[0].id
    started.close()

    # The caller says otherwise, and is ignored: a story you left running
    # does not acquire endings because you resumed it from somewhere else.
    back = library.resume(story_id, llm=FakeLLM(), open_ended=False)

    assert back.director.open_ended
    assert library.list()[0].open_ended
    back.close()


def test_a_story_from_before_this_existed_resumes_the_way_it_was_played(tmp_path):
    from fabula.library import Library

    library = Library(root=tmp_path, worlds_root=ASHGROVE.parent)
    started = library.start("ashgrove", "the_dinner", llm=FakeLLM())
    story_id = library.list()[0].id
    started.close()

    back = library.resume(story_id, llm=FakeLLM())

    assert not back.director.open_ended, "scenes and endings, as it was started"
    back.close()
