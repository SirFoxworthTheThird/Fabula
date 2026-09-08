"""Deterministic trust movement.

Relationships were durable but inert: authored, stored, aged, and never
consulted by anything that decided what happened. This is the one thing
that moves them during play, and the one thing that reads them back.

The rule is small on purpose. Judging trust from the *content* of what
was said is exactly the sort of correction invariant 2 forbids — it
would need a model deciding whether someone was being sincere, and a
wrong call there quietly rewrites a character's inner life. Witnessing a
visible refusal to answer needs no judgment at all.
"""
import pytest

from fabula.bidding import DISTRUST_ATTENTION, NEUTRAL_TRUST, heuristic_bid
from fabula.memory import project
from fabula.models import Event, Relationship
from fabula.persistence import (
    WITHHOLD_TRUST_FLOOR,
    WITHHOLD_TRUST_MOVE,
    begin_scene,
    witnessed_withholding,
)
from fabula.session import Session

from tests.conftest import ASHGROVE


def a_withholding(scene, seq=1, location="kitchen", audibility="room"):
    return Event(
        id=seq,
        scene_id=scene.id,
        seq=seq,
        story_time=scene.start_time,
        kind="action",
        actor_id="tomas",
        location_id=location,
        content="Tomás looks at the table and says nothing.",
        audibility=audibility,
        metadata={"withheld": True},
    )


def projection_of(character, event, world, characters, scene):
    """The one perceived event, exactly as the engine would hand it over."""
    projected = project(character, [event], world)
    return projected[0] if projected else None


@pytest.fixture
def seeded(scenario, store):
    world, characters, scene = scenario
    # The dinner starts Maria in the study; these need her at the table,
    # watching her brother not answer.
    characters["maria"].location_id = "kitchen"
    begin_scene(store, characters)
    return world, characters, scene


def test_trust_drops_when_a_witness_sees_someone_refuse_to_answer(seeded, store):
    world, characters, scene = seeded
    before = store.get_relationships("maria")["tomas"].trust
    seen = projection_of(characters["maria"], a_withholding(scene), world, characters, scene)
    assert seen.perception == "full"

    moved = witnessed_withholding(store, "maria", seen)

    after = store.get_relationships("maria")["tomas"].trust
    assert moved is True
    assert after < before
    assert after == pytest.approx(
        WITHHOLD_TRUST_FLOOR + (before - WITHHOLD_TRUST_FLOOR) * (1 - WITHHOLD_TRUST_MOVE)
    )


def test_it_never_reaches_zero(seeded, store):
    """A character who trusts you at nothing has nothing left to lose,
    which is the least interesting place to leave them."""
    world, characters, scene = seeded

    for seq in range(40):
        seen = projection_of(
            characters["maria"], a_withholding(scene, seq=seq + 1), world, characters, scene
        )
        witnessed_withholding(store, "maria", seen)

    assert store.get_relationships("maria")["tomas"].trust > WITHHOLD_TRUST_FLOOR


def test_a_witness_who_only_half_heard_it_holds_it_against_nobody(seeded, store):
    """The degraded descriptor never carries a name, so Elena in the
    study cannot know who that was — and a number that moved anyway
    would be invariant 1 leaking through arithmetic instead of prose."""
    world, characters, scene = seeded
    elena = characters["elena"]
    elena.location_id = "study"
    before = store.get_relationships("elena")["tomas"].trust

    seen = projection_of(elena, a_withholding(scene, audibility="adjacent"), world, characters, scene)
    assert seen.perception == "degraded"
    moved = witnessed_withholding(store, "elena", seen)

    assert moved is False
    assert store.get_relationships("elena")["tomas"].trust == before


def test_a_character_who_was_not_there_loses_no_trust(seeded, store):
    world, characters, scene = seeded
    maria = characters["maria"]
    maria.location_id = "garden"
    before = store.get_relationships("maria")["tomas"].trust

    seen = projection_of(maria, a_withholding(scene), world, characters, scene)

    assert seen is None  # never perceived it at all
    assert store.get_relationships("maria")["tomas"].trust == before


def test_you_do_not_lose_faith_in_yourself_for_keeping_your_own_counsel(seeded, store):
    world, characters, scene = seeded
    before = store.get_relationships("tomas")["maria"].trust
    seen = projection_of(characters["tomas"], a_withholding(scene), world, characters, scene)

    assert witnessed_withholding(store, "tomas", seen) is False
    assert store.get_relationships("tomas")["maria"].trust == before


def test_an_ordinary_event_moves_nothing(seeded, store):
    world, characters, scene = seeded
    event = a_withholding(scene)
    event.metadata = {}
    event.content = "The soup needs salt."
    before = store.get_relationships("maria")["tomas"].trust

    seen = projection_of(characters["maria"], event, world, characters, scene)

    assert witnessed_withholding(store, "maria", seen) is False
    assert store.get_relationships("maria")["tomas"].trust == before


def test_someone_already_at_the_floor_cannot_be_made_to_distrust_harder(seeded, store):
    world, characters, scene = seeded
    store.set_trust("maria", "tomas", WITHHOLD_TRUST_FLOOR)
    seen = projection_of(characters["maria"], a_withholding(scene), world, characters, scene)

    assert witnessed_withholding(store, "maria", seen) is False
    assert store.get_relationships("maria")["tomas"].trust == WITHHOLD_TRUST_FLOOR


def test_distrust_raises_a_bid(scenario):
    """Trust that moves and is never read is trust that does not exist.
    Distrust is attention: you are quicker to speak into what someone you
    have stopped believing says."""
    world, characters, scene = scenario
    maria = characters["maria"]
    event = a_withholding(scene)
    event.metadata = {}

    neutral, _ = heuristic_bid(maria, event, "full", Relationship(trust=NEUTRAL_TRUST))
    wary, reason = heuristic_bid(maria, event, "full", Relationship(trust=0.0))

    assert wary > neutral
    assert wary - neutral == pytest.approx(NEUTRAL_TRUST * DISTRUST_ATTENTION)
    assert "trust" in reason


def test_trust_above_neutral_changes_nothing(scenario):
    """Faith in someone is not a reason to talk over them."""
    world, characters, scene = scenario
    event = a_withholding(scene)
    event.metadata = {}

    without, _ = heuristic_bid(characters["maria"], event, "full")
    devoted, _ = heuristic_bid(characters["maria"], event, "full", Relationship(trust=1.0))

    assert devoted == without


def test_trust_moves_over_a_played_scene(fake_llm):
    """End to end, through the ordinary turn loop and nothing else: Elena
    asks about the music box, Tomás visibly does not answer, and Maria —
    who is sitting right there — believes him a little less than she did."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    before = session.store.get_relationships("maria")["tomas"].trust

    session.say("What happened to Grandma's music box?")

    events = session.store.get_events(session.scene.id)
    assert any(e.metadata.get("withheld") and e.actor_id == "tomas" for e in events)
    assert session.store.get_relationships("maria")["tomas"].trust < before


def test_the_players_own_trust_is_never_moved_for_them(fake_llm):
    """Deciding that Elena believes her brother less tonight is telling
    the person holding her how they feel — the same overreach as
    narrating her actions for her."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    before = session.store.get_relationships("elena")["tomas"].trust

    session.say("What happened to Grandma's music box?")

    events = session.store.get_events(session.scene.id)
    withheld = next(e for e in events if e.metadata.get("withheld"))
    # She was standing right there and saw it in full; it still cost her
    # nothing, because how she feels is not the engine's to decide.
    assert withheld.location_id == session.here()
    assert session.store.get_relationships("elena")["tomas"].trust == before
    assert session.store.get_relationships("maria")["tomas"].trust < before


def test_one_withheld_beat_is_only_ever_absorbed_once(fake_llm):
    """Regression. `_absorb` looks at each character's *newest perceived*
    event, which stays the same one while they miss what happens
    elsewhere — so a single refusal was charging Maria again on every
    append she could not see, and her interaction count was inflated to
    match."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    store = session.store
    store.append_event(
        session.director.build_event(
            "action", "tomas", "kitchen", "Tomás says nothing.", metadata={"withheld": True}
        )
    )
    session.director._absorb()
    once = store.get_relationships("maria")["tomas"].trust

    for i in range(3):
        store.append_event(
            session.director.build_event("utterance", "elena", "garden", f"far away {i}")
        )
        session.director._absorb()

    assert store.get_relationships("maria")["tomas"].trust == once
    assert store.get_relationships("maria")["tomas"].interactions == 1
