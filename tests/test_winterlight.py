"""A second world, and what building one found.

Everything in this project was tested against ashgrove: a warm house,
two mutually audible rooms, one secret, one person holding it. That is a
shape, and code fitted to one shape looks general until a second one
arrives. `winterlight` is a cold building on a plateau — a chain of four
rooms, one of them one-way, and two secrets held by two different people
from each other as much as from anyone.

The generic guards at the bottom of this file apply to every world in
the repo, present and future. One of them is the test that would have
caught the bug where Maria's persona named the music box while she said
she did not know about it.
"""
from pathlib import Path

import pytest

from fabula.llm import FakeLLM
from fabula.loader import load_characters, load_scenario, load_scene, load_world
from fabula.memory import project
from fabula.session import Session
from fabula.world import mentions_fact

WORLDS = Path(__file__).parent.parent / "worlds"
WINTERLIGHT = WORLDS / "winterlight"

FLIGHT = "the first flight has slipped to December"
FUEL = "I bled six hundred litres out of the reserve tank in June"


@pytest.fixture
def station():
    return load_scenario(WINTERLIGHT, "the_long_dark")


def seen_by(character, events, world):
    return project(character, events, world, initial_location=character.location_id)


def test_the_station_loads_with_two_holders_and_a_player(station):
    world, characters, scene = station

    holders = {c.id: c.protects for c in characters.values() if c.protects}
    assert holders == {"ilse": ["the_flight"], "yusuf": ["the_fuel"]}
    assert [c.id for c in characters.values() if c.is_user] == ["ana"]
    assert len(scene.cast) == 4  # one more agent than ashgrove has


def test_a_room_you_can_listen_into_but_not_out_of(station):
    """Edges are directed. The mess lists the generator shed, so from the
    mess you hear it; the shed lists nothing, because inside it the
    engine is all there is to hear. Nothing in the first world ever asked
    the perception model to do this."""
    world, _, _ = station

    assert world.distance("mess", "generator") == 1
    assert world.distance("generator", "mess") == -1
    assert world.distance("generator", "radio") == -1


def test_the_chain_puts_two_rooms_between_people(station):
    """Ashgrove has two rooms, so every pair is adjacent and a distance
    of two was never exercised."""
    world, _, _ = station

    assert world.distance("bunkroom", "radio") == 2
    assert world.distance("mess", "radio") == 1


def test_what_reaches_whom_across_the_chain(station):
    """The whole topology in one assertion. Ana in the mess is the hub;
    Yusuf in the shed hears nothing from anywhere, a shout included."""
    world, characters, scene = station
    session = Session.open(WINTERLIGHT, "the_long_dark", llm=FakeLLM())
    build = session.director.build_event

    events = [
        session.store.append_event(
            build("utterance", "ana", "mess", "Has anyone seen Ilse?", audibility="building")
        ),
        session.store.append_event(
            build("action", "yusuf", "generator", "drops a spanner", audibility="adjacent")
        ),
        session.store.append_event(
            build("utterance", "ilse", "radio", "Winterlight, go ahead.", audibility="adjacent")
        ),
    ]
    log = session.store.get_events(session.scene.id)

    # Numbered by the three events this test made, not by their place in
    # the log — the scene opens with a line of its own before them.
    said = {event.seq: index for index, event in enumerate(events, start=1)}

    def perceived(character_id):
        character = session.characters[character_id]
        return {
            said[p.event.seq]: p.perception
            for p in seen_by(character, log, session.world)
            if p.event.seq in said
        }

    # from the mess (building) | from the shed (adjacent) | from the radio (adjacent)
    assert perceived("ana") == {1: "full", 2: "degraded", 3: "degraded"}
    assert perceived("ilse") == {1: "degraded", 3: "full"}
    assert perceived("nadia") == {1: "degraded"}          # two rooms from the radio: nothing
    assert perceived("yusuf") == {2: "full"}              # sealed in with the engine


def test_neither_holder_learns_the_others_secret(station):
    """The structural point of a second world. Ashgrove's asymmetry runs
    one way; here two people are each keeping something, and the engine
    has to hold both directions at once."""
    session = Session.open(WINTERLIGHT, "the_long_dark", llm=FakeLLM())
    build = session.director.build_event
    session.store.append_event(build("utterance", "ilse", "radio", FLIGHT))
    session.store.append_event(build("utterance", "yusuf", "generator", FUEL))
    log = session.store.get_events(session.scene.id)

    def context(character_id):
        return session.director.contexts.for_character(
            session.characters[character_id], log
        ).lower()

    assert "reserve tank" not in context("ilse")      # she never hears his
    assert "slipped to december" not in context("yusuf")  # he never hears hers
    for outsider in ("ana", "nadia"):
        assert "reserve tank" not in context(outsider)
        assert "slipped to december" not in context(outsider)


def test_the_shed_is_a_room_nothing_enters_not_one_nothing_leaves(station):
    """Worth being exact about, because the obvious reading is wrong.
    The one-way edge means Yusuf hears nothing from anywhere — but the
    mess still hears *him*, because the mess is the end that has the
    edge. He can be overheard and cannot overhear, which is the better
    version dramatically: Ana can half-hear him working and he has no
    idea she is there.
    """
    session = Session.open(WINTERLIGHT, "the_long_dark", llm=FakeLLM())
    build = session.director.build_event
    carrying = session.store.append_event(
        build("action", "yusuf", "generator", "squares up the drums", audibility="building")
    )
    shout = session.store.append_event(
        build("utterance", "ana", "mess", "Yusuf? Are you through there?", audibility="building")
    )
    log = session.store.get_events(session.scene.id)

    def perceived(character_id):
        return {
            p.event.seq: p.perception
            for p in seen_by(session.characters[character_id], log, session.world)
        }

    assert perceived("ana")[carrying.seq] == "degraded"   # she hears him working
    assert shout.seq not in perceived("yusuf")            # he does not hear her at all
    assert perceived("yusuf") == {carrying.seq: "full"}


def test_a_quiet_act_in_the_shed_reaches_nobody(station):
    """And at ordinary room audibility it does not carry at all — which
    is what makes the shed the place to do something you would rather
    nobody saw."""
    session = Session.open(WINTERLIGHT, "the_long_dark", llm=FakeLLM())
    quiet = session.store.append_event(
        session.director.build_event(
            "action", "yusuf", "generator",
            "squares up the drums so the gap does not read from the doorway",
        )
    )
    log = session.store.get_events(session.scene.id)

    def saw_it(character_id):
        return [
            p for p in seen_by(session.characters[character_id], log, session.world)
            if p.event.seq == quiet.seq
        ]

    for other in ("ana", "ilse", "nadia"):
        assert saw_it(other) == []
    assert len(seen_by(session.characters["yusuf"], log, session.world)) == 1


def test_the_arc_is_over_only_when_both_things_are_in_the_room():
    """Two secrets, so the scene is not over while either is still held.
    The first world never needed to say this, and the trigger vocabulary
    could not express it until it did."""
    session = Session.open(WINTERLIGHT, "the_manifest", llm=FakeLLM())
    build = session.director.build_event

    assert session.scene.end_condition == {"fact_spoken": ["the_flight", "the_fuel"]}
    assert session.ended() is False

    session.store.append_event(build("utterance", "ilse", "mess", FLIGHT))
    assert session.ended() is False  # one of two is not an ending

    session.store.append_event(build("utterance", "yusuf", "mess", FUEL))
    assert session.ended() is True


def test_the_arc_seats_everyone_at_one_table():
    world, characters, scene = load_scenario(WINTERLIGHT, "the_manifest")

    assert set(scene.starting_positions.values()) == {"mess"}
    assert set(scene.cast) == set(scene.starting_positions)


def test_a_scene_can_start_with_everyone_out_of_earshot():
    """Ashgrove's opener seats the player beside the person with the
    secret. This one starts her alone in the middle of a chain."""
    world, characters, scene = load_scenario(WINTERLIGHT, "the_long_dark")

    positions = {cid: characters[cid].location_id for cid in scene.cast}
    assert len(set(positions.values())) == 4  # nobody starts with anybody


# --- Guards that hold for every world in the repo, not just this one ---

ALL_WORLDS = sorted(p.parent for p in WORLDS.glob("*/world.yaml"))


def _named_facts(condition: dict) -> list[str]:
    named = []
    for key in ("fact_spoken", "fact_unspoken"):
        value = condition.get(key)
        named += [value] if isinstance(value, str) else list(value or [])
    return named


@pytest.mark.parametrize("world_dir", ALL_WORLDS, ids=lambda p: p.name)
def test_no_persona_names_a_secret_that_is_not_its_own(world_dir):
    """The bug this is here for: Maria's persona once named the music box
    in the same breath as saying she did not know about it, which handed
    her agent the secret in every prompt it ever saw. Authored prose is
    the one place a leak can be written by hand, and the projection
    cannot catch it because it never passes through one.
    """
    world = load_world(world_dir)

    for character in load_characters(world_dir).values():
        named = [f for f in world.facts if mentions_fact(world.facts[f], character.persona)]
        assert set(named) <= set(character.protects), (
            f"{character.id}'s persona names {sorted(set(named) - set(character.protects))}"
        )


@pytest.mark.parametrize("world_dir", ALL_WORLDS, ids=lambda p: p.name)
def test_no_authored_text_names_a_fact(world_dir):
    """Four kinds of authored prose end up in the log as event content:
    a room description the narrator is given to describe a place, a
    pressure's intent that it is given to render, an intention's
    description, which is materialised as the action itself, and a
    scene's opening, which is appended as the first thing the player
    reads.

    The opening is the sharpest of the four. It is perceived in full by
    the player and by nobody else, so a keyword in it satisfies
    `fact_spoken` — the subject was raised in front of somebody — before
    the story has a first line. An arc could end on its own opening
    paragraph.

    A fact keyword in any of them lets the subject be raised by scenery.
    Keyword matching is deterministic and cannot tell a confession from a
    neutral mention, so the cost lands on the author: authored text must
    evoke a fact without naming it.

    The first version of this test checked only room descriptions. A
    played scene then ended its own arc, because a pressure intent read
    "counted down the days to the first flight" and the narration it
    produced satisfied the scene's `fact_spoken` condition — nobody had
    said anything.
    """
    from fabula.loader import load_pressures

    world = load_world(world_dir)

    def names(text):
        return [f for f in world.facts if mentions_fact(world.facts[f], text)]

    for room_id, room in world.rooms.items():
        assert not names(room.description), f"room {room_id}: {names(room.description)}"

    for pressure in load_pressures(world_dir):
        assert not names(pressure.intent), f"pressure {pressure.id}: {names(pressure.intent)}"

    for character in load_characters(world_dir).values():
        for intention in character.intentions:
            named = names(intention.description)
            assert not named, f"intention {character.id}/{intention.id}: {named}"

    for scene_file in sorted((world_dir / "scenes").glob("*.yaml")):
        scene = load_scene(world_dir, scene_file.stem)
        assert not names(scene.opening), f"opening {scene.id}: {names(scene.opening)}"


@pytest.mark.parametrize("world_dir", ALL_WORLDS, ids=lambda p: p.name)
def test_every_world_has_exactly_one_player_per_scene(world_dir):
    characters = load_characters(world_dir)

    for scene_file in sorted((world_dir / "scenes").glob("*.yaml")):
        _, cast, scene = load_scenario(world_dir, scene_file.stem)
        players = [cid for cid in scene.cast if cast[cid].is_user]
        assert players == [p for p in players], scene.id
        assert len(players) == 1, f"{scene.id} has players {players}"


@pytest.mark.parametrize("world_dir", ALL_WORLDS, ids=lambda p: p.name)
def test_every_authored_reference_resolves(world_dir):
    """A typo'd room or fact id fails silently otherwise: a pressure that
    never fires, an intention nobody can reach, a scene that never ends.
    """
    from fabula.loader import load_pressures

    world = load_world(world_dir)
    characters = load_characters(world_dir)

    for character in characters.values():
        assert character.location_id in world.rooms, character.id
        for fact_id in character.protects:
            assert fact_id in world.facts, f"{character.id} protects {fact_id}"
        for intention in character.intentions:
            assert intention.location_id in world.rooms, intention.id
        # Somebody put under with nothing to wake them stays under for the
        # rest of the scene, perceiving nothing — almost never what an
        # author meant, and silent when it is wrong.
        sleeps = [i for i in character.intentions if i.state == "asleep"]
        wakes = [i for i in character.intentions if i.state == "awake"]
        for turning_in in sleeps:
            assert any(w.ready_after_minutes > turning_in.ready_after_minutes for w in wakes), (
                f"{character.id} falls asleep at {turning_in.ready_after_minutes}m "
                "and nothing wakes them"
            )
        for goal in character.goals:
            # A goal naming a fact that does not exist stays open forever
            # and raises nobody's bid, silently.
            assert goal.about is None or goal.about in world.facts, (
                f"{character.id}'s goal {goal.id} is about {goal.about}"
            )
        for toward in character.relationships:
            assert toward in characters, f"{character.id} -> {toward}"

    for pressure in load_pressures(world_dir):
        effect = pressure.effect
        if effect.get("location"):
            assert effect["location"] in world.rooms, pressure.id
        if effect.get("actor"):
            assert effect["actor"] in characters, pressure.id
        for key in ("fact_spoken", "fact_unspoken"):
            named = pressure.trigger.get(key)
            for fact_id in [named] if isinstance(named, str) else (named or []):
                assert fact_id in world.facts, f"{pressure.id} -> {fact_id}"
        for key in ("character_at", "character_not_at"):
            for cid, room in (pressure.trigger.get(key) or {}).items():
                assert cid in characters and room in world.rooms, pressure.id

    for scene_file in sorted((world_dir / "scenes").glob("*.yaml")):
        _, _, scene = load_scenario(world_dir, scene_file.stem)
        for cid in scene.cast:
            assert cid in characters, f"{scene.id} casts {cid}"
        for cid in scene.may_arrive:
            assert cid in characters, f"{scene.id} awaits {cid}"
        assert not set(scene.cast) & set(scene.may_arrive), scene.id
        # An arrival pressure naming somebody outside the room can only
        # fire if the scene said they might turn up; otherwise it appends
        # an event with an actor nobody in the scene has ever heard of.
        for pressure in load_pressures(world_dir):
            actor = pressure.effect.get("actor")
            if pressure.effect.get("kind") == "arrival" and actor not in scene.cast:
                assert actor in scene.may_arrive, (
                    f"{scene.id}: {pressure.id} lands {actor}, who is neither cast nor awaited"
                )
        for cid, room in scene.starting_positions.items():
            assert cid in characters and room in world.rooms, scene.id
        named = scene.end_condition.get("fact_spoken")
        for fact_id in [named] if isinstance(named, str) else (named or []):
            assert fact_id in world.facts, f"{scene.id} ends on {fact_id}"
        # A story that leads somewhere that does not exist stops dead at
        # the seam, and nothing says so until a player gets there.
        scenes = {p.stem for p in (world_dir / "scenes").glob("*.yaml")}
        for successor in scene.next:
            following = successor.get("scene")
            assert following in scenes, f"{scene.id} leads to {following}, which is not a scene"
            assert following != scene.id or successor.get("when"), (
                f"{scene.id} leads to itself unconditionally"
            )
            for fact_id in _named_facts(successor.get("when") or {}):
                assert fact_id in world.facts, f"{scene.id} branches on {fact_id}"


def test_only_the_cast_is_in_the_scene(tmp_path):
    """`cast` was decorative. Every character in the world was handed to
    the director, so one authored for a single scene took turns in every
    scene the world had — bidding, forming beliefs, showing up in
    `present()`. A list you write and the engine ignores is worse than no
    list at all.
    """
    import shutil

    import yaml

    world_dir = tmp_path / "winterlight"
    shutil.copytree(WINTERLIGHT, world_dir)
    scene_file = world_dir / "scenes" / "the_manifest.yaml"
    scene = yaml.safe_load(scene_file.read_text(encoding="utf-8"))
    scene["cast"] = ["ilse", "ana"]
    scene["starting_positions"] = {"ilse": "mess", "ana": "mess"}
    scene_file.write_text(yaml.safe_dump(scene), encoding="utf-8")

    session = Session.open(world_dir, "the_manifest", llm=FakeLLM())
    session.say("Is anyone else here?")

    assert sorted(session.characters) == ["ana", "ilse"]
    acted = {e.actor_id for e in session.store.get_events(session.scene.id) if e.actor_id}
    assert acted <= {"ana", "ilse"}
    assert [c.id for c in session.present()] == ["ilse"]


def test_a_scene_that_casts_a_stranger_is_refused(tmp_path):
    """Silently dropping an unknown id would make a typo look like a
    character who simply never speaks."""
    import shutil

    import yaml

    world_dir = tmp_path / "winterlight"
    shutil.copytree(WINTERLIGHT, world_dir)
    scene_file = world_dir / "scenes" / "the_manifest.yaml"
    scene = yaml.safe_load(scene_file.read_text(encoding="utf-8"))
    scene["cast"] = ["ilse", "ana", "yusuff"]  # typo
    scene_file.write_text(yaml.safe_dump(scene), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown character"):
        Session.open(world_dir, "the_manifest", llm=FakeLLM())
