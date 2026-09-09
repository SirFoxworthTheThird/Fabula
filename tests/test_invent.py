"""A world from a sentence.

Everything a story turns on is authored YAML, which is right for the
parts a story turns on and wrong as the only way to *start* one: the
audience for this app does not write YAML, and the whole app offered
seven openings somebody else had thought of.

What makes this something other than a wish is that the engine disposes.
Ids are made here, not taken; references are dropped unless they
resolve; exactly one character is the player whatever the model said;
and nothing reaches disk that `fabula.inspect` has not read.
"""
import json
import shutil
from pathlib import Path

import pytest

from fabula.inspect import complaints
from fabula.invent import CannotInvent, invent
from fabula.llm import FakeLLM
from fabula.loader import load_characters, load_pressures, load_scene, load_world
from fabula.models import Event
from fabula.pressures import _KNOWN_TRIGGER_KEYS, is_eligible, scene_state
from fabula.session import Session

from tests.conftest import ASHGROVE

WORLDS = ASHGROVE.parent

WORLD = {
    "title": "The Hollow Wing",
    "blurb": "A hotel kitchen at four in the morning, and a job already gone wrong.",
    "rooms": [
        {"name": "the kitchen", "description": "Steel benches, a fridge humming.",
         "adjacent": ["the service corridor"]},
        {"name": "the service corridor", "description": "Strip light and a laundry cart.",
         "adjacent": ["the kitchen", "the loading bay"]},
        {"name": "the loading bay", "description": "Cold air, the shutter half up.",
         "adjacent": ["the service corridor"]},
    ],
    "secrets": [{"name": "the second key", "keywords": ["the second key", "a copy of the key"]}],
    "characters": [
        {"name": "Dessa Vane", "persona": "Dessa organises things and is calm about it.",
         "room": "the kitchen", "protects": [], "talkativeness": 0.7, "reticence": 0.2},
        {"name": "Ruben Ott", "persona": "Ruben deflects with jokes and does not sit down.",
         "room": "the kitchen", "protects": ["the second key"],
         "talkativeness": 0.5, "reticence": 0.8},
        {"name": "Inês Cardoso", "persona": "Inês drove, and wants to leave.",
         "room": "the loading bay", "protects": []},
    ],
    "player": "Dessa Vane",
}

SCENE = {
    "title": "Four in the morning",
    "premise": "The van is gone and the shutter is still half up.",
    "opening": "You have been awake nineteen hours and the van is not where it was left.",
    "positions": {"Dessa Vane": "the kitchen", "Ruben Ott": "the kitchen",
                  "Inês Cardoso": "the loading bay"},
    "ends_when": "the second key",
}


EXTRAS = {
    "pressures": [
        {"what": "The shutter rattles once in the wind and settles again.",
         "where": "the loading bay", "after_turns": 5,
         "while_unsaid": "the second key", "how_hard": 0.45, "at_most": 2},
        {"what": "A pan is set down harder than it needed to be.",
         "where": "the kitchen", "after_turns": 9, "while_unsaid": None,
         "how_hard": 0.35, "at_most": 1},
    ],
    "intentions": [
        {"who": "Ruben Ott", "what": "checks his coat pocket without taking anything out",
         "where": "the kitchen", "after_minutes": 20, "alone": True},
    ],
}


class Scripted(FakeLLM):
    """A model that answers the two design questions, and whatever else
    the engine asks, with something readable."""

    def __init__(self, world=None, scene=None, repair="A room, and nothing in it to say.",
                 extras=None):
        super().__init__()
        self.world = WORLD if world is None else world
        self.scene = SCENE if scene is None else scene
        self.extras = EXTRAS if extras is None else extras
        self.repair = repair

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        if key == "invent:world":
            return "```json\n" + json.dumps(self.world) + "\n```"
        if key == "invent:scene":
            return json.dumps(self.scene)
        if key == "invent:complications":
            return json.dumps(self.extras)
        if key == "invent:repair":
            return self.repair
        return super().complete(system, prompt, key)


@pytest.fixture
def root(tmp_path):
    return tmp_path / "worlds"


def test_a_sentence_becomes_a_world_that_passes_every_rule(root):
    """The whole point: what it writes is the same YAML an author would
    have written, and it is held to the same rules."""
    made = invent("a heist that goes wrong in a hotel kitchen", Scripted(), worlds_root=root)

    assert complaints(made.world_dir) == []
    assert made.title == "The Hollow Wing"
    assert (made.world_dir / "world.yaml").is_file()
    assert len(list((made.world_dir / "characters").glob("*.yaml"))) == 3


def test_and_it_can_be_played(root):
    """Not a different kind of world. The engine opens it without knowing
    where it came from."""
    made = invent("a heist in a hotel kitchen", Scripted(), worlds_root=root)

    session = Session.open(made.world_dir, made.scene, llm=FakeLLM())
    perceived = session.perceived_so_far()

    assert session.user_character.name == "Dessa Vane"
    assert perceived[0].event.metadata.get("opening")
    assert session.present(), "somebody is in the room with them"
    session.close()


def test_ids_are_made_here_rather_than_taken(root):
    """A name is not an id, and a model that hands you one has handed you
    a filename it invented."""
    made = invent("anything", Scripted(), worlds_root=root)

    written = {p.stem for p in (made.world_dir / "characters").glob("*.yaml")}

    assert written == {"dessa_vane", "ruben_ott", "ines_cardoso"}, "folded, and no accents"


def test_exactly_one_player_whatever_the_model_said(root):
    """It named nobody, so the engine picks: whoever protects nothing,
    because the player is the one who does not know."""
    world = dict(WORLD, player="Somebody Who Is Not In It")

    made = invent("anything", Scripted(world=world), worlds_root=root)

    from fabula.loader import load_characters

    players = [c for c in load_characters(made.world_dir).values() if c.is_user]
    assert len(players) == 1
    assert not players[0].protects, "and they are keeping nothing from themselves"


def test_a_secret_nobody_was_given_is_given_to_somebody(root):
    """A secret nobody keeps never comes up, and the story's own subject
    quietly stops being in it."""
    world = dict(WORLD, characters=[dict(c, protects=[]) for c in WORLD["characters"]])

    made = invent("anything", Scripted(world=world), worlds_root=root)

    from fabula.loader import load_characters

    cast = load_characters(made.world_dir)
    keepers = [c for c in cast.values() if c.protects]
    assert len(keepers) == 1 and not keepers[0].is_user


def test_prose_that_names_the_secret_is_rewritten(root):
    """The rule a generator breaks constantly: a room description that
    says the thing, handing the secret to the scenery where it satisfies
    `fact_spoken` before anybody has spoken."""
    world = dict(WORLD, rooms=[
        dict(WORLD["rooms"][0], description="Steel benches, and the second key on a hook."),
        *WORLD["rooms"][1:],
    ])

    made = invent("anything", Scripted(world=world), worlds_root=root)

    from fabula.loader import load_world

    assert made.repaired, "it said so"
    assert complaints(made.world_dir) == []
    # The keywords themselves are the fact's definition and belong in the
    # file; what may not carry them is the prose.
    for room in load_world(made.world_dir).rooms.values():
        assert "second key" not in room.description.lower()


def test_prose_that_cannot_be_rewritten_is_dropped_rather_than_shipped(root):
    """Shipping it would break the story rather than the prose."""
    world = dict(WORLD, rooms=[
        dict(WORLD["rooms"][0], description="The second key is here."),
        *WORLD["rooms"][1:],
    ])

    made = invent(
        "anything",
        Scripted(world=world, repair="Still the second key, sorry."),
        worlds_root=root,
    )

    assert made.dropped
    assert complaints(made.world_dir) == []


def test_a_persona_may_name_the_secret_its_own_character_keeps(root):
    """Which is the difference between a character who knows something
    and one who has been handed somebody else's."""
    world = dict(WORLD, characters=[
        WORLD["characters"][0],
        dict(WORLD["characters"][1], persona="Ruben cut the second key last week."),
        WORLD["characters"][2],
    ])

    made = invent("anything", Scripted(world=world), worlds_root=root)

    kept = (made.world_dir / "characters" / "ruben_ott.yaml").read_text()
    assert "second key" in kept
    assert complaints(made.world_dir) == []


def test_a_room_nobody_could_walk_to_is_joined_up(root):
    """A place the story can never reach is scenery in the wrong file."""
    world = dict(WORLD, rooms=[
        dict(WORLD["rooms"][0], adjacent=[]),
        dict(WORLD["rooms"][1], adjacent=[]),
        dict(WORLD["rooms"][2], adjacent=[]),
    ])

    made = invent("anything", Scripted(world=world), worlds_root=root)

    from fabula.loader import load_world

    built = load_world(made.world_dir)
    for room_id in built.rooms:
        assert any(
            room_id in other.adjacent or other.id in built.rooms[room_id].adjacent
            for other in built.rooms.values()
            if other.id != room_id
        ), room_id


def test_somebody_is_in_the_room_with_you(root):
    """A first scene that opens on an empty room is the app asking the
    player to entertain themselves."""
    scene = dict(SCENE, positions={
        "Dessa Vane": "the kitchen",
        "Ruben Ott": "the loading bay",
        "Inês Cardoso": "the loading bay",
    })

    made = invent("anything", Scripted(scene=scene), worlds_root=root)

    session = Session.open(made.world_dir, made.scene, llm=FakeLLM())
    assert session.present()
    session.close()


def test_a_world_that_cannot_be_made_playable_is_not_left_on_the_shelf(root):
    """Half of what makes a world playable cannot be noticed by reading
    it, so what fails is deleted rather than offered."""
    world = dict(WORLD, rooms=[WORLD["rooms"][0]])  # one room, nowhere to go

    with pytest.raises(CannotInvent):
        invent("anything", Scripted(world=world), worlds_root=root)

    assert not root.exists() or list(root.iterdir()) == []


def test_a_model_that_answers_with_nothing_readable_is_refused(root):
    class Rambles(FakeLLM):
        def complete(self, system, prompt, key=None):
            return "Certainly! Here is a wonderful world for your story."

    with pytest.raises(CannotInvent):
        invent("anything", Rambles(), worlds_root=root)


def test_an_empty_premise_is_refused_before_anything_is_asked(root):
    llm = Scripted()

    with pytest.raises(CannotInvent):
        invent("   ", llm, worlds_root=root)

    assert llm.calls == [], "and nobody's money was spent on it"


def test_it_lands_beside_the_shipped_worlds(tmp_path):
    """Not a different kind of world: the shelf lists it without knowing
    where it came from."""
    from fabula.loader import catalogue

    root = tmp_path / "worlds"
    shutil.copytree(WORLDS, root)

    made = invent("a heist in a hotel kitchen", Scripted(), worlds_root=root)

    shelf = {world["id"]: world for world in catalogue(root)}
    assert made.world_dir.name in shelf
    opening = [s for s in shelf[made.world_dir.name]["scenes"] if s["opens"]]
    assert len(opening) == 1 and opening[0]["you"] == "Dessa Vane"


def test_a_keyword_that_is_the_furniture_is_not_kept(root):
    """The generator must not hand the validator something it will
    refuse: a 1.5B made "stairs" and "library" the keywords of a secret,
    in a world with a library in it."""
    world = dict(
        WORLD,
        rooms=[
            {"name": "the library", "description": "Bookshelves, and a table.",
             "adjacent": ["the kitchen"]},
            {"name": "the kitchen", "description": "Steel benches.", "adjacent": ["the library"]},
        ],
        secrets=[
            {"name": "the second key", "keywords": ["library", "the second key"]},
        ],
        characters=[
            {"name": "Dessa Vane", "persona": "Dessa organises things.",
             "room": "the kitchen", "protects": []},
            {"name": "Ruben Ott", "persona": "Ruben deflects.",
             "room": "the kitchen", "protects": ["the second key"]},
        ],
    )

    made = invent("anything", Scripted(world=world), worlds_root=root)

    from fabula.loader import load_world

    kept = load_world(made.world_dir).facts["the_second_key"].keywords
    assert kept == ["the second key"]
    assert complaints(made.world_dir) == []


def test_a_secret_with_nothing_left_to_recognise_it_is_dropped(root):
    """A fact nobody can say is a fact nothing can turn on."""
    world = dict(
        WORLD,
        secrets=[{"name": "kitchen", "keywords": ["the kitchen", "kitchen"]}],
        characters=[dict(c, protects=[]) for c in WORLD["characters"]],
    )

    made = invent("anything", Scripted(world=world), worlds_root=root)

    from fabula.loader import load_world

    assert load_world(made.world_dir).facts == {}
    assert complaints(made.world_dir) == []


def test_the_example_handed_back_is_not_a_world(root):
    """An example is the only thing that reliably fixes the shape — hand
    a model "title": "two or three words" and it writes a scene called
    "two or three words" — and a small one then hands the example back
    rather than designing anything. Measured on a 1.5B: asked for a heist
    in a hotel kitchen, it returned Ashgrove, Elena and the music box."""
    example = {
        "title": "Ashgrove",
        "blurb": "A house, a family, and something one of them has not said out loud.",
        "rooms": [
            {"name": "the kitchen", "description": "Warm.", "adjacent": ["the study"]},
            {"name": "the study", "description": "Colder.", "adjacent": ["the kitchen"]},
        ],
        "secrets": [{"name": "the broken music box", "keywords": ["music box"]}],
        "characters": [
            {"name": "Elena Rey", "persona": "The youngest.", "room": "the kitchen",
             "protects": []},
            {"name": "Tomas Rey", "persona": "Her brother.", "room": "the kitchen",
             "protects": ["the broken music box"]},
        ],
        "player": "Elena Rey",
    }

    with pytest.raises(CannotInvent, match="example"):
        invent("a heist in a hotel kitchen", Scripted(world=example), worlds_root=root)


def test_a_second_answer_is_taken_when_the_first_was_the_example(root):
    """It is asked again, told plainly that the example is a shape rather
    than an answer, before anybody gives up."""
    example = {
        "title": "Ashgrove",
        "blurb": "A house.",
        "rooms": [{"name": "the kitchen", "description": "Warm.", "adjacent": ["the study"]},
                  {"name": "the study", "description": "Cold.", "adjacent": ["the kitchen"]}],
        "secrets": [{"name": "the broken music box", "keywords": ["music box"]}],
        "characters": [
            {"name": "Elena Rey", "persona": "The youngest.", "room": "the kitchen", "protects": []},
            {"name": "Tomas Rey", "persona": "Her brother.", "room": "the kitchen",
             "protects": ["the broken music box"]},
        ],
        "player": "Elena Rey",
    }

    class ParrotsOnce(Scripted):
        def __init__(self):
            super().__init__()
            self.asked = 0

        def complete(self, system, prompt, key=None):
            if key == "invent:world":
                self.asked += 1
                if self.asked == 1:
                    return json.dumps(example)
            return super().complete(system, prompt, key)

    made = invent("a heist in a hotel kitchen", ParrotsOnce(), worlds_root=root)

    assert made.title == "The Hollow Wing"
    assert complaints(made.world_dir) == []


def test_a_name_that_leaked_out_of_the_example_is_left_out(root):
    """The wholesale copy is caught before anything is built; this is the
    one that leaks through. Measured on a 3B, which designed a hotel
    heist and then put Elena Rey in it and called the scene The dinner."""
    world = dict(WORLD, characters=[
        *WORLD["characters"],
        {"name": "Elena Rey", "persona": "The youngest.", "room": "the kitchen",
         "protects": []},
    ])

    made = invent("a heist", Scripted(world=world, scene=dict(SCENE, title="The dinner")),
                  worlds_root=root)

    written = {p.stem for p in (made.world_dir / "characters").glob("*.yaml")}
    assert "elena_rey" not in written
    assert made.scene != "the_dinner"
    assert complaints(made.world_dir) == []


# --- What makes a scene escalate rather than converse ------------------
#
# A generated world with nobody but its cast in it is a conversation.
# Pressures are the room having its own opinion about how long this can
# go on; intentions are what somebody does while nobody is watching.
# Both are asked for in plain words, and both are built here — the model
# never sees the trigger vocabulary, because an unrecognised key makes a
# pressure always eligible.


def test_a_generated_world_has_something_that_happens_on_its_own(root):
    made = invent("a heist in a hotel kitchen", Scripted(), worlds_root=root)

    pressures = load_pressures(made.world_dir)
    characters = load_characters(made.world_dir)

    assert pressures, "the room has its own opinion about how long this goes on"
    assert any(c.intentions for c in characters.values()), "somebody means to do something"
    assert complaints(made.world_dir) == []


def test_a_pressure_from_a_generator_only_ever_narrates(root):
    """An arrival needs somebody the scene said might turn up, and a state
    change needs a scene written around it. Either from a generator is a
    pressure firing into a story nobody wrote."""
    extras = {
        "pressures": [
            {"what": "The shutter rattles.", "where": "the loading bay",
             "after_turns": 4, "kind": "arrival", "who": "Inês Cardoso"},
        ],
        "intentions": [],
    }

    made = invent("a heist", Scripted(extras=extras), worlds_root=root)

    for pressure in load_pressures(made.world_dir):
        assert pressure.effect["kind"] == "narration"
        assert "actor" not in pressure.effect


def test_a_pressure_triggers_only_on_words_the_engine_knows(root):
    """The vocabulary never leaves `invent`. A key the evaluator does not
    recognise raises rather than passing, so a made-up one would be a
    world that cannot be played at all."""
    made = invent("a heist", Scripted(), worlds_root=root)
    world = load_world(made.world_dir)
    # A scene nobody has said anything in yet.
    state = scene_state([], load_characters(made.world_dir), world)

    for pressure in load_pressures(made.world_dir):
        assert set(pressure.trigger) <= _KNOWN_TRIGGER_KEYS
        # Every fact it waits on is one this world actually has — an id
        # that does not resolve makes the trigger unsatisfiable rather
        # than raising, which is a pressure nobody can tell is broken.
        waits_on = pressure.trigger.get("fact_unspoken")
        assert waits_on is None or waits_on in world.facts
        # And it answers rather than raising, which an unrecognised key
        # would not.
        assert is_eligible(pressure, state, world.facts) is False, "not on turn zero"


def test_a_pressure_waits_on_the_secret_the_model_named(root):
    """It names the secret in the words it used two calls ago, not the id
    this engine made out of it."""
    made = invent("a heist", Scripted(), worlds_root=root)

    triggers = [p.trigger for p in load_pressures(made.world_dir)]

    assert {"fact_unspoken": "the_second_key"}.items() <= triggers[0].items()
    assert "fact_unspoken" not in triggers[1], "the one that said null waits on nothing"


def test_a_secret_the_model_misremembers_is_not_waited_on(root):
    """A trigger naming a fact that does not exist can never fire, which
    reads exactly like a pressure nobody wrote."""
    extras = {
        "pressures": [{"what": "The fridge cycles off.", "where": "the kitchen",
                       "after_turns": 5, "while_unsaid": "the missing ledger"}],
        "intentions": [],
    }

    made = invent("a heist", Scripted(extras=extras), worlds_root=root)

    trigger = load_pressures(made.world_dir)[0].trigger
    assert "fact_unspoken" not in trigger
    assert complaints(made.world_dir) == []


def test_a_pressure_in_a_room_that_does_not_exist_lands_somewhere_real(root):
    extras = {
        "pressures": [{"what": "Rain starts.", "where": "the roof garden", "after_turns": 6}],
        "intentions": [],
    }

    made = invent("a heist", Scripted(extras=extras), worlds_root=root)
    world = load_world(made.world_dir)

    assert load_pressures(made.world_dir)[0].effect["location"] in world.rooms


def test_the_numbers_are_clamped_rather_than_taken(root):
    """A pressure that fires on turn one, forever, is worse than one that
    never fires at all."""
    extras = {
        "pressures": [{"what": "A door closes.", "where": "the kitchen",
                       "after_turns": 0, "how_hard": 9.0, "at_most": 40}],
        "intentions": [{"who": "Ruben Ott", "what": "counts the coats", "after_minutes": 0}],
    }

    made = invent("a heist", Scripted(extras=extras), worlds_root=root)

    pressure = load_pressures(made.world_dir)[0]
    assert pressure.trigger["turns_elapsed"] == "> 2"
    assert pressure.weight <= 0.9
    assert pressure.max_fires <= 3
    assert pressure.cooldown_turns >= 4

    ruben = load_characters(made.world_dir)["ruben_ott"]
    assert ruben.intentions[0].ready_after_minutes >= 5


def test_the_player_is_never_given_an_intention(root):
    """An intention is what happens while nobody is watching, and the
    player is played by somebody who is here."""
    extras = {
        "pressures": [],
        "intentions": [
            {"who": "Dessa Vane", "what": "goes back for the van", "after_minutes": 20},
            {"who": "Ruben Ott", "what": "checks his coat pocket", "after_minutes": 30},
        ],
    }

    made = invent("a heist", Scripted(extras=extras), worlds_root=root)
    characters = load_characters(made.world_dir)

    assert characters["dessa_vane"].is_user
    assert characters["dessa_vane"].intentions == []
    assert [i.description for i in characters["ruben_ott"].intentions] == [
        "checks his coat pocket"
    ]


def test_an_intention_nobody_in_the_cast_owns_is_dropped(root):
    extras = {
        "pressures": [],
        "intentions": [{"who": "The night porter", "what": "locks the bay", "after_minutes": 20}],
    }

    made = invent("a heist", Scripted(extras=extras), worlds_root=root)

    assert all(not c.intentions for c in load_characters(made.world_dir).values())
    assert complaints(made.world_dir) == []


def test_nobody_is_put_to_sleep_by_a_generator(root):
    """Somebody put under with nothing to wake them stays under for the
    rest of the scene, perceiving nothing — which `inspect` refuses, and
    which is not a shape to arrive at by accident."""
    extras = {
        "pressures": [],
        "intentions": [{"who": "Ruben Ott", "what": "turns in", "after_minutes": 40,
                        "state": "asleep"}],
    }

    made = invent("a heist", Scripted(extras=extras), worlds_root=root)

    ruben = load_characters(made.world_dir)["ruben_ott"]
    assert [i.state for i in ruben.intentions] == [None]
    assert complaints(made.world_dir) == []


def test_a_complication_that_names_the_secret_is_rewritten(root):
    """A pressure intent reaches the log as narration, so a keyword in one
    lets the room raise the subject before anybody has said it."""
    extras = {
        "pressures": [{"what": "Somebody has left the second key on the bench.",
                       "where": "the kitchen", "after_turns": 5}],
        "intentions": [{"who": "Ruben Ott", "what": "moves a copy of the key to his coat",
                        "after_minutes": 20}],
    }

    made = invent("a heist", Scripted(extras=extras, repair="Something is out of place."),
                  worlds_root=root)

    assert "the second key" not in load_pressures(made.world_dir)[0].intent
    ruben = load_characters(made.world_dir)["ruben_ott"]
    assert "key" not in ruben.intentions[0].description
    assert len(made.repaired) >= 2
    assert complaints(made.world_dir) == []


def test_a_complication_that_will_not_stop_naming_it_is_dropped(root):
    extras = {
        "pressures": [{"what": "The second key is on the bench.", "where": "the kitchen",
                       "after_turns": 5}],
        "intentions": [{"who": "Ruben Ott", "what": "pockets the second key",
                        "after_minutes": 20}],
    }

    made = invent(
        "a heist",
        # A model that rewrites prose into the same problem, twice.
        Scripted(extras=extras, repair="He still has the second key."),
        worlds_root=root,
    )

    assert load_pressures(made.world_dir) == []
    assert all(not c.intentions for c in load_characters(made.world_dir).values())
    assert len(made.dropped) >= 2
    assert complaints(made.world_dir) == []


def test_a_world_whose_complications_failed_is_quieter_not_broken(root):
    """Losing the whole world over the part that escalates it would be
    the wrong trade: the scene still runs on the people in it."""

    class NoComplications(Scripted):
        def complete(self, system, prompt, key=None):
            if key == "invent:complications":
                return "I'm sorry, I can't help with that."
            return super().complete(system, prompt, key)

    made = invent("a heist", NoComplications(), worlds_root=root)

    assert load_pressures(made.world_dir) == []
    assert any("nothing happens on its own" in note for note in made.dropped)
    assert complaints(made.world_dir) == []

    session = Session.open(made.world_dir, made.scene, llm=FakeLLM())
    assert session.present()
    session.close()


def test_a_generated_pressure_becomes_eligible_once_the_scene_has_run(root):
    """The failure worth catching is a pressure that can never fire: it
    reads, in play, exactly like a world that has none. So it is not
    enough that the trigger parses — it has to turn true."""
    made = invent("a heist", Scripted(), worlds_root=root)
    world = load_world(made.world_dir)
    characters = load_characters(made.world_dir)
    scene = load_scene(made.world_dir, made.scene)
    first = load_pressures(made.world_dir)[0]

    def after(turns: int, said: str = "Nothing much.") -> object:
        # One line a turn, in the room the pressure is about, so the
        # fact-unspoken half of the trigger is exercised too.
        return scene_state(
            [
                Event(
                    id=seq, scene_id=scene.id, seq=seq, story_time=scene.start_time,
                    kind="utterance", actor_id="ruben_ott",
                    location_id=first.effect["location"], content=said, audibility="room",
                )
                for seq in range(1, turns + 1)
            ],
            characters,
            world,
        )

    assert not is_eligible(first, after(2), world.facts), "not while the scene is young"
    assert is_eligible(first, after(9), world.facts), "and then the room has an opinion"
    # It waits on the secret, so somebody saying it out loud stops it.
    assert not is_eligible(first, after(9, "You made a copy of the key."), world.facts)
