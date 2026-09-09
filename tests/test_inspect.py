"""Proving each rule catches the thing it is for.

A validator that never says no is worse than none: it is a promise
nobody checked. So every rule in `fabula.inspect` gets a world with
exactly that fault in it, built by taking a shipped world and breaking
one thing.

The faults are not hypothetical. Most of them are bugs this project
actually shipped and then found by playing: a persona that named a
secret and handed it to the agent in every prompt; a pressure intent
that ended an arc because the narration it produced satisfied
`fact_spoken` when nobody had spoken; a scene that led somewhere that
did not exist and stopped dead at the seam.
"""
import shutil
from pathlib import Path

import pytest
import yaml

from fabula.inspect import complaints, playable

from tests.conftest import ASHGROVE

WORLDS = ASHGROVE.parent


@pytest.fixture
def world(tmp_path):
    """A copy of ashgrove to break in one specific way."""
    copy = tmp_path / "ashgrove"
    shutil.copytree(ASHGROVE, copy)
    return copy


def edit(path: Path, change):
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(written)
    path.write_text(yaml.safe_dump(written, allow_unicode=True), encoding="utf-8")


def test_a_shipped_world_is_playable(world):
    assert playable(world)


# --- Prose that reaches the log ---------------------------------------


def test_a_room_that_names_a_fact(world):
    edit(world / "world.yaml", lambda w: w["rooms"]["kitchen"].update(
        {"description": "A kitchen, and the music box still on the mantel."}
    ))

    assert any("music_box" in c and "kitchen" in c for c in complaints(world))


def test_a_persona_that_names_somebody_else_s_secret(world):
    """The bug this is here for: Maria's persona once named the music box
    in the same breath as saying she did not know about it, which handed
    her agent the secret in every prompt it ever saw."""
    edit(world / "characters" / "maria.yaml", lambda c: c.update(
        {"persona": "Maria has no idea about the music box, and would be furious."}
    ))

    assert any("maria" in c and "not theirs" in c for c in complaints(world))


def test_but_a_persona_may_name_the_secret_it_is_keeping(world):
    """Which is the difference between a character who knows a secret and
    a character who has been handed one."""
    edit(world / "characters" / "tomas.yaml", lambda c: c.update(
        {"persona": "Tomás broke the music box in March and has said nothing since."}
    ))

    assert playable(world)


def test_a_pressure_intent_that_names_a_fact(world):
    """A played scene ended its own arc this way: an intent read "counted
    down the days to the first flight" and the narration it produced
    satisfied `fact_spoken` — nobody had said anything."""
    pressures = yaml.safe_load((world / "pressures.yaml").read_text(encoding="utf-8"))
    pressures[0]["intent"] = "Somebody looks at the music box for a moment too long."
    (world / "pressures.yaml").write_text(yaml.safe_dump(pressures), encoding="utf-8")

    assert any("intent" in c for c in complaints(world))


def test_a_scene_opening_that_names_a_fact(world):
    """The sharpest of the four: the player perceives it in full before
    the story has a first line, so an arc can end on its own opening
    paragraph."""
    edit(world / "scenes" / "the_reckoning.yaml", lambda s: s.update(
        {"opening": "Nobody has mentioned the music box since March."}
    ))

    assert any("opening" in c for c in complaints(world))


# --- References that have to resolve ----------------------------------


def test_a_character_who_starts_nowhere(world):
    edit(world / "characters" / "maria.yaml", lambda c: c.update({"location_id": "cellar"}))

    assert any("not a room" in c for c in complaints(world))


def test_a_goal_about_a_fact_that_does_not_exist(world):
    """It stays open forever and raises nobody's bid, silently."""
    edit(world / "characters" / "maria.yaml", lambda c: c.update(
        {"goals": [{"id": "find_out", "description": "Find out", "about": "the_will"}]}
    ))

    assert any("the_will" in c for c in complaints(world))


def test_somebody_who_falls_asleep_with_nothing_to_wake_them(world):
    """They stay under for the rest of the scene, perceiving nothing."""
    edit(world / "characters" / "maria.yaml", lambda c: c.update(
        {"intentions": [{
            "id": "turn_in", "description": "goes up", "location_id": "study",
            "ready_after_minutes": 30, "state": "asleep",
        }]}
    ))

    assert any("wakes them" in c for c in complaints(world))


def test_a_relationship_with_somebody_who_does_not_exist(world):
    edit(world / "characters" / "maria.yaml", lambda c: c.update(
        {"relationships": {"grandmother": {"affinity": 0.5, "trust": 0.5}}}
    ))

    assert any("does not exist" in c for c in complaints(world))


# --- Scenes ------------------------------------------------------------


def test_a_scene_with_nobody_for_the_player_to_be(world):
    edit(world / "characters" / "elena.yaml", lambda c: c.update({"is_user": False}))

    assert any("for the player to be" in c for c in complaints(world))


def test_a_scene_that_casts_somebody_who_does_not_exist(world):
    edit(world / "scenes" / "the_dinner.yaml", lambda s: s.update(
        {"cast": ["tomas", "maria", "elena", "grandmother"]}
    ))

    assert any("grandmother" in c for c in complaints(world))


def test_a_scene_that_leads_nowhere_it_could_go(world):
    """A story that leads somewhere that does not exist stops dead at the
    seam, and nothing says so until a player gets there."""
    edit(world / "scenes" / "the_reckoning.yaml", lambda s: s.update(
        {"next": [{"scene": "the_funeral"}]}
    ))

    assert any("the_funeral" in c for c in complaints(world))


def test_a_scene_that_ends_on_a_fact_nobody_wrote(world):
    edit(world / "scenes" / "the_reckoning.yaml", lambda s: s.update(
        {"end_condition": {"fact_spoken": "the_will"}}
    ))

    assert any("the_will" in c for c in complaints(world))


def test_a_condition_that_asks_something_the_engine_cannot_answer(world):
    """The sharp one. `evaluate_trigger` raises on a key it does not know
    rather than quietly passing — which is right at runtime, and means an
    author who invents `after_turns` has written a world that ends the
    scene it is in with a traceback instead of a morning."""
    edit(world / "scenes" / "the_reckoning.yaml", lambda s: s.update(
        {"next": [{"scene": "the_morning_after", "when": {"after_turns": 5}}]}
    ))

    assert any("after_turns" in c for c in complaints(world))


def test_a_branch_on_somebody_standing_somewhere_that_does_not_exist(world):
    edit(world / "scenes" / "the_reckoning.yaml", lambda s: s.update(
        {"next": [{"scene": "the_morning_after", "when": {"character_at": {"maria": "attic"}}}]}
    ))

    assert any("attic" in c for c in complaints(world))


def test_a_scene_that_ends_on_where_nobody_is(world):
    edit(world / "scenes" / "the_reckoning.yaml", lambda s: s.update(
        {"end_condition": {"character_at": {"the_gardener": "kitchen"}}}
    ))

    assert any("the_gardener" in c for c in complaints(world))


def test_a_world_with_no_scenes(world):
    for scene in (world / "scenes").glob("*.yaml"):
        scene.unlink()

    assert any("no scenes" in c for c in complaints(world))


def test_a_world_that_does_not_load_at_all(world):
    (world / "world.yaml").write_text("id: broken\nrooms: {\n", encoding="utf-8")

    found = complaints(world)

    assert len(found) == 1 and "does not load" in found[0]


# --- Keywords that are not particular to anything ---------------------


def test_a_secret_recognised_by_the_name_of_a_room(world):
    """Measured on a 1.5B, which made "kitchen" the keyword of a secret
    in a world with a kitchen in it. Nothing about that is visible by
    reading the world: it shows up as an arc ending on its own first
    line, because somebody said where they were."""
    edit(world / "world.yaml", lambda w: w["facts"].update(
        {"music_box": {"keywords": ["the kitchen"]}}
    ))

    assert any("saying where they are" in c for c in complaints(world))


def test_a_secret_recognised_by_a_word_already_doing_scenery(world):
    edit(world / "world.yaml", lambda w: w["facts"].update(
        {"music_box": {"keywords": ["kettle"]}}
    ))

    assert any("already scenery" in c for c in complaints(world))


def test_a_phrase_particular_to_the_thing_is_fine(world):
    """Judged against this world's own words rather than a list of common
    ones, because the engine speaks no language of its own — and a
    Portuguese world whose rooms are "a cozinha" and "o quintal" must not
    lose "o barco" to an English article list."""
    edit(world / "world.yaml", lambda w: w["facts"].update(
        {"music_box": {"keywords": ["grandmother's music box"]}}
    ))

    assert playable(world)
