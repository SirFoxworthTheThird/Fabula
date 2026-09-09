"""Everything that has to be true of a world before anybody plays it.

These checks lived only in the test suite, parametrised over the four
worlds in the repo. That was fine while every world was written by hand
and reviewed by a person — and it stops being fine the moment a world
arrives from somewhere else: generated from a sentence, copied in from
another machine, edited by somebody who has never read the spec. A rule
that is only enforced when somebody remembers to run pytest is not
enforced.

So they live here, they return complaints rather than raising, and the
test suite is one caller among several. What each of them is for is in
the check itself; the two worth knowing up front:

* **No authored prose may name a fact.** Room descriptions, pressure
  intents, intention text and scene openings all reach the log as event
  content, and a keyword in any of them lets the subject be raised by
  scenery — satisfying `fact_spoken` when nobody has said anything. A
  scene opening is the sharpest case, because it is perceived in full
  before the story has a first line.

* **No persona may name a secret that is not its own.** Authored prose
  is the one place a leak can be written by hand: the projection cannot
  catch it, because it never passes through one.
"""
from __future__ import annotations

from pathlib import Path

from fabula.loader import load_characters, load_pressures, load_scene, load_world
from fabula.world import mentions_fact


def _named_facts(condition: dict) -> list[str]:
    named: list[str] = []
    for key in ("fact_spoken", "fact_unspoken"):
        value = condition.get(key)
        named += [value] if isinstance(value, str) else list(value or [])
    return named


def complaints(world_dir: Path) -> list[str]:
    """Everything wrong with this world, in the order it was found.

    Empty means it is playable. Nothing here loads a model, opens a
    store or runs a scene: it is the authored files and the rules about
    them.
    """
    world_dir = Path(world_dir)
    try:
        world = load_world(world_dir)
        characters = load_characters(world_dir)
        pressures = load_pressures(world_dir)
        scenes = {
            path.stem: load_scene(world_dir, path.stem)
            for path in sorted((world_dir / "scenes").glob("*.yaml"))
        }
    except Exception as unreadable:
        return [f"the world does not load: {unreadable}"]

    found: list[str] = []

    def names(text: str) -> list[str]:
        return [f for f in world.facts if mentions_fact(world.facts[f], text or "")]

    if not scenes:
        found.append("there are no scenes to play")

    # --- Prose that reaches the log ------------------------------------
    for room_id, room in world.rooms.items():
        for fact_id in names(room.description):
            found.append(f"room {room_id} describes itself with the words of {fact_id}")
    for pressure in pressures:
        for fact_id in names(pressure.intent):
            found.append(f"pressure {pressure.id} names {fact_id} in its intent")
    for character in characters.values():
        for intention in character.intentions:
            for fact_id in names(intention.description):
                found.append(f"{character.id}'s intention {intention.id} names {fact_id}")
        # A persona may name what its own character is protecting, and
        # nothing else: that is the difference between a character who
        # knows a secret and a character who has been handed one.
        for fact_id in names(character.persona):
            if fact_id not in character.protects:
                found.append(f"{character.id}'s persona names {fact_id}, which is not theirs")
    for scene in scenes.values():
        for fact_id in names(scene.opening):
            found.append(f"scene {scene.id} names {fact_id} in its opening")

    # --- References that have to resolve -------------------------------
    for character in characters.values():
        if character.location_id not in world.rooms:
            found.append(f"{character.id} starts in {character.location_id}, which is not a room")
        for fact_id in character.protects:
            if fact_id not in world.facts:
                found.append(f"{character.id} protects {fact_id}, which is not a fact")
        for intention in character.intentions:
            if intention.location_id not in world.rooms:
                found.append(f"{character.id}'s intention {intention.id} is in no room")
        # Somebody put under with nothing to wake them stays under for
        # the rest of the scene, perceiving nothing.
        asleep = [i for i in character.intentions if i.state == "asleep"]
        awake = [i for i in character.intentions if i.state == "awake"]
        for turning_in in asleep:
            if not any(w.ready_after_minutes > turning_in.ready_after_minutes for w in awake):
                found.append(f"{character.id} falls asleep and nothing wakes them")
        for goal in character.goals:
            # A goal about a fact that does not exist stays open forever
            # and raises nobody's bid, silently.
            if goal.about is not None and goal.about not in world.facts:
                found.append(f"{character.id}'s goal {goal.id} is about {goal.about}, which is not a fact")
        for toward in character.relationships:
            if toward not in characters:
                found.append(f"{character.id} has a relationship with {toward}, who does not exist")

    for pressure in pressures:
        effect = pressure.effect
        if effect.get("location") and effect["location"] not in world.rooms:
            found.append(f"pressure {pressure.id} fires in a room that does not exist")
        if effect.get("actor") and effect["actor"] not in characters:
            found.append(f"pressure {pressure.id} acts as somebody who does not exist")
        for fact_id in _named_facts(pressure.trigger):
            if fact_id not in world.facts:
                found.append(f"pressure {pressure.id} triggers on {fact_id}, which is not a fact")
        for key in ("character_at", "character_not_at"):
            for cid, room in (pressure.trigger.get(key) or {}).items():
                if cid not in characters or room not in world.rooms:
                    found.append(f"pressure {pressure.id} triggers on {cid} in {room}")

    # --- Scenes ---------------------------------------------------------
    for scene in scenes.values():
        players = [cid for cid in scene.cast if cid in characters and characters[cid].is_user]
        if len(players) != 1:
            found.append(f"scene {scene.id} has {len(players)} characters for the player to be")
        for cid in scene.cast:
            if cid not in characters:
                found.append(f"scene {scene.id} casts {cid}, who does not exist")
        for cid in scene.may_arrive:
            if cid not in characters:
                found.append(f"scene {scene.id} awaits {cid}, who does not exist")
        for cid in sorted(set(scene.cast) & set(scene.may_arrive)):
            found.append(f"scene {scene.id} both casts and awaits {cid}")
        for cid, room in scene.starting_positions.items():
            if cid not in characters or room not in world.rooms:
                found.append(f"scene {scene.id} starts {cid} in {room}")
        for fact_id in _named_facts(scene.end_condition):
            if fact_id not in world.facts:
                found.append(f"scene {scene.id} ends on {fact_id}, which is not a fact")
        # An arrival pressure naming somebody outside the room can only
        # fire if the scene said they might turn up; otherwise it appends
        # an event with an actor nobody in the scene has heard of.
        for pressure in pressures:
            actor = pressure.effect.get("actor")
            if pressure.effect.get("kind") == "arrival" and actor and actor not in scene.cast:
                if actor not in scene.may_arrive:
                    found.append(
                        f"scene {scene.id}: pressure {pressure.id} lands {actor}, "
                        "who is neither cast nor awaited"
                    )
        # A story that leads somewhere that does not exist stops dead at
        # the seam, and nothing says so until a player gets there.
        for successor in scene.next:
            following = successor.get("scene")
            if following not in scenes:
                found.append(f"scene {scene.id} leads to {following}, which is not a scene")
            elif following == scene.id and not successor.get("when"):
                found.append(f"scene {scene.id} leads to itself unconditionally")
            for fact_id in _named_facts(successor.get("when") or {}):
                if fact_id not in world.facts:
                    found.append(f"scene {scene.id} branches on {fact_id}, which is not a fact")

    return found


def playable(world_dir: Path) -> bool:
    return not complaints(world_dir)
