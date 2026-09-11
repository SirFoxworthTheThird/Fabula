"""A world from a sentence.

Everything a story turns on is authored YAML, and that is right for the
parts a story turns on. It is wrong as the only way to *start* one: the
audience for this app does not write YAML, and until now the whole app
offered seven openings somebody else had thought of.

So this writes the file the author would have written. Not a new runtime
and not a different kind of world — the same directory of the same YAML,
which the shelf then lists beside the shipped ones and the engine plays
without knowing where it came from.

Two rules make it something other than a wish:

* **The engine disposes.** Everything the model returns is a proposal.
  Ids are slugged here, not taken; references are dropped unless they
  resolve; exactly one character is the player, whatever the model said.
  Nothing reaches disk that `fabula.inspect` has not read.

* **Prose is repaired, not accepted.** The one rule a generator breaks
  constantly is the one about authored prose naming a fact: it writes a
  room description that says "the music box" and hands the secret to the
  scenery. Those come back as complaints, and each gets one rewrite with
  the words it may not use spelled out. What still fails is dropped
  rather than shipped.

Cost is a handful of calls, once, when a world is made — not per turn,
not per character, not per remembered moment.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import yaml

from fabula.discovery import slug
from fabula.inspect import complaints, words_in
from fabula.llm import LLMClient
from fabula.loader import Scene
from fabula.shelf import stocked
from fabula.world import mentions_fact

# How many rewrites a piece of prose gets before it is dropped instead.
REPAIRS = 2

# When a generated story starts. Taken from the loader's own default so
# the two do not drift: the aftermath scenes are this plus the hours the
# model asked for, and that arithmetic has to be against the real value.
BEGINS = Scene.model_fields["start_time"].default


# The names in the example the model is shown. Small models hand the
# example back rather than designing anything — measured on a 1.5B,
# which returned Ashgrove, Elena and the music box when asked for a
# heist in a hotel kitchen. An example is still the only thing that
# reliably fixes the shape, so it stays and the copy is caught instead.
EXEMPLAR = {"ashgrove", "elena rey", "tomas rey", "the broken music box", "the dinner"}


# What kind of thing the room is circling, and therefore what story this
# is. Every generated world used to be the same one — somebody is hiding
# something, and the morning after it came out — whatever the premise
# asked for. A heist got it. A love story got it. The shipped worlds
# being quiet literary drama is the same mistake from the other side.
#
# The vocabulary never leaves this module, exactly as the trigger keys
# never do. A shape the model invents is not a story the engine cannot
# play, though: it is only a brief, so an unrecognised one falls back to
# the first rather than raising. Nothing downstream branches on it —
# every shape compiles to the same YAML, because the difference between
# a heist and a confession is what people are holding, not what the
# engine does with it.
#
# All four are one-to-many, and that is not a style choice. Withholding
# is not audience-aware — a character pressed on something they protect
# deflects whoever asked — so a secret is kept from *the room*, never
# from one person in it. Which means the player is always the one it is
# being kept from, and "you are on the crew" is a shape this engine
# cannot play. `a conspiracy` is the version it can: you are the one
# nobody is telling.
DEFAULT_SHAPE = "a secret"

SHAPES: dict[str, dict[str, str]] = {
    "a secret": {
        "is": "Somebody did something, or let something happen, and has never said so. "
              "The thing itself is small and particular — an object, an afternoon, a "
              "letter — and what it costs is not.",
        "held": "One character keeps it. Everybody else, the player included, does not "
                "know it exists.",
        "after": "the morning after it was finally said",
    },
    "a conspiracy": {
        "is": "Everybody in the room but you has agreed on something, and is being "
              "careful in front of you. The thing is a plan or an arrangement, already "
              "made, with a time on it.",
        "held": "Two or three characters keep the same one thing — all of them protect "
                "it. The player is the one it is being kept from, and is the only "
                "person in the room who does not know.",
        "holders": "everybody",
        "after": "the morning after you worked it out: the one where they went through "
                 "with it, and the one where they did not",
    },
    "a wanting": {
        "is": "Somebody has wanted to say something to you for a long time and has not. "
              "The thing being held back is about the player — how this person feels, "
              "what they decided, what they have been waiting for.",
        "held": "One character keeps it, and it is about the player. Nobody else in the "
                "room is holding anything.",
        "after": "the morning after they said it: the one where it was heard, and the "
                 "one where it was not",
    },
    "a danger": {
        "is": "Something is wrong with this place, and one person knows what. The thing "
              "is a fact about the building, the weather, the water, the road out — "
              "something that will not wait.",
        "held": "One character knows and has not said, because saying it starts "
                "something. Nobody else knows, and the player least of all.",
        "after": "the morning after it was said out loud: the one where it was in time, "
                 "and the one where it was not",
    },
}

SHAPE_SYSTEM = """You are told a premise for a roleplaying story and you say what kind of story it is. Answer with one JSON object and nothing else:

{"shape": "a secret"}

The only answers allowed are:

- "a secret" — one person did something, or let something happen, and has never said so. Guilt, shame, a thing that came apart. Families, old friends, houses.
- "a conspiracy" — several people have agreed on something and are being careful in front of one who has not been told. Heists, jobs, cons, mutinies, a room that goes quiet when somebody walks in. A plan going wrong is still a plan: this is the answer for anything with a crew in it.
- "a wanting" — somebody has wanted to say something to another person for a long time and has not. Love, apology, goodbye, a question they have been carrying.
- "a danger" — something is wrong with the *place*, and one person knows what. Storms, sickness, a building, the water, a thing in the dark. Not a plan that went wrong: a place that is not safe.

The test is what is being kept quiet and by how many people. One person with a past: "a secret". Several people with a plan: "a conspiracy". One person with a feeling: "a wanting". One person who knows the place is not safe: "a danger"."""



class CannotInvent(RuntimeError):
    """The model did not return a world that could be made playable."""


@dataclass
class Invented:
    """What was made, and what had to be given up making it."""

    world_dir: Path
    title: str
    scene: str
    opening: str = ""
    # What kind of story it decided this was. Reported rather than only
    # acted on: a premise that asked for a heist and got a confession is
    # a thing somebody should be told, and the shape is the one place
    # that goes wrong quietly.
    shape: str = DEFAULT_SHAPE
    repaired: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def _json(raw: str) -> dict:
    """The object in whatever the model said.

    Models fence JSON, preface it, and apologise after it. Anything that
    is not an object is nothing, and the caller asks again.
    """
    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        found = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return found if isinstance(found, dict) else {}


def _ask(llm: LLMClient, system: str, prompt: str, key: str, tries: int = 3) -> dict:
    for _ in range(tries):
        found = _json(llm.complete(system=system, prompt=prompt, key=key))
        if found:
            return found
    raise CannotInvent("the model did not answer with anything this could read")


# The shape is shown by example rather than described, because a model
# handed "title": "two or three words" writes a scene called "two or
# three words" — measured, on a 1.5B, first time out.
WORLD_SYSTEM = """You design the setting for a roleplaying story engine. Answer with one JSON object and nothing else, in the same shape as this one:

{
  "title": "Ashgrove",
  "blurb": "A house, a family, and something one of them has not said out loud.",
  "rooms": [
    {"name": "the kitchen", "description": "Warm, and the only room anyone sits in. A long scrubbed table, a kettle, the back door with its glass panel.", "adjacent": ["the study"]},
    {"name": "the study", "description": "Their grandmother's desk, and letters in date order that nobody has moved.", "adjacent": ["the kitchen"]}
  ],
  "secrets": [
    {"name": "the broken music box", "keywords": ["music box", "grandmother's box"]}
  ],
  "characters": [
    {"name": "Elena Rey", "persona": "The youngest, back for the weekend. She asks the question everybody else is working around, and she does not know what she is walking into.", "room": "the kitchen", "protects": [], "talkativeness": 0.6, "reticence": 0.2},
    {"name": "Tomas Rey", "persona": "Her brother. He broke it in March and let the cat be blamed. He deflects with small talk and cannot hold eye contact while he does it.", "room": "the kitchen", "protects": ["the broken music box"], "talkativeness": 0.4, "reticence": 0.8}
  ],
  "player": "Elena Rey"
}

That example is the *shape*, from a different story. Design a new world for the premise you are given: a different place, different people, a different secret, none of its names.

Rules that matter:
- Three or four rooms, joined by `adjacent` so somebody can walk between them. Small places: a scene is people in rooms, not a map.
- Two to four characters, one of whom is the player. The player protects nothing: they are the one who does not know.
- One or two secrets. Who keeps them is in the brief you are given; whoever it is, it is never the player. The player is the one it is being kept from.
- A secret's keywords are the exact phrases somebody would say out loud to name the thing itself. They must be particular to it: never the name of a room, never a single ordinary word like "window" or "money". "the second key" is a keyword; "key" is not.
- A room description must never contain any of the secrets' keywords. The room is where the story happens, not where it is told.
- No description or persona may say what the player thinks or feels."""


SCENE_SYSTEM = """You write the opening scene for a roleplaying story engine. Answer with one JSON object and nothing else, in the same shape as this one:

{
  "title": "The dinner",
  "premise": "Your brother has been strange all evening, and your sister is in the next room.",
  "opening": "Sunday at the house you grew up in. Tomas has been at the kitchen table since six and has said about twenty words, none of them about anything. Maria is in the study with the door not quite shut. Nobody has eaten yet.",
  "positions": {"Elena Rey": "the kitchen", "Tomas Rey": "the kitchen"},
  "ends_when": "the broken music box"
}

Rules that matter:
- The opening is read by the player and by nobody else, so it may say what only they would know. It must never contain a secret's keywords: they do not know them.
- The opening must not say what any other character is thinking or hiding.
- Everybody in the cast gets a position. Put the player in a room with at least one other person."""


def _shape(premise: str, llm: LLMClient) -> str:
    """What kind of story the premise is asking for.

    One extra call, on the writing model rather than the cheap one: it is
    one word, but it is the word that decides what the whole world is,
    and a wrong genre is not something `/reveal` shows you a turn later.

    Never fatal. A shape is a brief, and a premise that could not be
    classified still gets a story — the one every premise used to get.
    """
    try:
        answered = _ask(
            llm, SHAPE_SYSTEM, f"The story: {premise}\n\nWhat kind is it?",
            key="invent:shape", tries=2,
        )
    except CannotInvent:
        return DEFAULT_SHAPE
    said = str(answered.get("shape") or "").strip().lower()
    return said if said in SHAPES else DEFAULT_SHAPE


def _distinctive(keyword: str, rooms: dict) -> bool:
    """Is this phrase particular to the thing, or is it the furniture?

    The same rule `fabula.inspect` applies, applied before the file is
    written: the generator should not hand the validator something it is
    going to refuse.
    """
    said = words_in(keyword)
    if not said:
        return False
    for room in rooms.values():
        name = words_in(room["name"])
        if said <= name or name <= said:
            return False
    if len(said) == 1:
        prose = set()
        for room in rooms.values():
            prose |= words_in(room["description"])
        if said <= prose:
            return False
    return True


# What makes a scene escalate rather than converse. Both are asked for in
# plain words — a sentence, a room, a person, a number of turns — and the
# machinery is built here: the trigger vocabulary is never shown to the
# model, because an unrecognised key would make a pressure always
# eligible, which is the worst authoring failure there is.
COMPLICATIONS_SYSTEM = """You add the complications to a story somebody else has designed. Answer with one JSON object and nothing else, in the same shape as this one:

{
  "pressures": [
    {"what": "A pan is set down harder than it needed to be, and nobody looks up.",
     "where": "the kitchen", "after_turns": 5, "while_unsaid": "the broken music box",
     "how_hard": 0.45, "at_most": 2},
    {"what": "The light in the hall goes off on its timer, and somebody has to get up.",
     "where": "the hall", "after_turns": 12, "while_unsaid": null,
     "how_hard": 0.35, "at_most": 1}
  ],
  "intentions": [
    {"who": "Tomas Rey", "what": "checks the seam where he glued it, and puts it back",
     "where": "the kitchen", "after_minutes": 20, "alone": true}
  ],
  "goals": [
    {"who": "Tomas Rey", "wants": "to get through the evening without it coming up",
     "about": "the broken music box", "how_much": 0.8},
    {"who": "Maria Rey", "wants": "to find out what her brother is being strange about",
     "about": "the broken music box", "how_much": 0.5}
  ]
}

Rules that matter:
- A pressure is something that happens *to* the room: a noise, a light, a door, the weather. Never somebody speaking, and never anybody's thoughts.
- `what` must never contain a secret's exact words. It can make the subject harder to avoid without naming it.
- `while_unsaid` is the name of a secret, or null: the pressure only fires while nobody has said it out loud.
- An intention is something one person does when nobody is watching, in one room, at a set time. `alone: true` means it waits for the room to empty.
- Never give the player an intention. They are played by somebody who is here.
- A goal is a standing want, not a reflex: what somebody is still trying to do about the thing all evening. `about` is the name of a secret, and `wants` must never contain its exact words.
- Somebody keeping a secret wants it to stay quiet; somebody who has noticed wants to know. Both are goals. Never give the player one.
- Two or three pressures, one or two intentions, one goal each for the people who have a reason to have one."""


# Where the story goes once the secret is out. The branch is not asked
# for — the engine names the two situations and the model writes the
# prose for each, because a condition is a thing this file can build and
# a morning is not.
AFTERMATH_SYSTEM = """You write what happens the morning after, for a roleplaying story engine. You are given a scene that ends when a secret is finally said out loud, and you write the two scenes that could follow it. Answer with one JSON object and nothing else, in the same shape as this one:

{
  "heard": {
    "title": "The morning after",
    "premise": "Whatever was said last night, everyone woke up in the same house.",
    "opening": "Morning, and the kettle is on. Nobody has mentioned last night. Maria went into the study before you came down and has not come out of it.",
    "positions": {"Elena Rey": "the kitchen", "Tomas Rey": "the kitchen", "Maria Rey": "the study"},
    "hours_later": 13
  },
  "unheard": {
    "title": "Nobody said a word",
    "premise": "It kept. Breakfast, as though the evening had not happened.",
    "opening": "Breakfast, all three of you at the table, and the evening behind you filed away as though it had gone the way these evenings usually go. It did not.",
    "positions": {"Elena Rey": "the kitchen", "Tomas Rey": "the kitchen", "Maria Rey": "the kitchen"},
    "hours_later": 13
  }
}

Rules that matter:
- Same place, same people, later. Nobody new, nowhere new: you are writing a morning, not a sequel.
- The difference between the two is who was standing there. You are told whose presence decides it; write `heard` as the morning after they were, and `unheard` as the morning after they were not — in that one they still do not know, and are the only person at the table who does not.
- An opening is read by the player and by nobody else, so it may say what only they would know. It must never contain the secret's exact words: each scene counts the subject as unraised until somebody says it out loud in that scene.
- Say what is different about the morning, not what anybody is thinking or has decided.
- `hours_later` is how long after the first scene this one begins."""


def _traits(character: dict) -> dict:
    def number(key: str, fallback: float) -> float:
        try:
            return max(0.0, min(1.0, float(character.get(key, fallback))))
        except (TypeError, ValueError):
            return fallback

    return {
        "talkativeness": number("talkativeness", 0.6),
        "reticence": number("reticence", 0.3),
        "salience_bias": {"utterance": 1.0, "action": 1.0},
    }


def invent(
    premise: str,
    llm: LLMClient,
    worlds_root: Path | str | None = None,
    player_name: str = "",
) -> Invented:
    """Make a world from a sentence, and refuse to keep one that is not
    playable."""
    premise = " ".join((premise or "").split())
    if not premise:
        raise CannotInvent("say what the story is about")

    shape = _shape(premise, llm)
    circling = SHAPES[shape]
    asked = (
        f"The story: {premise}\n\n"
        f"What the room is circling: {circling['is']}\n"
        f"Who is holding it: {circling['held']}\n\n"
        "Design the setting."
    )
    drafted = _ask(llm, WORLD_SYSTEM, asked, key="invent:world")
    if _is_the_example(drafted):
        drafted = _ask(
            llm,
            WORLD_SYSTEM,
            asked + (
                "\n\nThe world in the instructions is an example of the shape, from a "
                "different story. Design a new one for this premise: different place, "
                "different people, different secret. Reuse none of its names."
            ),
            key="invent:world",
        )
        if _is_the_example(drafted):
            raise CannotInvent("the model handed back the example instead of a world")
    world_id = slug(str(drafted.get("title") or premise)[:40]) or "story"
    # Somewhere that is yours, rather than wherever the process happened
    # to be started.
    worlds_root = Path(worlds_root) if worlds_root is not None else stocked()
    world_dir = Path(worlds_root) / world_id
    for suffix in range(2, 40):
        if not world_dir.exists():
            break
        world_dir = Path(worlds_root) / f"{world_id}_{suffix}"

    built = _build(drafted, premise, llm, player_name, shape)
    built["shape"] = shape

    rooms = ", ".join(room["name"] for room in built["rooms"].values())
    who = "; ".join(
        f"{character['name']} — {character['persona']}"
        for character in built["cast"].values()
    )
    secrets = ", ".join(built["facts"]) or "none"
    scene = _ask(
        llm,
        SCENE_SYSTEM,
        f"The story: {premise}\n\n"
        f"The place: {built['blurb']}\n"
        f"Rooms: {rooms}\n"
        f"Who is in it: {who}\n"
        f"The player is {built['player_name']}.\n"
        f"Secrets, which the player does not know: {secrets}\n"
        f"What the room is circling: {circling['is']}\n\n"
        "Write the opening scene.",
        key="invent:scene",
    )
    # What makes a scene escalate rather than converse. Asked for after
    # the scene, so it can be about a story that already exists.
    try:
        built["extras"] = _ask(
            llm,
            COMPLICATIONS_SYSTEM,
            f"The story: {premise}\n\n"
            f"Rooms: {rooms}\n"
            f"Who is in it: {who}\n"
            f"The player is {built['player_name']}, and gets no intention.\n"
            f"Secrets: {secrets}\n\n"
            "Write the complications.",
            key="invent:complications",
        )
    except CannotInvent:
        # A world with no complications is quieter, not broken: the
        # scene runs on the people in it. Losing the whole world over
        # the part that escalates it would be the wrong trade.
        built["extras"] = {}
        built["quiet"] = True

    # And where it goes once the secret is out. Asked last, because it is
    # the only call that needs to know both what the scene ends on and
    # whose being in the room decides what that costs.
    ends_on = _ends_on(scene, built["facts"])
    built["witness"] = witness(built, ends_on)
    if ends_on and built["witness"]:
        seen = built["cast"][built["witness"]]
        try:
            built["after"] = _ask(
                llm,
                AFTERMATH_SYSTEM,
                f"The story: {premise}\n\n"
                f"Rooms: {rooms}\n"
                f"Who is in it: {who}\n"
                f"The player is {built['player_name']}.\n"
                f"The scene ends when {ends_on.replace('_', ' ')} is finally said out "
                f"loud, in {built['rooms'][built['cast'][built['player_id']]['location_id']]['name']}.\n"
                f"What you are writing is {circling['after']}.\n"
                f"Whether {seen['name']} was standing there when it came out is what "
                f"decides which of the two mornings gets played.\n\n"
                "Write both mornings.",
                key="invent:aftermath",
            )
        except CannotInvent:
            # Same trade as the complications: a story that stops after
            # one scene is smaller than the one asked for, and still a
            # story. Losing the world over its second act is not.
            built["after"] = {}

    made = _write(world_dir, built, scene, premise)
    if not made.opening:
        # The opening is the first thing anybody reads, and one that had
        # to be dropped for naming a secret leaves the app with nothing
        # to say. Worth one more scene rather than shipping silence.
        _burn(world_dir)
        scene = _ask(
            llm,
            SCENE_SYSTEM,
            f"The story: {premise}\n\n"
            f"The place: {built['blurb']}\n"
            f"Rooms: {rooms}\n"
            f"Who is in it: {who}\n"
            f"The player is {built['player_name']}.\n"
            f"Secrets, which the player does not know and must not be named "
            f"or described in the opening: {secrets}\n\n"
            "Write the opening scene. The opening must not mention the secrets at all.",
            key="invent:scene",
        )
        made = _write(world_dir, built, scene, premise)
        if not made.opening:
            _burn(world_dir)
            raise CannotInvent("the opening kept giving the secret away")

    # Whatever is left after the prose repairs. A world that still breaks
    # a rule is not shipped: those rules are what makes the fiction hold,
    # and half of them cannot be noticed by reading it.
    found = complaints(world_dir)
    if found:
        _burn(world_dir)
        raise CannotInvent("; ".join(found[:3]))
    return made


def _is_the_example(drafted: dict) -> bool:
    """Did it design a world, or copy the one it was shown?"""
    given = {str(drafted.get("title") or "").strip().lower()}
    given |= {
        str(person.get("name") or "").strip().lower()
        for person in drafted.get("characters") or []
        if isinstance(person, dict)
    }
    given |= {
        str(secret.get("name") or "").strip().lower()
        for secret in drafted.get("secrets") or []
        if isinstance(secret, dict)
    }
    return len(given & EXEMPLAR) >= 2


def _burn(world_dir: Path) -> None:
    """A world nobody can play should not be on the shelf."""
    import shutil

    shutil.rmtree(world_dir, ignore_errors=True)


def _unique(name: str, taken: set[str], fallback: str) -> str:
    base = slug(name or "") or fallback
    made = base
    for suffix in range(2, 40):
        if made not in taken:
            break
        made = f"{base}_{suffix}"
    taken.add(made)
    return made


def _build(
    drafted: dict, premise: str, llm: LLMClient, player_name: str,
    shape: str = DEFAULT_SHAPE,
) -> dict:
    """Turn what the model proposed into ids and references that resolve.

    Nothing here trusts a name to be an id, a room to exist, or the model
    to have remembered which of its own characters was the player.
    """
    rooms: dict[str, dict] = {}
    taken: set[str] = set()
    by_name: dict[str, str] = {}
    for drafted_room in drafted.get("rooms") or []:
        if not isinstance(drafted_room, dict) or not drafted_room.get("name"):
            continue
        name = str(drafted_room["name"]).strip()
        room_id = _unique(name, taken, f"room_{len(rooms) + 1}")
        by_name[name.lower()] = room_id
        rooms[room_id] = {
            "id": room_id,
            "name": name,
            "description": " ".join(str(drafted_room.get("description") or "").split()),
            "adjacent": [str(n).strip().lower() for n in drafted_room.get("adjacent") or []],
        }
    if len(rooms) < 2:
        raise CannotInvent("a story needs somewhere to be, and somewhere else to go")

    # Adjacency by name, resolved to ids and made mutual: a one-way edge
    # is a deliberate authoring move (a room you can listen into but not
    # out of) and not something to arrive at by accident.
    for room in rooms.values():
        neighbours = {by_name[n] for n in room["adjacent"] if n in by_name} - {room["id"]}
        room["adjacent"] = {other: "adjacent" for other in sorted(neighbours)}
    for room in rooms.values():
        for other in room["adjacent"]:
            rooms[other]["adjacent"].setdefault(room["id"], "adjacent")
    # Anywhere unreachable is joined to the first room rather than left
    # as a place the story can never get to.
    first = next(iter(rooms))
    for room in rooms.values():
        if not room["adjacent"] and room["id"] != first:
            room["adjacent"][first] = "adjacent"
            rooms[first]["adjacent"][room["id"]] = "adjacent"

    facts: dict[str, dict] = {}
    fact_taken: set[str] = set()
    fact_by_name: dict[str, str] = {}
    for drafted_fact in drafted.get("secrets") or []:
        if not isinstance(drafted_fact, dict):
            continue
        name = str(drafted_fact.get("name") or "").strip()
        keywords = [
            " ".join(str(k).split())
            for k in drafted_fact.get("keywords") or []
            if str(k).strip()
        ]
        # A fact nobody can say is a fact nothing can turn on: the whole
        # mechanism is keyword matching against what is said out loud.
        # And one that is the name of a room, or a single word already
        # doing scenery duty here, would end the arc on somebody saying
        # where they are — measured on a 1.5B, which made "kitchen" a
        # secret in a world with a kitchen in it.
        keywords = [k for k in keywords if len(k) > 3 and _distinctive(k, rooms)]
        if not keywords and len(name) > 3 and _distinctive(name, rooms):
            keywords = [name]
        if not name or not keywords:
            continue
        fact_id = _unique(name, fact_taken, f"secret_{len(facts) + 1}")
        fact_by_name[name.lower()] = fact_id
        facts[fact_id] = {"id": fact_id, "keywords": keywords}

    cast: dict[str, dict] = {}
    people_taken: set[str] = set()
    wanted_player = (player_name or str(drafted.get("player") or "")).strip().lower()
    for drafted_person in drafted.get("characters") or []:
        if not isinstance(drafted_person, dict) or not drafted_person.get("name"):
            continue
        name = str(drafted_person["name"]).strip()
        # A name out of the example, in a world about something else. The
        # wholesale copy is caught earlier; this is the one that leaks
        # through — measured on a 3B, which designed a hotel heist and
        # put Elena Rey in it.
        if name.lower() in EXEMPLAR:
            continue
        person_id = _unique(name, people_taken, f"person_{len(cast) + 1}")
        room = by_name.get(str(drafted_person.get("room") or "").strip().lower(), first)
        protects = [
            fact_by_name[str(p).strip().lower()]
            for p in drafted_person.get("protects") or []
            if str(p).strip().lower() in fact_by_name
        ]
        cast[person_id] = {
            "id": person_id,
            "name": name,
            "persona": " ".join(str(drafted_person.get("persona") or "").split()),
            "traits": _traits(drafted_person),
            "protects": protects,
            "location_id": room,
            "is_user": False,
        }
    if len(cast) < 2:
        raise CannotInvent("a story needs somebody in it besides you")

    # Exactly one player, whatever the model said. Preferring whoever it
    # named, then whoever protects nothing — the player is the one who
    # does not know.
    player_id = next(
        (cid for cid, c in cast.items() if c["name"].strip().lower() == wanted_player),
        next((cid for cid, c in cast.items() if not c["protects"]), next(iter(cast))),
    )
    cast[player_id]["is_user"] = True
    # And they protect nothing: a player keeping a secret from themselves
    # is the one shape this engine cannot play.
    cast[player_id]["protects"] = []
    if player_name.strip():
        cast[player_id]["name"] = player_name.strip()

    # A secret nobody keeps never comes up; give it to somebody who is
    # not the player rather than dropping the story's own subject.
    others = [cid for cid in cast if cid != player_id]
    for fact_id in facts:
        if not any(fact_id in cast[cid]["protects"] for cid in others):
            cast[others[0]]["protects"].append(fact_id)

    # A shape that says everybody is in on it gets everybody in on it.
    # Measured on the 3B: told in plain words that two or three of them
    # keep the same one thing, it designed a hotel heist and gave the
    # secret to one person, which is the shape it was told not to write.
    # The topology is the whole difference between a conspiracy and a
    # confession, so it is made true here rather than asked for — the
    # model proposes, and this is one of the things the engine disposes.
    if SHAPES.get(shape, {}).get("holders") == "everybody":
        for cid in others:
            cast[cid]["protects"] = list(facts)

    # Everybody knows of everybody: relationships are what regard moves
    # on, and a cast of strangers with no entries never moves at all.
    for person_id, person in cast.items():
        person["relationships"] = {
            other: {"affinity": 0.4, "trust": 0.5}
            for other in cast
            if other != person_id
        }

    return {
        "id": None,
        "title": str(drafted.get("title") or premise[:40]).strip(),
        "blurb": " ".join(str(drafted.get("blurb") or premise).split()),
        "rooms": rooms,
        "facts": facts,
        "cast": cast,
        "player_id": player_id,
        "player_name": cast[player_id]["name"],
        "llm": llm,
    }


def _clean(text: str, facts: dict, llm: LLMClient, about: str, repaired: list[str]) -> str:
    """Prose with none of the story's own keywords in it.

    The rule a generator breaks constantly: it writes a room description
    that says "the music box" and hands the secret to the scenery, where
    it satisfies `fact_spoken` before anybody has spoken. Rewriting is
    worth a call because the alternative is losing the description; the
    third failure loses it anyway, because shipping it would break the
    story rather than the prose.
    """
    text = " ".join((text or "").split())
    for _ in range(REPAIRS):
        named = [
            fact_id
            for fact_id, fact in facts.items()
            if mentions_fact(_fact(fact), text)
        ]
        if not named:
            return text
        forbidden = sorted({k for f in named for k in facts[f]["keywords"]})
        repaired.append(f"{about} named {', '.join(named)}")
        text = " ".join(
            llm.complete(
                system=(
                    "You rewrite one piece of prose for a story. Keep the length, the "
                    "voice and everything it is about. Answer with the rewritten prose "
                    "and nothing else."
                ),
                prompt=(
                    f"{text}\n\n"
                    f"Rewrite it without these words, and without naming what they name: "
                    f"{', '.join(forbidden)}.\n"
                    "It should still evoke the same place or moment."
                ),
                key="invent:repair",
            ).split()
        )
    named = [f for f, fact in facts.items() if mentions_fact(_fact(fact), text)]
    return "" if named else text


def _fact(fact: dict):
    from fabula.world import Fact

    return Fact(id=fact["id"], keywords=fact["keywords"])


def _clamp(value, low, high, fallback):
    try:
        number = type(fallback)(value)
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, number))


def _pressures(built: dict, repaired: list, dropped: list) -> list[dict]:
    """Complications, with the trigger written here rather than there.

    The model gets plain words — a sentence, a room, how many turns, and
    which secret it is about — and the vocabulary that decides when a
    pressure is eligible never leaves this file. An unrecognised trigger
    key would make a pressure always eligible, and a scene that escalates
    on turn one is worse than one that never does.
    """
    facts, rooms = built["facts"], built["rooms"]
    by_room = {room["name"].strip().lower(): room_id for room_id, room in rooms.items()}
    # A secret answers to its id, its id read as words, or anything it is
    # recognised by out loud — because the model is naming it from
    # memory, in whatever words it used two calls ago.
    by_fact: dict[str, str] = {}
    for fact_id, fact in facts.items():
        for said in [fact_id, fact_id.replace("_", " "), *fact["keywords"]]:
            by_fact.setdefault(said.strip().lower(), fact_id)
    written: list[dict] = []
    taken: set[str] = set()

    for drafted in (built.get("extras") or {}).get("pressures") or []:
        if not isinstance(drafted, dict):
            continue
        what = _clean(
            str(drafted.get("what") or ""), facts, built["llm"],
            f"pressure {len(written) + 1}", repaired,
        )
        if not what:
            dropped.append("a complication that would not stop naming the secret")
            continue
        where = by_room.get(str(drafted.get("where") or "").strip().lower())
        if where is None:
            where = next(iter(rooms))
        trigger: dict = {"turns_elapsed": f"> {_clamp(drafted.get('after_turns'), 2, 20, 5)}"}
        unsaid = str(drafted.get("while_unsaid") or "").strip().lower()
        # Matched against the ids this engine made, not the name the
        # model remembers giving it — and only ever to a fact that
        # exists. A trigger naming one that does not is a pressure that
        # can never fire, which reads exactly like one that was never
        # written.
        if len(unsaid) > 3:
            for name, fact_id in by_fact.items():
                if unsaid in name or name in unsaid:
                    trigger["fact_unspoken"] = fact_id
                    break
        written.append({
            "id": _unique(" ".join(what.split()[:4]), taken, f"pressure_{len(written) + 1}"),
            "intent": what,
            "weight": _clamp(drafted.get("how_hard"), 0.1, 0.9, 0.45),
            "trigger": trigger,
            # Narration only. An arrival needs somebody the scene said
            # might turn up, and a state change needs a scene written
            # around it; either from a generator is a pressure that fires
            # into a story nobody wrote.
            "effect": {"kind": "narration", "location": where},
            "cooldown_turns": _clamp(drafted.get("cooldown"), 4, 30, 10),
            "max_fires": _clamp(drafted.get("at_most"), 1, 3, 2),
        })
    return written


def _intentions(built: dict, repaired: list, dropped: list) -> dict[str, list[dict]]:
    """What somebody does when nobody is watching, per character."""
    facts, rooms, cast = built["facts"], built["rooms"], built["cast"]
    by_room = {room["name"].strip().lower(): room_id for room_id, room in rooms.items()}
    by_name = {person["name"].strip().lower(): pid for pid, person in cast.items()}
    theirs: dict[str, list[dict]] = {}

    for drafted in (built.get("extras") or {}).get("intentions") or []:
        if not isinstance(drafted, dict):
            continue
        person_id = by_name.get(str(drafted.get("who") or "").strip().lower())
        # Never the player: they are played by somebody who is here, and
        # an intention is what happens while they are not.
        if person_id is None or person_id == built["player_id"]:
            continue
        what = _clean(
            str(drafted.get("what") or ""), facts, built["llm"],
            f"{person_id}'s intention", repaired,
        )
        if not what:
            dropped.append(f"something {cast[person_id]['name']} meant to do")
            continue
        where = by_room.get(str(drafted.get("where") or "").strip().lower())
        theirs.setdefault(person_id, []).append({
            "id": f"intention_{len(theirs.get(person_id, [])) + 1}",
            "description": what,
            "location_id": where or cast[person_id]["location_id"],
            "ready_after_minutes": _clamp(drafted.get("after_minutes"), 5, 120, 25),
            "private": bool(drafted.get("alone", True)),
        })
    return theirs


def _goals(built: dict, repaired: list, dropped: list) -> dict[str, list[dict]]:
    """What somebody is still trying to do about it, per character.

    A generated cast used to have none, which made it a cast that only
    reacts: a goal raises the bid when its subject comes up and closes
    once the subject is out, so it is the difference between somebody
    who is keeping a secret and somebody who merely happens to hold one.

    Free, in calls — it comes back with the pressures and the intentions,
    from the one question that was already being asked.
    """
    facts, cast = built["facts"], built["cast"]
    by_name = {person["name"].strip().lower(): pid for pid, person in cast.items()}
    by_fact = {fact_id.replace("_", " "): fact_id for fact_id in facts}
    for fact_id, fact in facts.items():
        for keyword in fact["keywords"]:
            by_fact[keyword.strip().lower()] = fact_id
    theirs: dict[str, list[dict]] = {}

    for drafted in (built.get("extras") or {}).get("goals") or []:
        if not isinstance(drafted, dict):
            continue
        person_id = by_name.get(str(drafted.get("who") or "").strip().lower())
        # Never the player's. A want the engine raises bids on belongs to
        # somebody the engine is playing.
        if person_id is None or person_id == built["player_id"]:
            continue
        # A goal about a fact that does not exist stays open forever and
        # raises nobody's bid, silently — `inspect` refuses the world for
        # it, so it is resolved here rather than shipped.
        about = by_fact.get(str(drafted.get("about") or "").strip().lower())
        wants = _clean(
            str(drafted.get("wants") or ""), facts, built["llm"],
            f"{person_id}'s goal", repaired,
        )
        if not wants:
            dropped.append(f"something {cast[person_id]['name']} wanted")
            continue
        mine = theirs.setdefault(person_id, [])
        mine.append({
            "id": f"goal_{len(mine) + 1}",
            "description": wants,
            "priority": _clamp(drafted.get("how_much"), 0.1, 1.0, 0.5),
            **({"about": about} if about else {}),
        })
    return theirs


def _positions(named: dict | None, built: dict) -> dict[str, str]:
    """Who starts where, by name, resolved to ids that exist.

    Anybody the model forgot starts where their own file puts them, and
    the player never opens on an empty room: a first scene with nobody in
    it is the app asking somebody to entertain themselves.
    """
    positions: dict[str, str] = {}
    for name, room_name in (named or {}).items():
        person = next(
            (
                cid
                for cid, c in built["cast"].items()
                if c["name"].strip().lower() == str(name).strip().lower()
            ),
            None,
        )
        room = next(
            (
                rid
                for rid, r in built["rooms"].items()
                if r["name"].strip().lower() == str(room_name).strip().lower()
            ),
            None,
        )
        if person and room:
            positions[person] = room
    for person_id, person in built["cast"].items():
        positions.setdefault(person_id, person["location_id"])
    player_room = positions[built["player_id"]]
    if all(room != player_room for cid, room in positions.items() if cid != built["player_id"]):
        other = next(cid for cid in positions if cid != built["player_id"])
        positions[other] = player_room
    return positions


def _ends_on(scene: dict, facts: dict) -> str | None:
    """The secret this scene is over the moment somebody says out loud."""
    said = str(scene.get("ends_when") or "").strip().lower()
    return next(
        (
            fact_id
            for fact_id in facts
            if fact_id == slug(said)
            or said in [k.lower() for k in facts[fact_id]["keywords"]]
        ),
        next(iter(facts), None),
    )


def witness(built: dict, ends_on: str | None) -> str | None:
    """Whose being in the room decides which morning this becomes.

    The point of difference is that the same sentence lands differently
    depending on who heard it, and this is where that pays a story back
    rather than only a projection: the branch is *who was standing there*.

    So: somebody who is neither the player nor the person keeping it —
    they are the one it is news to — and preferably somebody who does not
    start in the room, because then their being there at the end is
    something that happened rather than something that was set up.
    """
    cast, player_id = built["cast"], built["player_id"]
    here = cast[player_id]["location_id"]
    others = [
        cid for cid in cast
        if cid != player_id and (ends_on is None or ends_on not in cast[cid]["protects"])
    ]
    elsewhere = [cid for cid in others if cast[cid]["location_id"] != here]
    if elsewhere:
        return elsewhere[0]
    if others:
        return others[0]
    # A two-hander: nobody but the person keeping it. Then the question
    # is whether they were still standing there when it came out, or had
    # walked off first.
    return next((cid for cid in cast if cid != player_id), None)


def _after(
    world_dir: Path,
    built: dict,
    first: str,
    positions: dict,
    ends_on: str | None,
    premise: str,
    repaired: list,
    dropped: list,
) -> list[dict]:
    """The mornings after, and the branch between them.

    A story goes somewhere; a scene stops. Until this existed a generated
    world was one room's worth of conversation and then nothing, which is
    the shape of a character-chat app and not of a story.

    The branch is built here and never asked for. `character_at` reads
    where somebody is standing when the scene is over, which is a proxy
    for who was there when it came out — the same proxy the hand-written
    worlds use, and the only one the condition language can see.
    """
    after, cast = built.get("after") or {}, built["cast"]
    if not ends_on or not after:
        # Nothing to be the morning *after*: a scene with no ending has no
        # sequel, and a branch on a fact this world does not have would be
        # a successor nobody could reach.
        return []
    heard = _one_morning(world_dir, built, after.get("heard"), first, 1, repaired, dropped)
    unheard = _one_morning(world_dir, built, after.get("unheard"), first, 2, repaired, dropped)
    who = built.get("witness")
    if heard and unheard and who and who != built["player_id"]:
        # Tried in order, first match winning, so the specific branch has
        # to come before the fallback.
        return [
            {"scene": heard, "when": {"character_at": {who: positions[built["player_id"]]}}},
            {"scene": unheard},
        ]
    # One morning is still somewhere to go. Unconditional, because a
    # branch with only one side is a condition that decides nothing.
    only = heard or unheard
    if only:
        dropped.append("one of the two mornings — the story goes the same way either way")
        return [{"scene": only}]
    return []


def _one_morning(
    world_dir: Path,
    built: dict,
    drafted: dict | None,
    first: str,
    which: int,
    repaired: list,
    dropped: list,
) -> str:
    """One aftermath scene, or "" if it could not be made into one."""
    if not isinstance(drafted, dict):
        return ""
    title = " ".join(str(drafted.get("title") or "").split())
    if not title or title.lower() in EXEMPLAR:
        title = f"After — {which}"
    scene_id = slug(title) or f"after_{which}"
    if scene_id == first:
        scene_id = f"{scene_id}_after_{which}"
    opening = _clean(
        str(drafted.get("opening") or ""), built["facts"], built["llm"],
        f"the opening of {scene_id}", repaired,
    )
    if not opening:
        # A morning that cannot say what is different about it is not a
        # scene, it is the same room again with the clock moved.
        dropped.append(f"the morning after, {title!r}, which kept naming the secret")
        return ""
    later = _clamp(drafted.get("hours_later"), 1, 48, 13)
    _save(
        world_dir / "scenes" / f"{scene_id}.yaml",
        {
            "id": scene_id,
            "world": world_dir.name,
            "title": title,
            "premise": " ".join(str(drafted.get("premise") or built["blurb"]).split()),
            "opening": opening,
            # Sandbox: there is no third act to escalate toward, and an
            # arc with no end condition is a sandbox that pushes for
            # nothing.
            "mode": "sandbox",
            "cast": list(built["cast"]),
            "starting_positions": _positions(drafted.get("positions"), built),
            "start_time": BEGINS + timedelta(hours=later),
            "turn_budget": 6,
            "max_consecutive_agent_turns": 3,
        },
    )
    return scene_id


def _write(world_dir: Path, built: dict, scene: dict, premise: str) -> Invented:
    """Write it out as the YAML an author would have written."""
    llm, facts = built["llm"], built["facts"]
    repaired: list[str] = []
    dropped: list[str] = []

    (world_dir / "characters").mkdir(parents=True, exist_ok=True)
    (world_dir / "scenes").mkdir(parents=True, exist_ok=True)

    rooms = {}
    for room_id, room in built["rooms"].items():
        description = _clean(room["description"], facts, llm, f"room {room_id}", repaired)
        if not description:
            dropped.append(f"the description of {room['name']}")
        rooms[room_id] = {
            "name": room["name"],
            "description": description,
            "adjacent": room["adjacent"],
        }

    world = {
        "id": world_dir.name,
        "title": built["title"],
        "blurb": built["blurb"],
        "rooms": rooms,
        "facts": {fact_id: {"keywords": fact["keywords"]} for fact_id, fact in facts.items()},
    }
    _save(world_dir / "world.yaml", world, f"Made from: {premise}")

    pressures = _pressures(built, repaired, dropped)
    if pressures:
        _save(world_dir / "pressures.yaml", pressures)
    intentions = _intentions(built, repaired, dropped)
    goals = _goals(built, repaired, dropped)

    for person_id, person in built["cast"].items():
        persona = person["persona"]
        # A persona may name what its own character protects and nothing
        # else: that is the difference between a character who knows a
        # secret and one who has been handed somebody else's.
        theirs = {f: facts[f] for f in facts if f not in person["protects"]}
        persona = _clean(persona, theirs, llm, f"{person_id}'s persona", repaired)
        if not persona:
            dropped.append(f"{person['name']}'s persona")
            persona = f"{person['name']} is here, and is not saying much about why."
        _save(
            world_dir / "characters" / f"{person_id}.yaml",
            {
                "id": person_id,
                "name": person["name"],
                "persona": persona,
                "traits": person["traits"],
                "protects": person["protects"],
                "goals": goals.get(person_id, []),
                "intentions": intentions.get(person_id, []),
                "relationships": person["relationships"],
                "location_id": person["location_id"],
                "is_user": person["is_user"],
            },
        )

    positions = _positions(scene.get("positions"), built)

    opening = _clean(str(scene.get("opening") or ""), facts, llm, "the opening", repaired)
    if not opening:
        dropped.append("the scene's opening")
    ends_on = _ends_on(scene, facts)
    title = " ".join(str(scene.get("title") or "").split())
    if not title or title.lower() in EXEMPLAR:
        # Same leak, in the scene's name: the 3B called its heist "The
        # Dinner", which is what the example's scene is called.
        title = built["title"]
    scene_id = slug(title) or "the_first_scene"
    onward = _after(
        world_dir, built, scene_id, positions, ends_on, premise, repaired, dropped
    )
    _save(
        world_dir / "scenes" / f"{scene_id}.yaml",
        {
            "id": scene_id,
            "world": world_dir.name,
            "title": title,
            "premise": " ".join(str(scene.get("premise") or built["blurb"]).split()),
            "opening": opening,
            "mode": "arc" if ends_on else "sandbox",
            "cast": list(built["cast"]),
            "starting_positions": positions,
            "start_time": BEGINS,
            "turn_budget": 6,
            "max_consecutive_agent_turns": 3,
            "end_condition": {"fact_spoken": ends_on} if ends_on else {},
            "next": onward,
        },
    )

    if built.get("quiet"):
        dropped.append("the complications — nothing happens on its own here")
    if not onward:
        dropped.append("everything after the first scene — the story stops when it ends")
    return Invented(
        world_dir=world_dir,
        title=built["title"],
        scene=scene_id,
        opening=opening,
        shape=built.get("shape", DEFAULT_SHAPE),
        repaired=repaired,
        dropped=dropped,
    )


def _save(path: Path, data, note: str = "") -> None:
    header = f"# {note}\n# Made by fabula, not by hand. Edit it like anything else.\n" if note else ""
    path.write_text(
        header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
