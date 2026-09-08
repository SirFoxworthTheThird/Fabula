"""The world model: rooms, adjacency, and perception.

Zero model calls in this module. This is the code that makes invariant 1
("a character can never be given information their character could not
perceive") mechanically true rather than a matter of prompting.
"""
from __future__ import annotations

import unicodedata

from pydantic import BaseModel, Field

from fabula.models import Audibility, Event, PerceptionLevel

# Room names carry their own article ("the kitchen"), so these must not
# supply a second one.
DEGRADED_TEMPLATES: dict[str, str] = {
    "utterance": "muffled voices from {location}, words unclear",
    "action": "sounds of movement from {location}",
    "narration": "something happening in {location}, out of view",
    "arrival": "a door somewhere near {location}",
    "departure": "footsteps fading from {location}",
    "time_skip": "time passes",
    "state_change": "a faint change felt from {location}",
}

DURATION_TEMPLATES: dict[str, str] = {
    "minute": "a minute",
    "minutes": "{n} minutes",
    "hour": "an hour",
    "hours": "{n} hours",
}

TIME_SKIP_TEMPLATES: dict[str, str] = {
    "awake": "({duration} pass, and you feel every one of them)",
    "asleep": "(a gap — you surface to find {duration} gone, unfelt)",
}

# Filtered out of lexical overlap, which drives rehearsal and retrieval.
# A default rather than a rule: a world in another language overrides it,
# and getting it wrong makes overlap noisier, never wrong.
STOPWORDS: frozenset[str] = frozenset(
    """
    about after again against because been before being between both could
    does doing during each from have having here into itself just more most
    only other over same should some such than that their them then there
    these they this those through under until very were what when where
    which while with would your yours
    """.split()
)


# What a client says in its own voice, next to the story rather than in
# it. Not story text, but it is read in the same breath — "You are in o
# quintal" is the same bug as an English descriptor, one layer out. The
# verbs (/go, /wait) are deliberately not here: those are commands, and a
# command that changes name per world is a command nobody can document.
CLIENT_TEMPLATES: dict[str, str] = {
    "you_are": "You are {name}, in {room}.",
    "now_in": "You are in {room}.",
    "no_such_room": "There is no {room} to go to.",
    "no_answer": "No one answers.",
    "nothing_changed": "Nothing here has changed since you last looked.",
    "nothing_pending": "Nothing is pending; time stays where it is.",
    "time_stays": "Time stays where it is.",
    "nothing_played": "Nothing has been played yet.",
    "again": "— again —",
    "ended": "— {scene} has reached its end. /reveal to see what you could not, or keep talking.",
    "let_time_pass": "Let {duration} pass? [y/N] ",
    "help": (
        "(/go <room>, /wait, /look, /again to replay the last moment, /quit. "
        "/reveal spoils the scene when you are done. Anything else you say aloud.)"
    ),
}


class Phrasing(BaseModel):
    """Every string the engine itself can put into a character's
    perception, and the words it ignores when judging what echoes what.

    Authored per world, because a scene in another language must not have
    English injected into what somebody hears. The defaults are English
    so the two worlds written before this existed keep working.
    """

    degraded: dict[str, str] = Field(default_factory=lambda: dict(DEGRADED_TEMPLATES))
    duration: dict[str, str] = Field(default_factory=lambda: dict(DURATION_TEMPLATES))
    time_skip: dict[str, str] = Field(default_factory=lambda: dict(TIME_SKIP_TEMPLATES))
    stopwords: frozenset[str] = STOPWORDS
    client: dict[str, str] = Field(default_factory=lambda: dict(CLIENT_TEMPLATES))

    def say(self, key: str, **fields: object) -> str:
        """One client line, in the world's words."""
        template = self.client.get(key) or CLIENT_TEMPLATES[key]
        return template.format(**fields)


class Room(BaseModel):
    id: str
    name: str
    # Authored grounding for scene-setting. Without it the narrator has
    # nothing to describe a room from and invents furniture; with it, the
    # improvisation is in the wording rather than the invention.
    description: str = ""
    # room_id -> a neighbouring room. Edges are directed: a room may list
    # a neighbour that does not list it back, which is how you author a
    # place you can listen into but not out of (the generator shed in
    # `winterlight`). Perception uses the presence of an edge and the
    # *event's* own audibility.
    #
    # The value is not read yet. It is meant to say how well sound
    # crosses that particular doorway, which would let an author seal one
    # without the one-way trick, but nothing consults it — see the rough
    # edges in the README before relying on it.
    adjacent: dict[str, Audibility] = Field(default_factory=dict)


class Fact(BaseModel):
    """A named thing the story can turn on, recognised in the log by
    keyword. Facts exist so pressures can trigger on what has and hasn't
    been said out loud; only the director reads them."""
    id: str
    keywords: list[str] = Field(default_factory=list)


class World(BaseModel):
    id: str
    rooms: dict[str, Room]
    facts: dict[str, Fact] = Field(default_factory=dict)
    # A BCP-47 tag, passed to the model so it writes in the world's
    # language rather than defaulting to the language of the prompts.
    language: str = "en"
    phrasing: Phrasing = Field(default_factory=Phrasing)

    def distance(self, from_room: str, to_room: str) -> int:
        """BFS distance in rooms. 0 = same room, 1 = adjacent, etc."""
        if from_room == to_room:
            return 0
        seen = {from_room}
        frontier = [from_room]
        dist = 0
        while frontier:
            dist += 1
            nxt = []
            for room_id in frontier:
                room = self.rooms.get(room_id)
                if room is None:
                    continue
                for neighbour in room.adjacent:
                    if neighbour == to_room:
                        return dist
                    if neighbour not in seen:
                        seen.add(neighbour)
                        nxt.append(neighbour)
            frontier = nxt
        return -1  # unreachable

    def room_name(self, room_id: str) -> str:
        room = self.rooms.get(room_id)
        return room.name if room else room_id


def write_in(language: str) -> str:
    """A line for a system prompt naming the language to write in.

    Empty for English, so the prompts the two English worlds have always
    used are unchanged — every prompt-level rule here was measured on
    them, and quietly adding a sentence would invalidate that.
    """
    if not language or language.split("-")[0].lower() == "en":
        return ""
    return (
        f"\nEverything you write is read by someone inside the story, and this story "
        f"is written in {language}. Write only in {language}."
    )


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def mentions_fact(fact: Fact, text: str) -> bool:
    """Does this text bring up the named fact? Keyword matching, so it is
    deterministic and inspectable — a model is never asked whether a
    subject came up."""
    folded = _fold(text)
    return any(_fold(keyword) in folded for keyword in fact.keywords)


def resolve_perception(
    event: Event, character_id: str, character_location: str, world: World
) -> PerceptionLevel:
    """Deterministically decide what a character perceives of an event.

    This is the single choke point invariant 1 depends on: it never
    consults a model, only event fields and room topology.
    """
    # You always know what you yourself did, wherever the bookkeeping says
    # you were standing at the time. Without this an actor perceives their
    # own arrival from the room they just left, i.e. degraded.
    if character_id is not None and character_id == event.actor_id:
        return "full"

    same_room = character_location == event.location_id

    if event.audibility == "private":
        addressed = character_id == event.actor_id or character_id in event.addressed_to
        return "full" if (addressed and same_room) else "none"

    if same_room:
        return "full"

    if event.audibility == "room":
        return "none"

    dist = world.distance(character_location, event.location_id)
    if dist == -1:
        return "none"

    if event.audibility == "adjacent":
        return "degraded" if dist == 1 else "none"

    if event.audibility == "building":
        return "degraded" if dist >= 1 else "none"

    return "none"


def degrade_content(event: Event, world: World) -> str:
    """Deterministic, template-based transformation of event content for
    a character who only partially perceives it. Never a model call.

    The templates come from the world, not from this module: what a
    character half-hears is story text, and story text is the author's.
    """
    template = world.phrasing.degraded.get(event.kind) or DEGRADED_TEMPLATES.get(
        event.kind, "something happens nearby, unclear"
    )
    return template.format(location=world.room_name(event.location_id))
