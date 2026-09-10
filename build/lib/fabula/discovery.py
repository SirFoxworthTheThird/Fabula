"""Rooms the author did not write.

A school has corridors. Nobody wants to write them all, and a story that
answers "there is no library to go to" is answering with its own
scaffolding. So the map can grow as somebody walks into it.

What makes this cheap is that almost none of a room carries any weight.
The one part that does is its **adjacency**, and the engine decides that
— a newly found place hangs off exactly the room it was reached from, by
one symmetric edge, so it behaves like any other doorway. A model that
got to choose edges could join the library to the headmaster's office,
which would be a leak rather than a bad sentence.

So the model's whole job here is a name and two lines of description:
prose, and prose is checked the same way every other generated line is —
a description naming a fact this world turns on is thrown away, because
a room description reaches the log as narration and would let the
narrator raise the subject just by describing the place.

The map that results is a tree, not a map: you can always go back the way
you came, and two discovered wings never join up. That is a real limit
rather than a bug, and it is the price of the engine keeping the edges.
"""
from __future__ import annotations

import re
import unicodedata

from fabula.db import EventStore
from fabula.world import Room, World, connect, mentions_fact


def slug(name: str) -> str:
    """A stable id from what somebody typed. Unicode-aware, so a room can
    be found in any script; falls back to a hash of the name when a script
    has nothing this can keep (CJK has no spaces and no case)."""
    folded = unicodedata.normalize("NFKD", name).casefold()
    stripped = "".join(c for c in folded if not unicodedata.combining(c))
    cleaned = re.sub(r"[^\w]+", "_", stripped, flags=re.UNICODE).strip("_")
    return cleaned or f"room_{abs(hash(name)) % 10**8}"


def invents_a_fact(description: str, world: World) -> str | None:
    """The id of a world fact this description names, if any.

    A room description reaches the log as narration, so a keyword in one
    lets the narrator raise the subject just by describing the room —
    firing pressures and ending arcs nobody spoke about. Authored worlds
    are guarded against this in review; a generated description has to be
    guarded here.
    """
    for fact_id, fact in world.facts.items():
        if mentions_fact(fact, description):
            return fact_id
    return None


def restore(world: World, store: EventStore) -> None:
    """Put back everything found on an earlier visit.

    Called when a scene opens, so a corridor found in the first scene of
    a story is still there in the third — and still joined to the same
    place, because the edge is stored rather than guessed again.
    """
    for row in store.get_discovered_rooms(world.id):
        if row["room_id"] in world.rooms or row["reached_from"] not in world.rooms:
            continue
        connect(
            world,
            Room(id=row["room_id"], name=row["name"], description=row["description"]),
            row["reached_from"],
        )


def discover(
    world: World, wanted: str, from_room_id: str, store: EventStore, narrator=None
) -> Room | None:
    """Bring a room into being because somebody walked into it.

    Returns None when the world has not opted in, so a two-room house
    goes on saying there is no library.
    """
    if not world.discover_rooms:
        return None
    name = wanted.strip()
    if not name or from_room_id not in world.rooms:
        return None

    room_id = slug(name)
    if room_id in world.rooms:
        return world.rooms[room_id]

    description = ""
    if narrator is not None:
        written = narrator.furnish(name, world.rooms[from_room_id], world).strip()
        # A description that names something this world turns on is worse
        # than no description: the room stands, bare, and the narrator
        # improvises from the name alone.
        if written and invents_a_fact(written, world) is None:
            description = written

    room = connect(world, Room(id=room_id, name=name, description=description), from_room_id)
    store.add_discovered_room(world.id, room_id, name, description, from_room_id)
    return room
