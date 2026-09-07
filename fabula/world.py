"""The world model: rooms, adjacency, and perception.

Zero model calls in this module. This is the code that makes invariant 1
("a character can never be given information their character could not
perceive") mechanically true rather than a matter of prompting.
"""
from __future__ import annotations

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


class Room(BaseModel):
    id: str
    name: str
    # room_id -> audibility of that connection (how well sound crosses it)
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


def resolve_perception(
    event: Event, character_id: str, character_location: str, world: World
) -> PerceptionLevel:
    """Deterministically decide what a character perceives of an event.

    This is the single choke point invariant 1 depends on: it never
    consults a model, only event fields and room topology.
    """
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
    a character who only partially perceives it. Never a model call."""
    template = DEGRADED_TEMPLATES.get(event.kind, "something happens nearby, unclear")
    return template.format(location=world.room_name(event.location_id))
