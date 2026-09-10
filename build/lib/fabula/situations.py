"""What happens next, when the author ran out of things that happen.

A world's pressures are finite. They have `max_fires`, and once those are
spent nothing in the story can ever happen again — measured, on
`ashgrove/the_dinner`: all three are gone by the seventh player line, and
every turn from the eighth to the twenty-fourth is identically two people
talking with a weather report between them. The scene does not break. It
goes slack, which is worse, because nothing reports it.

That is fine for a scene somebody wrote an ending for: it ends before it
runs dry. It is fatal for the way a lot of people actually want to play,
which is to start a story and stay in it — a childhood friend after
twenty years, a school year, a room you keep coming back to. There is no
ending to reach and no author to write the twentieth complication.

So when a story is being played that way and the room has been nothing
but talk for a while, the engine writes one.

**An invented situation is an ordinary `Pressure`.** Not a new event
kind, not a special case in the director, not a second path through
perception — it is the same object an author writes in `pressures.yaml`,
built here instead of read from disk, and from there `_fire` renders it
and the world model filters it exactly as it does the authored ones. The
whole feature is one function that returns a `Pressure`.

**What it is allowed to know is narrower than the director's.** The
director may be omniscient; this is not. It gets the rooms, who is
standing in them, and the lines people have said *out loud* — nothing
about facts, nobody's beliefs, nobody's protects. Not because invariant 4
demands it, but because the deterministic guard downstream can only catch
a secret's actual keywords, and a writer that had been told the secret
could paraphrase around them: "he looks at the empty space on the mantel"
names nothing and gives away everything. A writer that was never told
cannot allude to it.

Cost, decided here rather than afterwards: one model call, only when the
last several events were all people talking, and never twice inside a
cooldown. On the measured run that is four or five calls across
twenty-four player lines, against thirty-five for every three.
"""
from __future__ import annotations

import json
import re

from fabula.discovery import invents_a_fact
from fabula.steering import told
from fabula.llm import LLMClient, ModelUnavailable
from fabula.models import Character, Event, Pressure
from fabula.world import World

# How many things the player can say without anything happening around
# them before the engine steps in. Counted in *their* lines, not in log
# positions, because that is the unit the pacing is actually felt in: a
# scene where four rooms of people are talking racks up events fast and
# still feels like nothing is happening.
#
# Deliberately not "no narration for a while". Measured on the run this
# was built for, there is a narration every single turn once the
# pressures are spent — the atmosphere beats never run out. Atmosphere
# describes the room; it does not change anything in it. What runs out is
# events, so events are what this counts.
RESPITE = 4
# The room is still making its own tension. Same threshold sandbox
# pressures use, because it is the same judgement: do not push into a
# scene that is already going.
QUIET = 0.4
# How much an invented situation wants the turn. Deliberately modest: it
# competes with the people in the room, and it should lose to somebody
# who actually has something to say.
WEIGHT = 0.5
# How many spoken lines the writer is shown.
RECENT = 8
# What an invented pressure's id begins with, so the log can be read
# back and one of these told from anything an author wrote.
MADE = "invented_"

SYSTEM = """You decide what happens next in a story that is still going.

The people in the room have been talking for a while and nothing has happened around them. Say one thing that happens. Answer with one JSON object and nothing else, in the same shape as this one:

{"what": "The light in the hall goes off on its timer, and somebody will have to get up.", "where": "the hall"}

Rules that matter:
- Something that happens *to* the room: a noise, a door, the weather, a light, a phone, an object giving way, somebody moving about elsewhere in the building. Never somebody in the scene speaking, and never anybody's thoughts.
- It has to be possible in this place. Use the rooms you are given and nothing else.
- Never invent a person. The cast is who it is.
- Do not resolve anything, and do not answer whatever they were talking about. You are making the next moment harder, or stranger, or simply different — not ending it.
- One sentence, two at most."""


def drifting(events: list[Event], protagonist_id: str | None, top_bid: float) -> bool:
    """Has the story stopped happening and started only being discussed?

    Two conditions, both read off the log rather than off anybody's
    opinion of it. The room has gone quiet by the same measure the
    authored sandbox pressures use — so this never pushes into a scene
    that is already going — and nothing has happened around the player
    for several things they said.

    "Happened" means an event some pressure put there, authored or
    invented. Not narration: the atmosphere beats never run out, so a
    story can produce one every turn for ever while nothing whatsoever
    changes. That is exactly the state this exists to detect.
    """
    if top_bid >= QUIET:
        return False
    last = max((e.seq for e in events if e.metadata.get("pressure_id")), default=-1)
    said = sum(
        1
        for e in events
        if e.seq > last
        and e.kind == "utterance"
        and e.actor_id == protagonist_id
        and not e.metadata.get("opening")
    )
    return said >= RESPITE


def compose(
    llm: LLMClient,
    world: World,
    events: list[Event],
    present: dict[str, Character],
    here: str,
    steering: str = "",
) -> Pressure | None:
    """One thing that happens, or None if nothing usable came back.

    None is an ordinary answer, not a failure: the turn carries on
    exactly as it did before this existed. Nothing about a story should
    depend on a model having been in the mood.
    """
    rooms = ", ".join(room.name for room in world.rooms.values())
    who = ", ".join(person.name for person in present.values()) or "nobody"
    said = [
        f"{world.room_name(e.location_id)} — {present[e.actor_id].name if e.actor_id in present else 'someone'}: {e.content}"
        for e in events
        if e.kind == "utterance"
    ][-RECENT:]

    try:
        answered = llm.complete(
            system=SYSTEM + told(steering),
            prompt=(
                f"Rooms: {rooms}\n"
                f"In the room: {who}, in {world.room_name(here)}\n\n"
                "What has been said, oldest first:\n"
                + ("\n".join(said) or "nothing yet")
                + "\n\nWhat happens?"
            ),
            key="situation",
        )
    except ModelUnavailable:
        # The story is mid-turn and the model is somebody else's machine.
        # Losing an interruption is not worth losing the turn over.
        return None

    drafted = _json(answered)
    what = " ".join(str(drafted.get("what") or "").split())
    if not what:
        return None

    # The same guard the generated room descriptions get. A situation
    # naming a secret's own words would raise the subject as scenery —
    # satisfying `fact_spoken` before anybody in the story had said it.
    if invents_a_fact(what, world):
        return None

    where = _room(str(drafted.get("where") or ""), world) or here
    return Pressure(
        id=f"{MADE}{len(events)}",
        intent=what,
        weight=WEIGHT,
        trigger={},
        # Narration only, and nowhere but a room that exists. An arrival
        # would need somebody the scene said might turn up, and a state
        # change would need a scene written around it; neither is
        # something to decide mid-turn on a model's say-so.
        effect={"kind": "narration", "location": where},
        cooldown_turns=0,
        max_fires=1,
    )


def _room(named: str, world: World) -> str | None:
    wanted = named.strip().lower()
    if not wanted:
        return None
    for room_id, room in world.rooms.items():
        if wanted in (room_id.lower(), room.name.strip().lower()):
            return room_id
    # A room named loosely — "hall" for "the front hall" — rather than
    # not at all. Exact first, so a world with both is not surprised.
    for room_id, room in world.rooms.items():
        if wanted in room.name.strip().lower():
            return room_id
    return None


def _json(raw: str) -> dict:
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


def is_invented(pressure: Pressure) -> bool:
    """Did the engine write this one, or did somebody?"""
    return pressure.id.startswith(MADE)
