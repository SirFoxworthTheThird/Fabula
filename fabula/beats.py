"""What the room could use right now, and who decides.

The narrator used to be handed a triggering event and asked for "one or
two sentences of scene-setting narration". It knew what had just
happened and nothing about what the scene needed, so it fired rarely —
an arrival, a departure, five straight lines of dialogue — and when it
did fire it wrote whatever the last line suggested. A scene played that
way is people talking in a white room.

So the director says what to narrate. It is the only omniscient
component (spec §4) and its output has to stay narrow, which is the
whole design problem here: an instruction written freely from what the
director knows is a channel straight from the world log into prose every
character in the room perceives. "Narrate that Tomás is nervous about
the music box" would be a leak with extra steps.

Two rules keep it safe, and they are the same two this project keeps
arriving at:

* **A closed vocabulary.** A beat is one of a fixed set of ids, chosen
  by code from the event log and the world — never a sentence, and never
  a model's idea. Adding a beat means adding one here, in the open.

* **Only what the room can see.** Every input a beat carries — a name, an
  item, a room — is something anybody standing there perceives already.
  Nothing is read from beliefs, from trust, from what a character
  protects, or from anything that happened elsewhere.

What the narrator does with a beat is still prose, and prose is still
only as good as the model writing it. But it can no longer be pointed at
something the room does not know.
"""
from __future__ import annotations

from dataclasses import dataclass

from fabula.memory import location_at_seq
from fabula.models import Character, Event
from fabula.world import World

# Consecutive lines of talk before the room wants a beat of its own. Was
# five, which is a long time to go without the place existing.
LULL_WINDOW = 3
# How recently prose in this room disqualifies more of it. A lull needs
# three clean events of its own on top, so in practice the room gets a
# beat about every fourth event rather than whenever the talk runs on.
#
# Atmosphere only. An arrival, a departure, or a player alone in a room
# still gets its beat: those are moments the story would otherwise lose.
COOLDOWN = 3
# How long somebody can stand in a room saying nothing before it is worth
# putting them in the frame.
QUIET_FOR = 4


# The last resort. Nobody bid to speak and no beat was due, which would
# leave the turn producing nothing at all — the player types into a room
# with somebody standing in it and the app answers "No one answers." A
# room that never responds is not a story, so the room answers.
NOBODY_SPEAKS = "nobody_speaks"


@dataclass(frozen=True)
class Beat:
    """One thing the narrator could do next, and how much it wants to."""

    id: str
    desire: float
    reason: str
    # A name, an item, a room: always something already perceivable by
    # anybody standing there. Never a fact, a belief or a motive.
    subject: str = ""


def _spoke_recently(events: list[Event], character_id: str, window: int) -> bool:
    return any(e.actor_id == character_id for e in events[-window:])


def _mentioned(events: list[Event], text: str) -> bool:
    lowered = text.lower()
    return any(lowered in e.content.lower() for e in events)


def choose(
    event: Event,
    events: list[Event],
    world: World,
    characters: dict[str, Character] | None = None,
    protagonist_id: str | None = None,
    alone: bool = False,
) -> Beat | None:
    """The beat this moment wants, or None to leave the room alone."""
    if not events:
        return Beat("the_room", 0.9, "establish the scene")

    # Never gild prose that is already prose: a narration, or a pressure's
    # effect, which the narrator itself just rendered.
    if event.kind == "narration" or event.metadata.get("pressure_id"):
        return None

    settled = _recent_narration(events, event.location_id)

    # The player's own action. Never narrate them back at themselves: a
    # model told not to still answers a move with "Elena's light blue
    # dress caught the dim light", inventing a dress and playing the one
    # character the player controls. The room they walked into is not
    # them, and the silence after they speak is not them either.
    if protagonist_id and event.actor_id == protagonist_id:
        if event.kind == "arrival":
            return Beat("the_room", 0.6, "describe the room they walked into")
        if alone:
            # Nobody is here to answer, so refusing to narrate means the
            # turn produces nothing at all. A room that never responds is
            # not a story.
            return Beat("alone", 0.55, "nobody is here to answer them")
        return None if settled else _atmosphere(event, events, world, characters)

    if event.kind in ("arrival", "departure", "time_skip"):
        return Beat(event.kind, 0.7, f"describe the {event.kind}")

    if settled:
        return None

    # Somebody has just visibly not answered. The room notices a pause
    # like that, and it is the moment most worth holding on.
    if event.metadata.get("withheld"):
        return Beat("after_deflection", 0.5, "hold on the moment nobody answered")

    return _atmosphere(event, events, world, characters)


def _recent_narration(events: list[Event], room: str) -> bool:
    """Has this room had prose lately?

    Per room, because a paragraph in the study is not a paragraph the
    kitchen just read — and without the scene's own opening words, which
    are addressed to the player rather than to the room.
    """
    return any(
        e.kind == "narration" and e.location_id == room and not e.metadata.get("opening")
        for e in events[-COOLDOWN:]
    )


def _atmosphere(
    event: Event,
    events: list[Event],
    world: World,
    characters: dict[str, Character] | None,
) -> Beat | None:
    """The beats that are nobody's turn: the room, the people not talking
    in it, the things in it."""
    here = [e for e in events if e.location_id == event.location_id]

    # Somebody standing there who has not said anything for a while. That
    # they are quiet is observable by everyone in the room — it is what
    # they are quiet *about* that would not be, and a beat never carries
    # that.
    if characters:
        silent = [
            character
            for character in characters.values()
            if character.id != event.actor_id
            and not character.is_user
            # Where they are now, replayed from the log — not the
            # position the scene started them in. People move.
            and location_at_seq(
                character.id, character.location_id, events, event.seq + 1
            )
            == event.location_id
            and not _spoke_recently(here, character.id, QUIET_FOR)
        ]
        if silent and len(here) > QUIET_FOR:
            return Beat(
                "held_back", 0.45, "somebody in the room has not spoken", silent[0].name
            )

    talk = here[-LULL_WINDOW:]
    if len(talk) >= LULL_WINDOW and all(e.kind == "utterance" for e in talk):
        return Beat("lull", 0.4, "the scene has been nothing but dialogue")

    # Something in the room nobody has looked at yet. Items are placed by
    # the author and visible to anybody standing there.
    for item in world.items_in(event.location_id):
        if not _mentioned(events, item.name):
            return Beat("object", 0.3, "there is something here nobody has looked at", item.name)

    return None
