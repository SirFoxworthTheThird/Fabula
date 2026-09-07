"""The post-scene reveal: what you did not know while you were playing.

Information asymmetry is invisible by construction. A character not
mentioning something looks exactly like a character with nothing to
mention, and a scene where the secret held is indistinguishable — from
the inside — from a scene with no secret in it. The payoff has to come
afterwards, which is what this is: the moment the curtain goes up and
you find out what everyone else thought was happening.

Two rules make this safe rather than a hole in invariant 1:

* It lives here rather than on `Session`. Everything on that object is
  the player's point of view, with no exceptions; this deliberately
  steps outside one, and keeping it out of `Session` means nobody
  reaches for it by accident.
* It is read-only. Nothing here appends an event or touches a belief, so
  asking for a reveal cannot contaminate the scene it describes. Look
  behind the curtain and keep playing, and the characters know exactly
  what they knew before.

It is still a spoiler. A client must only produce one when the player
asks for it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fabula.models import Character, Event
from fabula.session import Session
from fabula.world import mentions_fact


@dataclass
class Reveal:
    character: Character
    # Things that happened which they never perceived at all.
    missed: list[Event] = field(default_factory=list)
    # (what they heard, what it actually was)
    half_heard: list[tuple[str, Event]] = field(default_factory=list)
    # fact id -> character name -> did they know it by the end
    knowledge: dict[str, dict[str, bool]] = field(default_factory=dict)


def knows_fact(session: Session, character: Character, fact_id: str) -> bool:
    """Did this character end the scene knowing the fact?

    Either they were authored holding it — `protects` names a secret of
    their own — or they perceived it said. Checked against *perceived*
    content, so someone who only half-heard the moment it came up does
    not count as knowing: the degraded descriptor never carries the words.
    """
    if fact_id in character.protects:
        return True
    fact = session.world.facts.get(fact_id)
    if fact is None:
        return False
    events = session.store.get_events(session.scene.id)
    return any(
        mentions_fact(fact, projected.perceived_content)
        for projected in session.director.contexts.project(character, events)
    )


def build_reveal(session: Session) -> Reveal:
    events = session.store.get_events(session.scene.id)
    perceived = session.perceived_so_far()
    seen = {p.event.seq for p in perceived}
    you = session.user_character

    return Reveal(
        character=you,
        missed=[e for e in events if e.seq not in seen and e.actor_id != you.id],
        half_heard=[
            (p.perceived_content, p.event)
            for p in perceived
            # A time skip is "degraded" for anyone outside the room it is
            # filed in, but its content is written per character on
            # purpose — asleep or awake — not muffled through a wall.
            # Nobody mishears an hour passing.
            if p.perception == "degraded" and p.event.kind != "time_skip"
        ],
        knowledge={
            fact_id: {
                character.name: knows_fact(session, character, fact_id)
                for character in session.characters.values()
            }
            for fact_id in session.world.facts
        },
    )


def render(reveal: Reveal, session: Session) -> str:
    you = reveal.character
    lines = [f"=== what {you.name} never knew ==="]

    if reveal.missed:
        lines.append("\nHappened without you:")
        for event in reveal.missed:
            where = session.world.room_name(event.location_id)
            actor = session.characters.get(event.actor_id) if event.actor_id else None
            who = f"{actor.name}: " if actor else ""
            lines.append(f"  in {where} — {who}{event.content}")
    else:
        lines.append("\nNothing happened out of your sight.")

    if reveal.half_heard:
        lines.append("\nWhat you half-heard:")
        for heard, event in reveal.half_heard:
            actor = session.characters.get(event.actor_id) if event.actor_id else None
            who = f"{actor.name}: " if actor else ""
            lines.append(f"  you got — {heard}")
            lines.append(f"  it was  — {who}{event.content}")

    for fact_id, holders in reveal.knowledge.items():
        knew = sorted(name for name, yes in holders.items() if yes)
        did_not = sorted(name for name, yes in holders.items() if not yes)
        lines.append(f"\nWho knew about the {fact_id.replace('_', ' ')}:")
        lines.append(f"  knew        — {', '.join(knew) if knew else 'no one'}")
        lines.append(f"  never found out — {', '.join(did_not) if did_not else 'no one'}")

    return "\n".join(lines)


def reveal_text(session: Session) -> str:
    return render(build_reveal(session), session)
