"""Cross-scene persistence (spec §10).

Characters are durable entities. Between scenes they carry their belief
store (aged and decayed), their unresolved goals, and their relationship
state toward every other character — including the user's. A character
who remembers last week's betrayal is the point; resetting them each
session is not acceptable.

Aging never consults world state. Invariant 2 says no component ever
corrects a character's beliefs against what is actually true, so decay
here is a pure function of a belief and its own salience.
"""
from __future__ import annotations

from fabula.db import EventStore
from fabula.models import Belief, Character, Goal, Relationship

BELIEF_SALIENCE_FLOOR = 0.5   # below this, a perceived moment is not worth encoding
DECAY_PER_SCENE = 0.85
MINIMUM_SALIENCE = 0.05


def encode_belief(store: EventStore, character: Character, belief: Belief | None) -> bool:
    """Store a belief if it was salient enough to be worth keeping.

    Not every perceived moment becomes a memory; the store would fill
    with the weather. What passes the floor is decided by the character's
    own salience bias, so two characters keep different things.
    """
    if belief is None or belief.salience < BELIEF_SALIENCE_FLOOR:
        return False
    store.add_belief(belief)
    return True


def age_beliefs(store: EventStore, character_id: str, factor: float = DECAY_PER_SCENE) -> None:
    """Decay a character's beliefs by one scene's worth of forgetting.

    Salient memories decay more slowly than unremarkable ones — the same
    principle as tier exemption in §6, applied across scene boundaries.
    """
    for belief in store.get_beliefs(character_id):
        if belief.id is None:
            continue
        decayed = belief.salience * (factor + (1.0 - factor) * belief.salience)
        store.set_belief_salience(belief.id, max(MINIMUM_SALIENCE, decayed))


def unresolved_goals(store: EventStore, character: Character) -> list[Goal]:
    resolved = store.get_resolved_goals(character.id)
    return [
        goal for goal in character.goals if goal.id not in resolved and not goal.resolved
    ]


def begin_scene(store: EventStore, characters: dict[str, Character]) -> None:
    """Bring durable state into a new scene.

    Seeding is idempotent, so authored starting values apply on a
    character's first appearance and never overwrite what play has since
    made of them. Anything already carrying beliefs has lived through a
    previous scene, so those beliefs age on the way in.
    """
    for character in characters.values():
        if store.get_beliefs(character.id):
            age_beliefs(store, character.id)

        for toward_id, relationship in character.relationships.items():
            store.seed_relationship(character.id, toward_id, relationship)

        # Everyone holds some stance toward everyone else, the user
        # included — an unauthored one simply starts neutral.
        for other_id in characters:
            if other_id != character.id:
                store.seed_relationship(character.id, other_id, Relationship())
