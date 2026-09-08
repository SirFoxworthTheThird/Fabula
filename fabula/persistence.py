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

from typing import Callable

from fabula.db import EventStore
from fabula.models import Belief, Character, Goal, ProjectedEvent, Relationship
from fabula.world import World, mentions_fact

BELIEF_SALIENCE_FLOOR = 0.5   # below this, a perceived moment is not worth encoding
DECAY_PER_SCENE = 0.85
MINIMUM_SALIENCE = 0.05

# Watching somebody visibly decline to answer costs them something. The
# move is a fraction of the distance to a floor rather than a fixed step,
# so an evening full of deflection erodes trust without ever spending it:
# a character who trusts you at zero has nothing left to lose by lying,
# which is the least interesting place to leave them.
GOALS_IN_CONTEXT = 3      # the most pressing few; a list is not a character
WITHHOLD_TRUST_FLOOR = 0.1
WITHHOLD_TRUST_MOVE = 0.2


def encode_belief(
    store: EventStore,
    character: Character,
    belief: Belief | None,
    interpret: Callable[[], str] | None = None,
) -> bool:
    """Store a belief if it was salient enough to be worth keeping.

    Not every perceived moment becomes a memory; the store would fill
    with the weather. What passes the floor is decided by the character's
    own salience bias, so two characters keep different things.

    `interpret` is called only once a belief has cleared the floor, and
    only then — reading a moment costs a model call, and paying for one
    on something about to be discarded is money spent on the weather.
    """
    if belief is None or belief.salience < BELIEF_SALIENCE_FLOOR:
        return False
    if interpret is not None:
        belief.interpretation = interpret()
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


def witnessed_withholding(
    store: EventStore, witness_id: str, projected: ProjectedEvent
) -> bool:
    """Move a witness's trust when they *saw* somebody refuse to answer.

    Deterministic, and read from the witness's own projection rather than
    from the event log. A character who was not in the room loses no
    trust in anybody, and neither does one who only half-heard it: the
    degraded descriptor never carries a name, so they cannot know who
    that was. Invariant 1 is not only about sentences — a number that
    moved on an unperceived event is a channel too, and a slow drift in
    who bids after whom would carry real information about a room the
    character was never in.

    Returns whether anything moved, which is what the tests assert on.
    """
    event = projected.event
    actor_id = event.actor_id
    if not event.metadata.get("withheld") or projected.perception != "full":
        return False
    if not actor_id or actor_id == witness_id:
        # You do not lose faith in yourself for keeping your own counsel.
        return False

    relationship = store.get_relationships(witness_id).get(actor_id)
    if relationship is None or relationship.trust <= WITHHOLD_TRUST_FLOOR:
        # Already at the floor. Someone who has stopped believing you
        # cannot be made to stop harder.
        return False

    distance = relationship.trust - WITHHOLD_TRUST_FLOOR
    store.set_trust(
        witness_id, actor_id, WITHHOLD_TRUST_FLOOR + distance * (1.0 - WITHHOLD_TRUST_MOVE)
    )
    return True


def unresolved_goals(store: EventStore, character: Character) -> list[Goal]:
    """What this character is still trying to do, most pressing first."""
    resolved = store.get_resolved_goals(character.id)
    return sorted(
        (goal for goal in character.goals if goal.id not in resolved and not goal.resolved),
        key=lambda goal: -goal.priority,
    )


def wants(store: EventStore, character: Character, limit: int = GOALS_IN_CONTEXT) -> str:
    """The character's own wants, for their own prompt.

    Authored inner life, like the persona — it describes them rather than
    the room, so it carries no risk of telling them something they could
    not perceive. It is only ever assembled for the character it belongs
    to, and the narrator is never given it: what somebody wants is not
    something the room can see.
    """
    open_goals = unresolved_goals(store, character)[:limit]
    if not open_goals:
        return ""
    return "\nWhat you are still trying to do: " + "; ".join(
        goal.description for goal in open_goals
    )


def close_reached_goals(
    store: EventStore, character: Character, projected: list[ProjectedEvent], world: World
) -> list[str]:
    """Close any goal whose subject this character has now heard raised.

    Judged from their own projection, not the log: a secret that came out
    in a room they were not in has not stopped being a secret *to them*,
    and they go on guarding it. Degraded perception does not count either
    — the descriptor carries no words, so they did not catch the subject.

    Only goals that name a fact can close this way. A goal written as
    prose alone stays open, which is honest: nothing here can read
    "sort out grandmother's belongings fairly" and judge it done.
    """
    closed = []
    for goal in unresolved_goals(store, character):
        fact = world.facts.get(goal.about) if goal.about else None
        if fact is None:
            continue
        if any(
            perceived.perception == "full" and mentions_fact(fact, perceived.perceived_content)
            for perceived in projected
        ):
            store.resolve_goal(character.id, goal.id)
            closed.append(goal.id)
    return closed


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
