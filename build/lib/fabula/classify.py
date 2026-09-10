"""Reading a line for what it says happened, not for what it says.

Some sentences report a fact about the world rather than only adding to
it. "I told my brother last week" is a claim that a perception happened
off-screen — and if the engine ignores it, a character who walks in
later arrives ignorant of something the scene established.

Keyword lists cannot catch this. A *thing* has a name, so `mentions_fact`
works in any language; a *meaning* does not, and asking an author to
enumerate the ways of saying "I told him" is asking them to enumerate a
language. So this is a model's judgement — but the model's judgement is
confined to picking from a closed set, and everything it names is
checked against authored data before anything is appended.

**The model proposes, the engine disposes.** The classifier returns a
structure, never prose, and every field of it has to survive:

* `from` is whoever actually spoke the line, not the model's choice
* `fact` must be an id the author wrote — a fact cannot be invented here
* `to` must be a character in the scene or waiting to enter it
* the speaker must already know the fact themselves

A hallucinated classification fails one of those and becomes a no-op.
That is the same shape as the guard on interpretations: the softest
component in the engine gets to suggest, never to decide.

What it costs, stated plainly: a misclassification grants somebody
knowledge that nobody on screen conveyed. The backdated event makes that
formally a perception, so invariant 1 holds on paper, but it was created
on a model's say-so. Three things bound it — you can only pass on what
you know, nothing is hidden (the beat is in the log and in the reveal),
and the turn it happened in can be taken again.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from fabula.db import EventStore
from fabula.llm import LLMClient
from fabula.models import Character, Event, ProjectedEvent
from fabula.world import Fact, World, mentions_fact


@dataclass
class Transmission:
    """A validated report that somebody was told something off-screen."""

    speaker_id: str
    recipient_id: str
    fact_id: str


def worth_classifying(event: Event, world: World) -> bool:
    """The gate, and it is the cheap deterministic one.

    A line that names none of the world's facts cannot be reporting that
    somebody was told one, so almost every line skips the model call
    entirely. Keyword matching, so this costs nothing and works in any
    script.
    """
    return event.kind == "utterance" and any(
        mentions_fact(fact, event.content) for fact in world.facts.values()
    )


def knows(
    character: Character, fact: Fact, projected: list[ProjectedEvent], store: EventStore
) -> bool:
    """Is this fact the character's to pass on?

    Their own authored secret, something they have perceived said, or
    something they already remember. Checked against belief *content*,
    which is verbatim projection, so an earlier reading's invention can
    never satisfy it. The mirror of `interpret.invented_fact`.
    """
    if fact.id in character.protects:
        return True
    if any(
        perceived.perception == "full" and mentions_fact(fact, perceived.perceived_content)
        for perceived in projected
    ):
        return True
    return any(
        mentions_fact(fact, belief.content) for belief in store.get_beliefs(character.id)
    )


def _parse(raw: str) -> dict:
    """Best-effort JSON out of whatever came back.

    Anything malformed is 'no', not an error: a classifier that crashed
    the turn would be worse than one that missed a beat, and FakeLLM's
    placeholder text lands here every time.
    """
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def classify(
    event: Event,
    speaker: Character,
    candidates: dict[str, Character],
    world: World,
    projected: list[ProjectedEvent],
    store: EventStore,
    llm: LLMClient | None,
) -> Transmission | None:
    """Ask whether this line reports telling somebody something, and
    return it only if every referent survives checking."""
    if llm is None or not worth_classifying(event, world):
        return None

    people = {cid: c.name for cid, c in candidates.items() if cid != speaker.id}
    if not people:
        return None

    answer = _parse(_ask(event, speaker, people, world, llm))
    if not answer.get("reports_telling"):
        return None

    recipient_id, fact_id = answer.get("to"), answer.get("fact")
    if recipient_id not in people or fact_id not in world.facts:
        return None  # the model named somebody or something that is not there
    if not knows(speaker, world.facts[fact_id], projected, store):
        return None  # you cannot pass on what you do not know

    return Transmission(speaker_id=speaker.id, recipient_id=recipient_id, fact_id=fact_id)


def _ask(
    event: Event, speaker: Character, people: dict[str, str], world: World, llm: LLMClient
) -> str:
    roster = ", ".join(f"{cid} ({name})" for cid, name in people.items())
    subjects = ", ".join(
        f"{fid} ({fact.keywords[0]})" for fid, fact in world.facts.items() if fact.keywords
    )
    system = (
        "You label one line of dialogue and answer with JSON only.\n"
        "The question is narrow: does the speaker say that they ALREADY TOLD one of "
        "these people about one of these subjects, at some earlier time? Wanting to "
        "tell them, threatening to, wondering whether they know, or asking about it "
        "are all no. Only a report of having already told them is yes.\n"
        'Answer {"reports_telling": false} or '
        '{"reports_telling": true, "to": "<person id>", "fact": "<subject id>"}. '
        "Use only the ids given."
    )
    prompt = (
        f"People: {roster}\n"
        f"Subjects: {subjects}\n"
        f"{speaker.name} says: {event.content}\n\n"
        "JSON:"
    )
    return llm.complete(system=system, prompt=prompt, key=f"classify:{speaker.id}")
