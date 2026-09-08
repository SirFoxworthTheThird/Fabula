"""What a character made of what they perceived.

A belief used to be an echo: the perceived line, stored verbatim. That
is provable but inert — nobody remembers a conversation as a transcript.
They remember what they took it to mean, and two people in the same room
take it to mean different things.

So a belief now carries both. `content` stays the verbatim projection —
the half the leak tests assert against, and the half that survives with
no model at all. `interpretation` is the reading, written by a model from
this character's own perceived lines and nothing else.

Three things keep that safe rather than making it the softest part of
the engine:

* It is built from `perceived_content`, exactly like a summary. The raw
  event is never read, so an interpretation cannot reintroduce what the
  projection excluded or blurred.
* It is checked before it is stored. An interpretation that names a
  world fact this character has no business knowing is thrown away and
  the belief keeps its echo. A memory is durable and crosses scenes, so
  a hallucination here would not be a bad line — it would be a false
  memory a character carries for good.
* It is computed once and keyed by a hash of the exact lines it was read
  from (the same rule summaries follow, spec §6 rule 5). A memory that
  was rewritten on every turn would drift underneath the character who
  holds it.
"""
from __future__ import annotations

from fabula.db import EventStore
from fabula.llm import LLMClient
from fabula.models import Character, ProjectedEvent
from fabula.summaries import span_key
from fabula.world import World, mentions_fact

# How much run-up the reading gets. A line alone is not interpretable —
# "he said nothing" means one thing after small talk and another after
# being asked where the music box went — but the window is small on
# purpose: it is one model call per remembered moment per character.
INTERPRETATION_WINDOW = 4


def invented_fact(
    text: str, window: list[ProjectedEvent], character: Character, world: World, store: EventStore
) -> str | None:
    """The id of a world fact the reading names that the character had no
    way to be thinking about, or None if it invented nothing.

    Keyword matching against authored facts, exactly as `mentions_fact`
    is used everywhere else: deterministic, inspectable, and never a
    model deciding whether a model behaved.
    """
    perceived = " ".join(p.perceived_content for p in window)
    remembered = " ".join(belief.content for belief in store.get_beliefs(character.id))
    for fact_id, fact in world.facts.items():
        if not mentions_fact(fact, text):
            continue
        if fact_id in character.protects:
            continue  # their own secret; they may think about it freely
        if mentions_fact(fact, perceived) or mentions_fact(fact, remembered):
            continue
        return fact_id
    return None


def interpret(
    character: Character,
    window: list[ProjectedEvent],
    world: World,
    store: EventStore,
    llm: LLMClient | None,
) -> str:
    """This character's reading of the moment at the end of `window`.

    Returns "" when there is nothing trustworthy to store — no model, an
    empty answer, or a reading that invented a fact. The caller keeps the
    verbatim echo in that case, which is why every path here can fail
    safely.
    """
    if llm is None or not window:
        return ""

    key = span_key(character.id, window)
    stored = store.get_interpretation(character.id, key)
    if stored is not None:
        return stored

    text = _read(character, window, llm).strip()
    if not text or invented_fact(text, window, character, world, store):
        # Store the refusal too, so a bad reading is not paid for twice.
        store.put_interpretation(character.id, key, "")
        return ""

    store.put_interpretation(character.id, key, text)
    return text


def _read(character: Character, window: list[ProjectedEvent], llm: LLMClient) -> str:
    lines = "\n".join(f"- {p.perceived_content}" for p in window)
    unclear = any(p.perception == "degraded" for p in window)
    system = (
        f"You write down what {character.name} privately made of a moment. "
        f"{character.persona}\n"
        "Everything you are given is what they personally perceived, and some of it "
        "may have been unclear to them. Write one sentence, third person, present "
        "tense, naming what they now think is going on. Never add a fact that is not "
        "in the lines given, never resolve something unclear into something specific, "
        "and never write what anyone else is thinking."
    )
    prompt = (
        f"What {character.name} perceived, oldest first:\n{lines}\n\n"
        + ("Some of it they did not catch clearly.\n" if unclear else "")
        + f"In one sentence: what does {character.name} now think is going on?"
    )
    return llm.complete(system=system, prompt=prompt, key=f"interpret:{character.id}")
