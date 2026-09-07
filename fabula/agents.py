"""Character agents: bid to speak (see fabula.bidding) and generate
utterances/actions from exactly their own projection."""
from __future__ import annotations

import unicodedata

from fabula.llm import LLMClient
from fabula.memory import ContextBuilder
from fabula.models import Character, Event


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def strip_name_prefix(text: str, name: str) -> str:
    """Drop a leading "Name:" the model added to its own line.

    Models routinely answer "Tomás: I'm fine" when asked to speak as
    Tomás, and clients render the speaker themselves — leaving it gives
    "Tomás: Tomás: I'm fine". Repeated because they sometimes do it twice.
    """
    first = _fold(name.split()[0])
    cleaned = text.strip()
    while True:
        head, separator, tail = cleaned.partition(":")
        if not separator or _fold(head.strip()) not in (first, _fold(name)):
            return cleaned
        cleaned = tail.strip()


def _guarded_subjects(character: Character, contexts: ContextBuilder) -> str:
    """What this character will not bring up of their own accord.

    `protects` already makes them deflect when asked. Nothing stopped
    them volunteering it unprompted, and in playtesting Tomás opened the
    scene by announcing he had been repairing the music box. Naming the
    subject in his own prompt is safe — it is his secret; he knows it.
    """
    subjects = [
        contexts.world.facts[fact_id].keywords[0]
        for fact_id in character.protects
        if fact_id in contexts.world.facts and contexts.world.facts[fact_id].keywords
    ]
    if not subjects:
        return ""
    return (
        f"\nYou guard this and never raise it yourself: {', '.join(subjects)}. "
        "If someone brings it up, you turn the conversation instead of answering."
    )


def _system_prompt(character: Character, contexts: ContextBuilder) -> str:
    return (
        f"You are {character.name}, played as a character in an interactive story, "
        "not narrating and not breaking character.\n"
        f"Persona: {character.persona}\n"
        "You only know what appears below under 'What you have perceived'. Never "
        "reveal, reference, or act on anything outside it, even if it would make a "
        "better line — you do not have access to it."
        f"{_guarded_subjects(character, contexts)}"
    )


def generate_utterance(
    character: Character,
    events: list[Event],
    contexts: ContextBuilder,
    llm: LLMClient,
) -> str:
    context = contexts.for_character(character, events)
    system = _system_prompt(character, contexts)
    prompt = (
        f"What you have perceived so far:\n{context}\n\n"
        f"Speak as {character.name}: one short line, in the first person, in your own "
        "voice. Say only the words you say aloud — no narration, no stage directions, "
        "and do not write your own name in front of them."
    )
    return strip_name_prefix(llm.complete(system=system, prompt=prompt, key=character.id), character.name)
