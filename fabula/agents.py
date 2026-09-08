"""Character agents: bid to speak (see fabula.bidding) and generate
utterances/actions from exactly their own projection."""
from __future__ import annotations

import unicodedata

from fabula.llm import LLMClient
from fabula.memory import ContextBuilder, co_present
from fabula.models import Character, Event
from fabula.persistence import wants
from fabula.world import write_in


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


def _guarded_subjects(
    character: Character, contexts: ContextBuilder, events: list[Event]
) -> str:
    """What this character guards, and who is currently in earshot.

    Deliberately a reminder and not a rule. The engine governs what a
    character *knows*; what they say is theirs to get wrong, and a
    character who is mechanically incapable of slipping is both less
    believable and less interesting — a secret that cannot escape by
    accident can only ever come out by authorial fiat, which is most of
    the tension gone. So this names the subject and who is present, and
    leaves the judgment where it belongs.

    Naming the subject in this character's own prompt is safe: it is
    their secret, and they know it.

    Measured, and unproven. Over 16 scene openings on a 1.5B local model,
    Tomás named the music box unprompted in 4/16 with this line, 3/16
    with no line at all, and 3/16 with a variant that withheld the
    subject's name — indistinguishable. That model ignores most prompt
    discipline (it also invents props and plays the protagonist), so this
    says little about a capable one; it does mean nobody should claim
    this works without measuring it again where it matters. The parts
    that hold regardless of model are mechanical: the withholding bid,
    the presence line, and the narrator's refusal to bid.
    """
    subjects = [
        contexts.world.facts[fact_id].keywords[0]
        for fact_id in character.protects
        if fact_id in contexts.world.facts and contexts.world.facts[fact_id].keywords
    ]
    if not subjects:
        return ""

    company = sorted(other.name for other in co_present(character, contexts.characters, events))
    who = (
        f" In the room with you: {', '.join(company)}. Weigh who can hear you before you "
        "speak of it."
        if company
        else " There is no one else here."
    )
    return (
        f"\nYou guard this and do not raise it lightly: {', '.join(subjects)}.{who} "
        "If someone else brings it up, you turn the conversation rather than answer."
    )


def _system_prompt(
    character: Character, contexts: ContextBuilder, events: list[Event]
) -> str:
    return (
        f"You are {character.name}, played as a character in an interactive story, "
        "not narrating and not breaking character.\n"
        f"Persona: {character.persona}\n"
        "You only know what appears below under 'What you have perceived'. Never "
        "reveal, reference, or act on anything outside it, even if it would make a "
        "better line — you do not have access to it."
        f"{_guarded_subjects(character, contexts, events)}"
        f"{wants(contexts.store, character)}"
        f"{write_in(contexts.world.language)}"
    )


def generate_utterance(
    character: Character,
    events: list[Event],
    contexts: ContextBuilder,
    llm: LLMClient,
) -> str:
    context = contexts.for_character(character, events)
    system = _system_prompt(character, contexts, events)
    prompt = (
        f"What you have perceived so far:\n{context}\n\n"
        f"Speak as {character.name}: one short line, in the first person, in your own "
        "voice. Say only the words you say aloud — no narration, no stage directions, "
        "and do not write your own name in front of them."
    )
    return strip_name_prefix(llm.complete(system=system, prompt=prompt, key=character.id), character.name)
