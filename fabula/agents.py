"""Character agents: bid to speak (see fabula.bidding) and generate
utterances/actions from exactly their own projection."""
from __future__ import annotations

from fabula.llm import LLMClient
from fabula.memory import assemble_context, project
from fabula.models import Character, Event
from fabula.world import World


def generate_utterance(
    character: Character, events: list[Event], world: World, llm: LLMClient
) -> str:
    projected = project(character, events, world, initial_location=character.location_id)
    context = assemble_context(character, projected)
    system = (
        f"You are {character.name}, played as a character in an interactive story, "
        "not narrating and not breaking character.\n"
        f"Persona: {character.persona}\n"
        "You only know what appears below under 'What you have perceived'. Never "
        "reveal, reference, or act on anything outside it, even if it would make a "
        "better line — you do not have access to it."
    )
    prompt = (
        f"What you have perceived so far:\n{context}\n\n"
        f"Respond in character as {character.name}: one short beat of dialogue or action."
    )
    return llm.complete(system=system, prompt=prompt, key=character.id)
