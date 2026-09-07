"""The narrator: describes perceivable action, bids on scene-setting
triggers rather than conversational ones. Its output is prose, but that
prose becomes an ordinary Event with a location and audibility — it is
filtered through the same world model as everything else, so narration
can never describe what an off-scene character couldn't perceive."""
from __future__ import annotations

from fabula.llm import LLMClient
from fabula.models import Bid, Character, Event, Pressure
from fabula.world import World

NARRATOR_ID = "__narrator__"


def _manner(character: Character) -> str:
    """How this person deflects, derived from mechanical traits alone —
    never from authored prose, which is where secrets live."""
    traits = character.traits
    if traits.reticence >= 0.7:
        return "guarded; deflects rather than refuses outright"
    if traits.talkativeness >= 0.6:
        return "normally talkative, which makes the pause conspicuous"
    return "reluctant, and not good at hiding it"


class Narrator:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def bid(self, event: Event, events: list[Event], world: World) -> Bid | None:
        """Bids on a lull, an undescribed physical action, or a scene that
        needs establishing — never on ordinary dialogue exchange."""
        if not events:
            return Bid(character_id=NARRATOR_ID, desire=0.9, one_line_reason="establish the scene")
        if event.kind in ("arrival", "departure", "time_skip"):
            return Bid(
                character_id=NARRATOR_ID,
                desire=0.7,
                one_line_reason=f"describe the {event.kind}",
            )
        return None

    def render_pressure(self, pressure: Pressure, location_id: str, world: World) -> str:
        """Turn a chosen pressure's authored intent into perceivable prose.

        The intent is director-only and never reaches a character's
        context; what characters get is this rendering, appended as an
        ordinary event and filtered by the world model like any other."""
        system = (
            "You are the narrator of an interactive story: third-person, present-tense, "
            "spare prose. You are given the beat that should happen now. Render it as "
            "perceivable action in the given location — what someone standing there would "
            "see or hear. Never explain the beat's purpose, never name it as a device, and "
            "never state anything no one present could observe."
        )
        prompt = (
            f"Location: {world.room_name(location_id)}\n"
            f"The beat: {pressure.intent}\n"
            "Write one or two sentences."
        )
        return self.llm.complete(system=system, prompt=prompt, key=f"pressure:{pressure.id}")

    def render_withholding(self, character: Character, world: World, location_id: str) -> str:
        """Render a character visibly declining to answer.

        It is never told *what* is being withheld, and must not be: the
        beat is perceived by everyone in the room, so a narrator that knew
        the secret could hand it to them in the act of describing it being
        kept.

        Note what is *not* passed: the persona. Authored persona prose is
        where a character's secret usually lives — Tomás's says outright
        what he broke — so handing it to a narrator whose output is public
        leaks the thing this whole beat exists to protect. Manner is
        derived from traits instead, which carry no world knowledge.
        """
        system = (
            "You are the narrator of an interactive story: third-person, present-tense, "
            "spare prose. A character has just been asked something they do not want to "
            "answer. Describe only what someone in the room would see them do instead — "
            "a deflection, a busied hand, a look away, a change of subject. Never state "
            "or hint at what they are avoiding, never explain why, and never give them "
            "dialogue that answers the question."
        )
        prompt = (
            f"Character: {character.name}\n"
            f"Manner: {_manner(character)}\n"
            f"Location: {world.room_name(location_id)}\n"
            "Write one sentence of them not answering."
        )
        return self.llm.complete(
            system=system, prompt=prompt, key=f"withhold:{character.id}"
        )

    def materialize(self, summary_event: Event, world: World) -> str:
        """Expand a coarsely-resolved off-screen action into what is
        visible now, in the room, to someone who has just walked in — not
        a replay of what happened while they were away."""
        system = (
            "You are the narrator of an interactive story: third-person, present-tense, "
            "spare prose. You are told, in one coarse line, something that happened in "
            "this room while no one was watching. Describe only the traces of it that "
            "are visible now to someone standing here — what was left, moved, or "
            "disturbed. Never narrate the act itself as if it were witnessed."
        )
        prompt = (
            f"Location: {world.room_name(summary_event.location_id)}\n"
            f"What happened here, unobserved: {summary_event.content}\n"
            "Describe what is visible now, in one or two sentences."
        )
        return self.llm.complete(
            system=system, prompt=prompt, key=f"materialize:{summary_event.id}"
        )

    def generate(self, event: Event, events: list[Event], world: World) -> str:
        location_name = world.room_name(event.location_id)
        system = (
            "You are the narrator of an interactive story: third-person, present-tense, "
            "spare prose. Describe only perceivable action in the given location. Never "
            "narrate a character's private thoughts, never state information no one "
            "present could observe, never resolve dialogue for a character."
        )
        prompt = (
            f"Location: {location_name}\n"
            f"Triggering event ({event.kind}): {event.content}\n"
            "Write one or two sentences of scene-setting narration."
        )
        return self.llm.complete(system=system, prompt=prompt, key=NARRATOR_ID)
