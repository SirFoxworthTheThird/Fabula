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


RESTRAINT = (
    "Introduce nothing that is not already established — no objects, food, weather, "
    "smells, times of day, or people that have not appeared. Describe the space and "
    "what is happening in it, not anyone's appearance or inner state."
)


class Narrator:
    def __init__(
        self,
        llm: LLMClient,
        protagonist: str | None = None,
        protagonist_id: str | None = None,
    ):
        self.llm = llm
        # The player's character. The narrator must never move, speak for,
        # or describe them: doing so takes the one character the player
        # controls and plays it for them.
        self.protagonist = protagonist
        self.protagonist_id = protagonist_id

    def _hands_off(self) -> str:
        if not self.protagonist:
            return ""
        return (
            f" {self.protagonist} is played by someone else: never describe what "
            f"{self.protagonist} does, says, feels, or looks like, and never give them "
            "an object or an action they did not take."
        )

    def bid(self, event: Event, events: list[Event], world: World) -> Bid | None:
        """Bids on a lull, an undescribed physical action, or a scene that
        needs establishing — never on ordinary dialogue exchange."""
        if not events:
            return Bid(character_id=NARRATOR_ID, desire=0.9, one_line_reason="establish the scene")
        # Never gild prose that is already prose: a narration, or a
        # pressure's effect, which the narrator itself just rendered.
        if event.kind == "narration" or event.metadata.get("pressure_id"):
            return None
        # Never narrate the player's own action back at them. Asking a
        # model not to is not enough — a weak one answers a move with
        # "Elena's light blue dress caught the dim light", inventing a
        # dress and playing the one character the player controls. The
        # only reliable version of this rule is not bidding.
        if self.protagonist_id and event.actor_id == self.protagonist_id:
            return None
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
            f"never state anything no one present could observe. {RESTRAINT}{self._hands_off()}"
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
            f"disturbed. Never narrate the act itself as if it were witnessed. {RESTRAINT}"
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
            "spare prose, at most two sentences. Describe only perceivable action in the "
            "given location. Never narrate a character's private thoughts, never state "
            "information no one present could observe, never resolve dialogue for a "
            f"character. {RESTRAINT}{self._hands_off()}"
        )
        prompt = (
            f"Location: {location_name}\n"
            f"Triggering event ({event.kind}): {event.content}\n"
            "Write one or two sentences of scene-setting narration."
        )
        return self.llm.complete(system=system, prompt=prompt, key=NARRATOR_ID)
