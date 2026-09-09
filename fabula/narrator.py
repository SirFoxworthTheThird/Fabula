"""The narrator: describes perceivable action, bids on scene-setting
triggers rather than conversational ones. Its output is prose, but that
prose becomes an ordinary Event with a location and audibility — it is
filtered through the same world model as everything else, so narration
can never describe what an off-scene character couldn't perceive."""
from __future__ import annotations

from fabula import beats
from fabula.llm import LLMClient
from fabula.models import Bid, Character, Event, Pressure
from fabula.world import Room, World, write_in

NARRATOR_ID = "__narrator__"

# Kept as a name other modules and tests import; the number lives with
# the beats now, and it came down from five — which is a long time for a
# scene to go without the place existing.
LULL_WINDOW = beats.LULL_WINDOW

# What each beat asks for. The director picks the id; this is the only
# place that turns one into words, and every line here is written to stay
# on what anybody standing in the room can see.
BEATS: dict[str, str] = {
    "lull": (
        "They have been talking and nothing has happened around them. Give the room "
        "a beat of its own: what the place is doing while they talk."
    ),
    "after_deflection": (
        "Somebody has just not answered. Hold on the moment — the pause itself and "
        "what the room does with it. Never say what anybody is thinking or hiding."
    ),
    "held_back": (
        "{subject} is standing there and has not said anything for a while. Put them "
        "in the frame: what their hands are doing, where they are looking. Never what "
        "they feel, and never what they are about to say."
    ),
    "object": (
        "There is {subject} in this room, and nobody has looked at it yet. Give it one "
        "beat."
    ),
    "nobody_speaks": (
        "Nobody answers. Describe the pause: what the room and the people standing in "
        "it are doing while nobody speaks. Never say what any of them is thinking."
    ),
    "alone": (
        "There is nobody here to answer. Describe what the room does with the silence."
    ),
    "arrival": "Describe the arrival.",
    "departure": "Describe the departure.",
    "time_skip": "Describe the room now that the time has moved.",
}


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


def as_bid(beat: beats.Beat | None) -> Bid | None:
    """A beat, as an offer the director can arbitrate over. Bid rationales
    stop at the director and never reach an event, so the reason a beat
    was chosen cannot reach a character's context."""
    if beat is None:
        return None
    return Bid(character_id=NARRATOR_ID, desire=beat.desire, one_line_reason=beat.reason)


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

    def bid(
        self, event: Event, events: list[Event], world: World, alone: bool = False
    ) -> Bid | None:
        """What the narrator would do with this moment, as an offer.

        The choosing is `beats.choose` — the director works out what the
        room could use, and this turns it into something to arbitrate
        over. Kept as a method so a caller with no cast to hand still
        gets the shape of the answer.
        """
        return as_bid(
            beats.choose(
                event, events, world, protagonist_id=self.protagonist_id, alone=alone
            )
        )

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
            f"never state anything no one present could observe. {RESTRAINT}{write_in(world.language)}{self._hands_off()}"
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
            + write_in(world.language)
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
            f"disturbed. Never narrate the act itself as if it were witnessed. {RESTRAINT}{write_in(world.language)}"
        )
        prompt = (
            f"Location: {world.room_name(summary_event.location_id)}\n"
            f"What happened here, unobserved: {summary_event.content}\n"
            "Describe what is visible now, in one or two sentences."
        )
        return self.llm.complete(
            system=system, prompt=prompt, key=f"materialize:{summary_event.id}"
        )

    def furnish(self, name: str, reached_from: Room, world: World) -> str:
        """Write what a newly discovered place is like.

        Grounding rather than narration: this is stored as the room's
        description and read later by `describe_place`, which is what
        stops the narrator improvising different furniture every time
        somebody walks in. It is checked before it is kept — a
        description naming one of the world's facts is thrown away.
        """
        system = (
            "You furnish a place in an interactive story. Two sentences, no more: "
            "what is in it, the light, the sound, what it feels like to stand in. "
            "Ordinary and specific. Introduce no people and no event — nothing is "
            "happening here yet, and nobody is in it."
            + write_in(world.language)
        )
        prompt = (
            f"The place: {name}\n"
            f"Reached from: {reached_from.name}"
            + (f" — {reached_from.description.strip()}" if reached_from.description else "")
            + "\n\nDescribe it."
        )
        return self.llm.complete(system=system, prompt=prompt, key=f"furnish:{name}")

    def describe_place(self, location_id: str, world: World) -> str:
        """The room itself, for when the player has just walked into it.

        Grounded in the room's authored description where there is one,
        so the improvisation is in the wording rather than the invention.
        The protagonist is not mentioned at all: their arrival has
        already been shown, and describing them is playing them.
        """
        room = world.rooms.get(location_id)
        grounding = f"\nWhat is here: {room.description}" if room and room.description else ""
        system = (
            "You are the narrator of an interactive story: third-person, present-tense, "
            "spare prose, at most two sentences. Describe the room itself — what is in "
            "it, the light, the sound, what it feels like to stand in. Introduce no "
            "people, and no object a character could pick up or refer to later. Do not "
            "describe anyone arriving; that has already been shown."
            + (
                f" Do not mention {self.protagonist} at all."
                if self.protagonist
                else ""
            )
            + write_in(world.language)
        )
        prompt = f"Room: {world.room_name(location_id)}{grounding}\nDescribe the room."
        return self.llm.complete(system=system, prompt=prompt, key=f"place:{location_id}")

    def generate(
        self,
        event: Event,
        events: list[Event],
        world: World,
        beat: beats.Beat | None = None,
        present: list[str] | None = None,
    ) -> str:
        """Write the beat the director asked for.

        With no beat it reads the last event and writes something about
        it, which is what it used to do everywhere and is still what a
        caller with nothing to say about the moment gets.
        """
        walked_in = (
            self.protagonist_id
            and event.actor_id == self.protagonist_id
            and event.kind == "arrival"
        )
        if (beat is not None and beat.id == "the_room") or (beat is None and walked_in):
            return self.describe_place(event.location_id, world)

        location_name = world.room_name(event.location_id)
        system = (
            "You are the narrator of an interactive story: third-person, present-tense, "
            "spare prose, at most two sentences. Describe only perceivable action in the "
            "given location. Never narrate a character's private thoughts, never state "
            "information no one present could observe, never resolve dialogue for a "
            f"character. {RESTRAINT}{write_in(world.language)}{self._hands_off()}"
        )
        asked = (
            BEATS[beat.id].format(subject=beat.subject)
            if beat is not None and beat.id in BEATS
            else "Write one or two sentences of scene-setting narration."
        )
        # Names only, and only of people standing in the room: what the
        # beat is allowed to know is what anybody there can see.
        who = f"Who is here: {', '.join(present)}\n" if present else ""
        prompt = (
            f"Location: {location_name}\n"
            f"{who}"
            f"The last thing that happened ({event.kind}): {event.content}\n\n"
            f"{asked}"
        )
        return self.llm.complete(system=system, prompt=prompt, key=NARRATOR_ID)
