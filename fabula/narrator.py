"""The narrator: describes perceivable action, bids on scene-setting
triggers rather than conversational ones. Its output is prose, but that
prose becomes an ordinary Event with a location and audibility — it is
filtered through the same world model as everything else, so narration
can never describe what an off-scene character couldn't perceive."""
from __future__ import annotations

from fabula.llm import LLMClient
from fabula.models import Bid, Event
from fabula.world import World

NARRATOR_ID = "__narrator__"


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
