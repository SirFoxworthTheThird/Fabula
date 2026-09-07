"""One playable scene, wired up.

The engine knows nothing about frontends, but every client needs the
same handful of operations — say something, move, let time pass, look
around — and every client needs them expressed as *the user character's
POV*, never as the raw log. That shared surface lives here, so the CLI
and the HTTP service are both thin layers and a POV rule fixed once is
fixed for both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from fabula.chronology import derive_skip_minutes
from fabula.db import EventStore
from fabula.director import Director
from fabula.llm import LLMClient, get_default_llm
from fabula.loader import Scene, load_pressures, load_scenario
from fabula.models import Character, Event, ProjectedEvent
from fabula.narrator import Narrator
from fabula.persistence import begin_scene
from fabula.world import World


class Session:
    def __init__(
        self,
        world: World,
        characters: dict[str, Character],
        scene: Scene,
        store: EventStore,
        director: Director,
        user_character: Character,
    ):
        self.world = world
        self.characters = characters
        self.scene = scene
        self.store = store
        self.director = director
        self.user_character = user_character

    @classmethod
    def open(
        cls,
        world_dir: Path,
        scene_name: str,
        db_path: str = ":memory:",
        llm: LLMClient | None = None,
    ) -> Session:
        world, characters, scene = load_scenario(world_dir, scene_name)
        pressures = load_pressures(world_dir)
        store = EventStore(db_path)
        llm = llm or get_default_llm()
        director = Director(store, world, characters, scene, Narrator(llm), llm, pressures)

        # Characters are durable: with a real db path they arrive carrying
        # what they already believe, aged by the time between scenes.
        begin_scene(store, characters)

        user_character = next(
            (
                characters[cid]
                for cid in scene.cast
                if cid in characters and characters[cid].is_user
            ),
            None,
        )
        if user_character is None:
            raise ValueError("scene has no user-controlled character (is_user: true) in its cast")

        return cls(world, characters, scene, store, director, user_character)

    def here(self) -> str:
        return self.director.current_location(self.user_character)

    def pov(self, events: list[Event]) -> list[ProjectedEvent]:
        """What the user's character perceived of these events.

        The full log is projected (perception depends on where they were
        at the time), then narrowed to the events asked about. Anything
        they could not perceive is simply absent from the result — it is
        never included and marked, because a client that receives it can
        leak it.
        """
        all_events = self.store.get_events(self.scene.id)
        seqs = {event.seq for event in events}
        return [
            projected
            for projected in self.director.contexts.project(self.user_character, all_events)
            if projected.event.seq in seqs
        ]

    def perceived_so_far(self) -> list[ProjectedEvent]:
        """The whole scene as this character experienced it — what a
        client needs to catch up on reconnect."""
        return self.director.contexts.project(
            self.user_character, self.store.get_events(self.scene.id)
        )

    def say(self, text: str) -> list[ProjectedEvent]:
        """The whole turn as this character perceived it, their own line
        included. Trimming the echo is a presentation choice, so it
        belongs to the client — a session that withheld a perceived event
        would hand different transcripts to different clients."""
        event = self.director.build_event("utterance", self.user_character.id, self.here(), text)
        return self.pov(self.director.run_turn(event))

    def move(self, room_id: str) -> list[ProjectedEvent]:
        if room_id not in self.world.rooms:
            raise ValueError(f"there is no room {room_id!r} in this world")
        here = self.here()
        if room_id == here:
            return []
        arrival = self.director.build_event(
            "arrival",
            self.user_character.id,
            room_id,
            f"{self.user_character.name} comes in from {self.world.room_name(here)}.",
            audibility="adjacent",
        )
        return self.pov(self.director.run_turn(arrival))

    def pending_skip(self) -> int | None:
        """How long the next derived skip would be, so a client can ask
        the user before spending their character's time."""
        return derive_skip_minutes(self.characters, self.store.get_events(self.scene.id))

    def wait(self, consent: Callable[[int], bool] | None = None) -> list[ProjectedEvent]:
        return self.pov(self.director.advance_time(consent=consent))

    def look(self) -> list[ProjectedEvent]:
        """Take in the room: coarse off-screen events here expand into
        what is visible now."""
        revealed = []
        for summary in self.director.unmaterialized_here(self.user_character):
            event = self.director.materialize(summary, self.user_character)
            if event is not None:
                revealed.append(event)
        return self.pov(revealed)
