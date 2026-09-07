"""M0 CLI: loads a scene from YAML, runs the turn loop, prints to stdout.
The user plays a character via stdin.

No frontend assumptions live here beyond argv/stdin/stdout — the engine
(director, world, memory) knows nothing about this file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fabula.db import EventStore
from fabula.director import Director
from fabula.llm import get_default_llm
from fabula.loader import load_pressures, load_scenario
from fabula.models import Character, Event
from fabula.narrator import Narrator


def _format_event(event: Event, characters: dict[str, Character]) -> str:
    # Only an utterance is somebody speaking. Arrivals, departures and the
    # rest carry narrator-rendered prose about a character, not words from
    # their mouth, so they must not be printed behind a speaker's name.
    if event.kind != "utterance":
        return f"  {event.content}"
    actor = characters.get(event.actor_id) if event.actor_id else None
    name = actor.name if actor else (event.actor_id or "???")
    return f"{name}: {event.content}"


def run(world_dir: Path, scene_name: str, db_path: str = ":memory:") -> None:
    world, characters, scene = load_scenario(world_dir, scene_name)
    pressures = load_pressures(world_dir)
    store = EventStore(db_path)
    llm = get_default_llm()
    narrator = Narrator(llm)
    director = Director(store, world, characters, scene, narrator, llm, pressures)

    user_characters = [
        characters[cid] for cid in scene.cast if cid in characters and characters[cid].is_user
    ]
    if not user_characters:
        raise SystemExit("scene has no user-controlled character (is_user: true) in its cast")
    user_character = user_characters[0]

    print(f"--- {scene.id} ({scene.mode}) ---")
    print(f"You are {user_character.name}, in {world.room_name(user_character.location_id)}.")
    print("(Ctrl-D or /quit to leave the scene.)\n")

    while True:
        try:
            raw = input(f"{user_character.name}> ")
        except EOFError:
            print()
            break
        raw = raw.strip()
        if not raw:
            continue
        if raw in ("/quit", "/exit"):
            break

        user_event = director.build_event(
            "utterance", user_character.id, user_character.location_id, raw
        )
        turn_events = director.run_turn(user_event)
        for event in turn_events[1:]:
            print(_format_event(event, characters))
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fabula", description="Run a Fabula scene from YAML.")
    parser.add_argument("world_dir", type=Path, help="Path to worlds/<name>/")
    parser.add_argument("scene", help="Scene name (file stem under scenes/)")
    parser.add_argument("--db", default=":memory:", help="SQLite file path (default: in-memory)")
    args = parser.parse_args(argv)
    run(args.world_dir, args.scene, args.db)


if __name__ == "__main__":
    main()
