"""M0 CLI: loads a scene from YAML, runs the turn loop, prints to stdout.
The user plays a character via stdin.

No frontend assumptions live here beyond argv/stdin/stdout — the engine
(director, world, memory) knows nothing about this file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fabula.chronology import describe_duration
from fabula.db import EventStore
from fabula.director import Director
from fabula.llm import get_default_llm
from fabula.loader import load_pressures, load_scenario
from fabula.models import Character, Event, ProjectedEvent
from fabula.narrator import Narrator


def _format(projected: ProjectedEvent, characters: dict[str, Character]) -> str:
    """Render one event as the user's character perceived it.

    Always `perceived_content`, never `event.content`: what reaches the
    terminal is the player character's POV, not the world log. A client
    that prints the log hands the player their own character's blind
    spots, which would defeat the whole engine.
    """
    event = projected.event
    # Only an utterance is somebody speaking. Arrivals, departures and the
    # rest carry narrator-rendered prose about a character, not words from
    # their mouth, so they must not be printed behind a speaker's name.
    if event.kind != "utterance" or projected.perception != "full":
        return f"  {projected.perceived_content}"
    actor = characters.get(event.actor_id) if event.actor_id else None
    name = actor.name if actor else (event.actor_id or "???")
    return f"{name}: {projected.perceived_content}"


def _show(
    director: Director,
    user_character: Character,
    new_events: list[Event],
    characters: dict[str, Character],
) -> None:
    """Print only what the user's character perceived of these events."""
    all_events = director.store.get_events(director.scene.id)
    new_seqs = {event.seq for event in new_events}
    for projected in director.contexts.project(user_character, all_events):
        if projected.event.seq in new_seqs:
            print(_format(projected, characters))


def _ask_consent(minutes: int) -> bool:
    """Large skips are the user's call — they are a character with agency,
    and time is not something they should lose without noticing."""
    answer = input(f"  Let {describe_duration(minutes)} pass? [y/N] ").strip().lower()
    return answer in ("y", "yes")


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
    print("(/go <room>, /wait, /look, /quit. Anything else you say aloud.)\n")

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

        here = director.current_location(user_character)

        if raw == "/wait":
            skipped = director.advance_time(consent=_ask_consent)
            if not skipped:
                print("  Nothing is pending; time stays where it is.\n")
                continue
            _show(director, user_character, skipped, characters)
            print()
            continue

        if raw == "/look":
            waiting = director.unmaterialized_here(user_character)
            revealed = [
                event
                for summary in waiting
                if (event := director.materialize(summary, user_character)) is not None
            ]
            if not revealed:
                print("  Nothing here has changed since you last looked.\n")
                continue
            _show(director, user_character, revealed, characters)
            print()
            continue

        if raw.startswith("/go "):
            destination = raw[len("/go ") :].strip()
            if destination not in world.rooms:
                print(f"  There is no {destination} to go to.\n")
                continue
            if destination == here:
                print(f"  You are already in {world.room_name(destination)}.\n")
                continue
            arrival = director.build_event(
                "arrival",
                user_character.id,
                destination,
                f"{user_character.name} comes in from {world.room_name(here)}.",
                audibility="adjacent",
            )
            turn_events = director.run_turn(arrival)
            print(f"  You are in {world.room_name(destination)}.")
            _show(director, user_character, turn_events[1:], characters)
            print()
            continue

        user_event = director.build_event("utterance", user_character.id, here, raw)
        turn_events = director.run_turn(user_event)
        _show(director, user_character, turn_events[1:], characters)
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
