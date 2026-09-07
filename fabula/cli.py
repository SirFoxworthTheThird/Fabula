"""Terminal client: loads a scene from YAML, runs the turn loop, prints
to stdout. The user plays a character via stdin.

This is a client, not the engine. All it does is turn typed lines into
`Session` calls and render the `ProjectedEvent`s that come back — which
is why it can only ever show the player what their character perceived.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fabula.chronology import describe_duration
from fabula.commands import run_command
from fabula.llm import LiteLLMClient
from fabula.models import Character, ProjectedEvent
from fabula.session import Session


def _format(projected: ProjectedEvent, characters: dict[str, Character]) -> str:
    """Render one event as the user's character perceived it.

    Always `perceived_content`, never `event.content`: what reaches the
    terminal is the player character's POV, not the world log.
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


def _show(projected: list[ProjectedEvent], characters: dict[str, Character]) -> None:
    for event in projected:
        print(_format(event, characters))


def _without_own_echo(
    projected: list[ProjectedEvent], you: Character
) -> list[ProjectedEvent]:
    """Drop the leading event the player just performed themselves.

    A session hands over everything the character perceived, their own
    action included; in a terminal the player has just typed it, so
    echoing it is noise. Purely a presentation trim.
    """
    if projected and projected[0].event.actor_id == you.id:
        return projected[1:]
    return projected


def _ask_consent(minutes: int) -> bool:
    """Large skips are the user's call — they are a character with agency,
    and time is not something they should lose without noticing."""
    answer = input(f"  Let {describe_duration(minutes)} pass? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def run(
    world_dir: Path,
    scene_name: str,
    db_path: str = ":memory:",
    model: str | None = None,
    api_base: str | None = None,
) -> None:
    llm = LiteLLMClient(model=model, api_base=api_base) if model else None
    session = Session.open(world_dir, scene_name, db_path, llm=llm)
    world, characters = session.world, session.characters
    you = session.user_character

    print(f"--- {session.scene.id} ({session.scene.mode}) ---")
    print(f"You are {you.name}, in {world.room_name(session.here())}.")
    print(
        "(/go <room>, /wait, /look, /quit. /reveal spoils the scene when "
        "you are done. Anything else you say aloud.)\n"
    )

    while True:
        try:
            raw = input(f"{you.name}> ")
        except EOFError:
            print()
            break
        outcome = run_command(session, raw, consent=_ask_consent)
        if outcome.quit:
            break
        if outcome.message:
            # A reveal is a block, not a note; do not indent it into a line.
            print(outcome.message if "\n" in outcome.message else f"  {outcome.message}")
        _show(_without_own_echo(outcome.perceived, you), characters)
        if outcome.message or outcome.perceived:
            print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fabula", description="Run a Fabula scene from YAML.")
    parser.add_argument("world_dir", type=Path, help="Path to worlds/<name>/")
    parser.add_argument("scene", help="Scene name (file stem under scenes/)")
    parser.add_argument("--db", default=":memory:", help="SQLite file path (default: in-memory)")
    parser.add_argument("--model", default=None, help="Any model id litellm understands")
    parser.add_argument("--api-base", default=None, help="An OpenAI-compatible endpoint")
    args = parser.parse_args(argv)
    run(args.world_dir, args.scene, args.db, args.model, args.api_base)


if __name__ == "__main__":
    main()
