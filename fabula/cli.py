"""Terminal client: loads a scene from YAML, runs the turn loop, prints
to stdout. The user plays a character via stdin.

This is a client, not the engine. All it does is turn typed lines into
`Session` calls and render the `ProjectedEvent`s that come back — which
is why it can only ever show the player what their character perceived.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from fabula.chronology import describe_duration
from fabula.commands import run_command
from fabula.env import load_env
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


def _consent_asker(session: Session) -> Callable[[int], bool]:
    """Large skips are the user's call — they are a character with agency,
    and time is not something they should lose without noticing.

    The duration is phrased by the world, so the question does not read
    half in one language and half in another.
    """

    def ask(minutes: int) -> bool:
        duration = describe_duration(minutes, session.world.phrasing)
        question = session.world.phrasing.say("let_time_pass", duration=duration)
        return input(f"  {question}").strip().lower() in ("y", "yes")

    return ask


def run(
    world_dir: Path,
    scene_name: str,
    db_path: str = ":memory:",
    model: str | None = None,
    api_base: str | None = None,
    interpret_beliefs: bool = True,
) -> None:
    llm = LiteLLMClient(model=model, api_base=api_base) if model else None
    session = Session.open(
        world_dir, scene_name, db_path, llm=llm, interpret_beliefs=interpret_beliefs
    )
    world, characters = session.world, session.characters
    you = session.user_character

    say = world.phrasing.say
    print(f"--- {session.scene.id} ({session.scene.mode}) ---")
    print(say("you_are", name=you.name, room=world.room_name(session.here())))
    print(say("help") + "\n")

    while True:
        try:
            raw = input(f"{you.name}> ")
        except EOFError:
            print()
            break
        outcome = run_command(session, raw, consent=_consent_asker(session))
        if outcome.quit:
            break
        if outcome.replaced:
            print(f"  {say('again')}")
        if outcome.message:
            # A reveal is a block, not a note; do not indent it into a line.
            print(outcome.message if "\n" in outcome.message else f"  {outcome.message}")
        _show(_without_own_echo(outcome.perceived, you), characters)
        if outcome.ended:
            print("\n  " + say("ended", scene=session.scene.id) + "\n")
        elif outcome.message or outcome.perceived:
            print()


def main(argv: list[str] | None = None) -> None:
    # Read a local .env first, so a key in the file is available to
    # everything below. Only in an entry point: importing a library
    # should never mutate the process environment.
    load_env()
    parser = argparse.ArgumentParser(prog="fabula", description="Run a Fabula scene from YAML.")
    parser.add_argument("world_dir", type=Path, help="Path to worlds/<name>/")
    parser.add_argument("scene", help="Scene name (file stem under scenes/)")
    parser.add_argument("--db", default=":memory:", help="SQLite file path (default: in-memory)")
    parser.add_argument("--model", default=None, help="Any model id litellm understands")
    parser.add_argument("--api-base", default=None, help="An OpenAI-compatible endpoint")
    parser.add_argument(
        "--no-interpret",
        action="store_true",
        help="Skip the per-memory reading — most of a scene's model calls",
    )
    args = parser.parse_args(argv)
    run(
        args.world_dir,
        args.scene,
        args.db,
        args.model,
        args.api_base,
        interpret_beliefs=not args.no_interpret,
    )


if __name__ == "__main__":
    main()
