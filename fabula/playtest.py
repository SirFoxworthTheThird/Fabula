"""Run a scripted scene and print it for reading.

The spec gates everything on M0: *if this scene is not compelling in a
terminal, nothing built later will save it.* That is a judgment a person
has to make, but it should not require playing by hand every time
something changes. This runs a fixed script against a real model and
prints the transcript, so the judgment is repeatable and diffable.

**This is an author's tool, and it is deliberately omniscient.** The
belief dump at the end shows every character's private store, which is
exactly what no player-facing client may ever do. It exists so you can
check that the asymmetry landed — that Maria really does not know — and
nothing here should be reused in a client.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fabula.commands import run_command
from fabula.env import load_env
from fabula.llm import LiteLLMClient, LLMClient, get_default_llm
from fabula.models import Character, ProjectedEvent
from fabula.session import Session

# Presses on the secret, leaves the room, lets the world move without the
# player, and comes back to find it changed.
DEFAULT_SCRIPT = [
    "Tomás, you have been quiet all evening.",
    "What happened to Grandma's music box?",
    "/look",
    "/go study",
    "/wait",
    "/go kitchen",
    "/look",
]


def read_script(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_SCRIPT)
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def format_line(projected: ProjectedEvent, characters: dict[str, Character]) -> str:
    event = projected.event
    if event.kind != "utterance" or projected.perception != "full":
        return f"    {projected.perceived_content}"
    actor = characters.get(event.actor_id) if event.actor_id else None
    return f"{(actor.name if actor else event.actor_id or '???'):>10}: {projected.perceived_content}"


def playtest(
    world_dir: Path,
    scene_name: str,
    script: list[str],
    model: str | None = None,
    show_beliefs: bool = True,
    llm: LLMClient | None = None,
    api_base: str | None = None,
    interpret_beliefs: bool = True,
) -> None:
    if llm is None:
        llm = LiteLLMClient(model=model, api_base=api_base) if model else get_default_llm()
    session = Session.open(
        world_dir, scene_name, llm=llm, interpret_beliefs=interpret_beliefs
    )
    you = session.user_character
    characters = session.characters

    print(f"=== {session.scene.id} ({session.scene.mode}) ===")
    print(f"{you.name}, in {session.world.room_name(session.here())}.")
    # Say plainly that the player character is on rails here. Fixed input
    # is what makes two runs comparable, but without a word about it the
    # transcript reads as though the protagonist keeps making the same
    # choices on her own.
    print(
        f"(scripted run: {you.name}'s lines are fixed, so runs can be compared. "
        "--script for your own; play freely with `python -m fabula.cli`.)\n"
    )

    for line in script:
        print(f"{you.name:>10}> {line}")
        outcome = run_command(session, line, consent=lambda _minutes: True)
        if outcome.quit:
            break
        if outcome.message:
            print(f"    ({outcome.message})")
        for projected in outcome.perceived:
            if projected.event.actor_id == you.id:
                continue  # already shown as the prompt line
            print(format_line(projected, characters))
        print()

    session.close()  # settle the last turn before reading anything back

    if show_beliefs:
        print("=== what each character came away believing ===")
        print("(author's view — no client may ever show this)\n")
        for character in characters.values():
            beliefs = sorted(
                session.store.get_beliefs(character.id), key=lambda b: -b.salience
            )
            print(f"  {character.name} ({session.director.current_location(character)})")
            if not beliefs:
                print("    — nothing they thought worth keeping")
            for belief in beliefs[:8]:
                confidence = "sure" if belief.confidence >= 1.0 else "unsure"
                print(f"    [{belief.salience:.2f} {confidence:>6}] {belief.content}")
                if belief.interpretation:
                    # What they made of it, under what they perceived —
                    # the most useful thing on this page when you are
                    # deciding whether a character is reading the room.
                    print(f"    {'':>15} → {belief.interpretation}")
            print()


def main(argv: list[str] | None = None) -> None:
    # Read a local .env first, so a key in the file is available to
    # everything below. Only in an entry point: importing a library
    # should never mutate the process environment.
    load_env()
    parser = argparse.ArgumentParser(
        prog="fabula-playtest",
        description="Run a scripted scene and print it for reading.",
    )
    parser.add_argument("world_dir", type=Path, help="Path to worlds/<name>/")
    parser.add_argument("scene", help="Scene name (file stem under scenes/)")
    parser.add_argument("--script", type=Path, default=None, help="One command per line")
    parser.add_argument("--model", default=None, help="Any model id litellm understands")
    parser.add_argument("--api-base", default=None, help="An OpenAI-compatible endpoint")
    parser.add_argument("--no-beliefs", action="store_true", help="Transcript only")
    parser.add_argument(
        "--no-interpret",
        action="store_true",
        help="Skip the per-memory reading — most of a scene's model calls",
    )
    args = parser.parse_args(argv)

    playtest(
        args.world_dir,
        args.scene,
        read_script(args.script),
        model=args.model,
        show_beliefs=not args.no_beliefs,
        api_base=args.api_base,
        interpret_beliefs=not args.no_interpret,
    )


if __name__ == "__main__":
    main()
