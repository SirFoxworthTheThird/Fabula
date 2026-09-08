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
from fabula.api import serve
from fabula.concurrency import DEFAULT_WORKERS
from fabula.library import DEFAULT_ROOT, Library
from fabula.player import Player
from fabula.settings import DEFAULT_SETTINGS, Settings
from fabula.env import load_env
from fabula.llm import LiteLLMClient, ModelUnavailable, missing_credentials
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


def show_library(library: Library) -> None:
    stories = library.list()
    if not stories:
        print("No stories yet. Start one:  fabula <world> <scene>")
        return
    print(f"Your stories  ({library.root})\n")
    for card in stories:
        played = card.played_at.strftime("%d %b %H:%M")
        turns = "unplayed" if card.unplayed else f"{card.turns} turn{'s' * (card.turns != 1)}"
        title = f"{card.title} (as {card.character})" if card.character else card.title
        print(f"  {card.id}  {title:<40} {card.scene:<20} {turns:>9}   {played}")
    print("\nResume one:  fabula --resume <id>")


def run(
    session: Session,
    interpret_beliefs: bool = True,
) -> None:
    world, characters = session.world, session.characters
    you = session.user_character

    say = world.phrasing.say
    print(f"--- {session.scene.id} ({session.scene.mode}) ---")
    print(say("you_are", name=you.name, room=world.room_name(session.here())))
    print(say("help") + "\n")
    # Whatever the character has already perceived: the scene's opening
    # line on a new story, and everything they lived through on a resumed
    # one. A player picking a story back up needs to be told where they
    # left off, not just which room they are standing in.
    _show(session.perceived_so_far(), characters)
    print()

    try:
        session = _play(session, you, characters, say) or session
    finally:
        # However this ends — quit, end of input, Ctrl-C — the turn still
        # open has to be settled or the last exchange never reaches disk.
        session.close()


def _play(session: Session, you: Character, characters, say) -> None:
    while True:
        # `session` is rebound when the story goes on, so everything below
        # reads it fresh rather than the one the loop started with.
        world = session.world
        say = world.phrasing.say
        you = session.user_character
        characters = session.characters
        try:
            raw = input(f"{you.name}> ")
        except EOFError:
            print()
            return session
        try:
            outcome = run_command(session, raw, consent=_consent_asker(session))
        except ModelUnavailable as failure:
            # The scene is intact — the take was rolled back — so say what
            # happened and hand the prompt back rather than dying with a
            # stack trace and taking the story with it.
            print(f"  {say('model_failed', reason=failure)}\n")
            continue
        if outcome.quit:
            return session
        if outcome.replaced:
            print(f"  {say('again')}")
        if outcome.message:
            # A reveal is a block, not a note; do not indent it into a line.
            print(outcome.message if "\n" in outcome.message else f"  {outcome.message}")
        _show(_without_own_echo(outcome.perceived, you), characters)
        if outcome.went_on is not None:
            session = outcome.went_on
            print(say("you_are", name=session.user_character.name,
                      room=session.world.room_name(session.here())))
            # The next scene opens with its own line, exactly as the first
            # one did — a seam in the story is still a curtain going up.
            _show(session.perceived_so_far(), session.characters)
            print()
            continue
        if outcome.ended:
            key = "ended_with_next" if session.next_scene() else "ended"
            print("\n  " + say(key, scene=session.scene.id) + "\n")
        elif outcome.message or outcome.perceived:
            print()


def main(argv: list[str] | None = None) -> None:
    # Read a local .env first, so a key in the file is available to
    # everything below. Only in an entry point: importing a library
    # should never mutate the process environment.
    load_env()
    parser = argparse.ArgumentParser(prog="fabula", description="Play a story.")
    parser.add_argument("world", nargs="?", help="A directory under --worlds, or a path to one")
    parser.add_argument("scene", nargs="?", help="Scene name (file stem under scenes/)")
    parser.add_argument("--worlds", type=Path, default=Path("worlds"), help="Where worlds live")
    parser.add_argument("--library", type=Path, default=DEFAULT_ROOT, help="Where your stories live")
    parser.add_argument("--list", action="store_true", help="List your stories and stop")
    parser.add_argument("--resume", metavar="ID", help="Pick a story back up")
    parser.add_argument("--delete", metavar="ID", help="Delete a story and stop")
    parser.add_argument("--title", default=None, help="Name a new story")
    parser.add_argument(
        "--as", dest="played_as", default=None, metavar="NAME",
        help="Play as somebody of your own rather than the authored character",
    )
    parser.add_argument(
        "--look", default=None, metavar="LINE",
        help="One line the room can see about you, at the start of the story",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Play against a database file directly, outside the library",
    )
    parser.add_argument("--model", default=None, help="Any model id litellm understands")
    parser.add_argument("--api-base", default=None, help="An OpenAI-compatible endpoint")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="How many model calls a turn may have in flight at once (1 = one at a time)",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS,
        help="Where your choice of model is kept",
    )
    parser.add_argument(
        "--no-interpret",
        action="store_true",
        help="Skip the per-memory reading — most of a scene's model calls",
    )
    parser.add_argument("--port", type=int, default=8000, help="Where the app listens")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the app without opening a browser at it",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Play here instead of in the browser",
    )
    args = parser.parse_args(argv)

    library = Library(root=args.library, worlds_root=args.worlds)
    # The same choice the app's own settings panel writes, so the two
    # doors do not disagree about which model answers. A flag still wins:
    # it is for this run.
    settings = Settings.load(args.settings)
    if args.model:
        # Before the story opens, not several turns in. A key that was
        # never set used to surface as a provider traceback mid-scene.
        unreachable = missing_credentials(args.model, args.api_base)
        if unreachable:
            parser.error(unreachable)
    llm = (
        LiteLLMClient(model=args.model, api_base=args.api_base)
        if args.model
        else settings.client()
    )
    interpret = settings.interpret and not args.no_interpret
    opening = {
        "interpret_beliefs": interpret,
        "workers": args.workers if args.workers is not None else settings.workers,
    }

    if args.delete:
        # A mistyped id is a typo, not a crash: the ids are for people to
        # copy off a listing by hand.
        try:
            gone = library.delete(args.delete)
        except ValueError:
            parser.error(f"not a story id: {args.delete} (see fabula --list)")
        print("Deleted." if gone else f"No story {args.delete}.")
        return
    if args.list:
        show_library(library)
        return

    # Typing the name of the app opens the app. Everything else here —
    # naming a scene, resuming an id, --terminal — is the developer's
    # door, and stays exactly where it was; but a person who installed a
    # roleplay app and typed its name should get the app, not a REPL.
    if not (args.resume or args.world or args.terminal):
        serve(
            worlds_root=args.worlds,
            port=args.port,
            model=args.model,
            api_base=args.api_base,
            workers=args.workers,
            library_root=args.library,
            settings_path=args.settings,
            open_browser=not args.no_browser,
        )
        return

    if args.terminal and not (args.resume or args.world):
        show_library(library)
        return

    if args.resume:
        try:
            session = library.resume(args.resume, llm=llm, **opening)
        except (ValueError, FileNotFoundError):
            parser.error(f"no story {args.resume} (see fabula --list)")
    elif args.db:
        # The escape hatch: a file you name, outside the library.
        session = Session.open(
            _world_dir(args.worlds, args.world), args.scene, args.db, llm=llm, **opening
        )
    else:
        if not args.scene:
            parser.error("a new story needs a scene: fabula <world> <scene>")
        player = Player(name=args.played_as or "", look=args.look or "")
        complaint = player.complaint()
        if complaint:
            parser.error(complaint)
        session = library.start(
            args.world, args.scene, title=args.title, llm=llm, player=player, **opening
        )

    run(session, interpret_beliefs=interpret)


def _world_dir(worlds_root: Path, world: str) -> Path:
    """A world is a name under the worlds root, or a path to one."""
    named = worlds_root / world
    return named if named.is_dir() else Path(world)


if __name__ == "__main__":
    main()
