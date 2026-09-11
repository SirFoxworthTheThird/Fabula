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
from fabula.library import DEFAULT_ROOT, Library
from fabula.player import Player
from fabula.settings import DEFAULT_SETTINGS, Settings, remember_player
from fabula.shelf import stocked
from fabula.env import load_env
from fabula.llm import (
    LiteLLMClient,
    ModelUnavailable,
    Routed,
    get_default_llm,
    missing_credentials,
)
from fabula.cards import NotACard
from fabula.cards import as_world as card_world
from fabula.cards import read as cards_read
from fabula.models import Character, ProjectedEvent
from fabula.session import Session


def _wrap(text: str, width: int = 76) -> str:
    """A paragraph a terminal can read, rather than one long line."""
    import textwrap

    return "\n".join(textwrap.wrap(" ".join(text.split()), width=width)) + "\n"


def _format(projected: ProjectedEvent, characters: dict[str, Character]) -> str:
    """Render one event as the user's character perceived it.

    Always `perceived_content`, never `event.content`: what reaches the
    terminal is the player character's POV, not the world log.
    """
    event = projected.event
    # The scene's own first words. Not indented with the things that
    # happened in the room, because it is not one of them.
    if event.metadata.get("opening"):
        return _wrap(projected.perceived_content)
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
    parser.add_argument(
        "--worlds", type=Path, default=None,
        help="Where worlds live (default ~/.fabula/worlds)",
    )
    parser.add_argument("--library", type=Path, default=DEFAULT_ROOT, help="Where your stories live")
    parser.add_argument("--list", action="store_true", help="List your stories and stop")
    parser.add_argument("--resume", metavar="ID", help="Pick a story back up")
    parser.add_argument("--delete", metavar="ID", help="Delete a story and stop")
    parser.add_argument(
        "--invent", default=None, metavar="PREMISE",
        help="Make a world from a sentence and play it",
    )
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
        "--fast-model", default=None, metavar="MODEL",
        help="A cheaper model for the calls nobody reads — the memory readings, "
             "the summaries, the beat. Most of a turn.",
    )
    parser.add_argument(
        "--fast-api-base", default=None, metavar="URL",
        help="Where that one lives, if it is not where --api-base points",
    )
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
    parser.add_argument(
        "--no-direct",
        action="store_true",
        help="Take beats in the engine's order instead of asking the model to choose",
    )
    parser.add_argument("--port", type=int, default=8000, help="Where the app listens")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the app without opening a browser at it",
    )
    parser.add_argument(
        "--card", default=None, metavar="FILE",
        help="Play a character card (a .png or .json from Chub, SillyTavern, Risu…)",
    )
    parser.add_argument(
        "--cards", default=None, metavar="WORLD",
        help="Write everybody in a world out as character cards and stop",
    )
    parser.add_argument(
        "--open-ended", action="store_true",
        help="Play to stay in the story rather than to finish it: no endings, no scene "
             "seams, and the engine writes what happens next when the world runs out",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        help="Play here instead of in the browser",
    )
    args = parser.parse_args(argv)
    # Resolved here rather than as an argparse default: it makes a
    # directory and copies files, which is an entry point's business and
    # not something that should happen because somebody imported this.
    if args.worlds is None:
        args.worlds = stocked()

    library = Library(root=args.library, worlds_root=args.worlds)
    # The same choice the app's own settings panel writes, so the two
    # doors do not disagree about which model answers. A flag still wins:
    # it is for this run.
    settings = Settings.load(args.settings)
    # Who they usually are, unless this run says otherwise. Learned from
    # the last story they actually started rather than set on a screen.
    if args.played_as is None and settings.player_name:
        args.played_as = settings.player_name
    if args.look is None and settings.player_look:
        args.look = settings.player_look
    if args.fast_model and not args.model:
        parser.error(
            "--fast-model needs --model: it is for the calls the first model "
            "would otherwise make, so on its own it changes nothing"
        )
    if args.model:
        # Before the story opens, not several turns in. A key that was
        # never set used to surface as a provider traceback mid-scene.
        # Both models, because a second one is a second way to die.
        for named, base in (
            (args.model, args.api_base),
            (args.fast_model, args.fast_api_base or args.api_base),
        ):
            unreachable = named and missing_credentials(named, base)
            if unreachable:
                parser.error(unreachable)
    llm = settings.client()
    if args.model:
        llm = LiteLLMClient(model=args.model, api_base=args.api_base)
        if args.fast_model:
            # One client still, and what the call is for decides which
            # of the two answers it.
            llm = Routed(
                llm,
                LiteLLMClient(
                    model=args.fast_model,
                    api_base=args.fast_api_base or args.api_base,
                ),
            )
    interpret = settings.interpret and not args.no_interpret
    opening = {
        "open_ended": args.open_ended,
        "interpret_beliefs": interpret,
        "direct_beats": settings.direct and not args.no_direct,
        "workers": args.workers if args.workers is not None else settings.workers,
    }

    if args.cards:
        _export_cards(args, parser)
        return

    if args.card:
        try:
            made, opening_scene = card_world(
                cards_read(Path(args.card).read_bytes()), args.worlds,
                player_name=(args.played_as or ""),
            )
        except (OSError, NotACard) as refused:
            parser.error(str(refused))
        print(f"Imported {made.name} — {made}\n")
        args.world, args.scene = made.name, opening_scene

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
    if not (args.resume or args.world or args.terminal or args.invent):
        serve(
            worlds_root=args.worlds,
            port=args.port,
            model=args.model,
            api_base=args.api_base,
            fast_model=args.fast_model,
            fast_api_base=args.fast_api_base,
            workers=args.workers,
            library_root=args.library,
            settings_path=args.settings,
            open_browser=not args.no_browser,
        )
        return

    if args.terminal and not (args.resume or args.world):
        show_library(library)
        return

    if args.invent:
        world_id, scene_id = _invent(args, llm or get_default_llm(), parser)
        args.world, args.scene = world_id, scene_id

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
        remember_player(
            settings, args.settings, player.called, player.look.strip()
        )

    run(session, interpret_beliefs=interpret)


def _invent(args, llm, parser) -> tuple[str, str]:
    """Make a world from a sentence, and say what it cost to make it."""
    from fabula.invent import CannotInvent, invent

    print(f"Making a world from: {args.invent}")
    try:
        made = invent(args.invent, llm, worlds_root=args.worlds)
    except CannotInvent as refused:
        parser.error(f"could not make that into a world: {refused}")
    print(f"  {made.title} — {made.world_dir}")
    # What it decided the premise was asking for. Worth one line: a
    # heist that came out a confession is the failure nothing else
    # reports, because every shape produces a world that plays.
    print(f"  shaped as: {made.shape}")
    # Said out loud rather than buried: a description that had to be
    # rewritten, or lost, is the difference between the story you asked
    # for and the one that is playable.
    for note in made.repaired:
        print(f"  rewritten: {note}")
    for note in made.dropped:
        print(f"  dropped: {note}")
    print()
    return made.world_dir.name, made.scene


def _export_cards(args, parser) -> None:
    """Everybody in a world, as cards the rest of the shelf can read."""
    from fabula.cards import png as card_png
    from fabula.loader import load_characters, load_scene, load_world

    where = _world_dir(args.worlds, args.cards)
    try:
        world, cast = load_world(where), load_characters(where)
    except Exception as unreadable:
        parser.error(f"could not read {args.cards}: {unreadable}")
    scenes = sorted((where / "scenes").glob("*.yaml"))
    scene = load_scene(where, scenes[0].stem) if scenes else None
    into = where / "cards"
    into.mkdir(exist_ok=True)
    for character in cast.values():
        if character.is_user:
            # The player is whoever is holding them. There is nobody to
            # export.
            continue
        path = into / f"{character.id}.png"
        path.write_bytes(card_png(world, character, scene, cast))
        print(f"  {character.name} — {path}")
    print()


def _world_dir(worlds_root: Path, world: str) -> Path:
    """A world is a name under the worlds root, or a path to one."""
    named = worlds_root / world
    return named if named.is_dir() else Path(world)


if __name__ == "__main__":
    main()
