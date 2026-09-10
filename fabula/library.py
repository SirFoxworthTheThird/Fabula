"""Your stories, on your machine.

The application is the local part, not the model — so what "local" has to
mean to somebody using it is that the stories are *theirs*: findable,
resumable, copyable, deletable, without an account and without asking
anything for permission.

A story is one SQLite file, and the library is the directory they live
in. Its own description lives inside it, in a single `story` row, which
is what makes the directory the index: nothing to rebuild, nothing to
fall out of sync, and `rm` is a supported way to delete a story. Copy the
file to another machine and the story goes with it, characters and
grudges and all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fabula.db import EventStore
from fabula.llm import LLMClient
from fabula.player import Player
from fabula.session import Session
from fabula.shelf import stocked

DEFAULT_ROOT = Path.home() / ".fabula" / "stories"


@dataclass
class StoryCard:
    """What the library knows about a story without opening it properly."""

    id: str
    title: str
    world: str
    scene: str
    # Who the player is in it, when it is somebody they made.
    character: str
    # Whether it is being played to stay in rather than to finish.
    open_ended: bool
    created_at: datetime
    played_at: datetime
    turns: int
    path: Path

    @property
    def unplayed(self) -> bool:
        return self.turns == 0


def default_title(world: str, scene: str) -> str:
    """A fallback for a story whose scene could not be read."""
    return f"{world.replace('_', ' ').title()} — {scene.replace('_', ' ')}"


class Library:
    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        worlds_root: Path | str | None = None,
    ):
        self.root = Path(root)
        # Never the current directory. A relative default is a bet that
        # the caller is standing in this repository, and everyone who is
        # not got an empty shelf.
        self.worlds_root = Path(worlds_root) if worlds_root is not None else stocked()

    def _path(self, story_id: str) -> Path:
        # Ids are generated here and never taken from a client, but this
        # is the one place a bad one would become a filesystem path.
        if not re.fullmatch(r"[a-f0-9]{12}", story_id):
            raise ValueError(f"not a story id: {story_id!r}")
        return self.root / f"{story_id}.sqlite"

    def list(self) -> list[StoryCard]:
        """Every story in the directory, most recently played first.

        A file that is not a story, or is a story from a future version
        this build cannot read, is skipped rather than raised on: a
        library that refuses to open because of one bad file is worse
        than one that quietly shows the rest.
        """
        cards = []
        for path in sorted(self.root.glob("*.sqlite")):
            try:
                store = EventStore(str(path))
                row = store.get_story()
                store.close()
            except Exception:
                continue
            if row:
                cards.append(
                    StoryCard(
                        id=row["story_id"],
                        title=row["title"],
                        world=row["world"],
                        scene=row["scene"],
                        character=row["character_name"],
                        open_ended=bool(row["open_ended"]),
                        created_at=datetime.fromisoformat(row["created_at"]),
                        played_at=datetime.fromisoformat(row["played_at"]),
                        turns=row["turns"],
                        path=path,
                    )
                )
        return sorted(cards, key=lambda card: card.played_at, reverse=True)

    def start(
        self,
        world: str,
        scene: str,
        title: str | None = None,
        llm: LLMClient | None = None,
        player: Player | None = None,
        open_ended: bool = False,
        **kwargs,
    ) -> Session:
        """Begin a story and put it in the library."""
        self.root.mkdir(parents=True, exist_ok=True)
        story_id = uuid4().hex[:12]
        path = self._path(story_id)
        store = EventStore(str(path))
        try:
            session = Session.open(
                self.worlds_root / world, scene, llm=llm, store=store, player=player,
                open_ended=open_ended, **kwargs,
            )
            # Named after the scene it begins in, in the author's words —
            # "The dinner" rather than "Ashgrove — the_dinner". Written
            # after the scene loads, because that is where the name is.
            store.start_story(
                story_id, title or session.scene.name or default_title(world, scene),
                world, scene,
                character_name=(player.called if player else ""),
                character_look=(player.look.strip() if player else ""),
                open_ended=open_ended,
            )
            return session
        except Exception:
            # A world that does not load leaves no card behind. The file
            # is written before the scene is opened (the story row is
            # what makes it a story), so a failed start has to take it
            # back out or the library shows a story nobody can resume.
            store.close()
            path.unlink(missing_ok=True)
            raise

    def resume(self, story_id: str, llm: LLMClient | None = None, **kwargs) -> Session:
        """Pick a story back up where it was left.

        On the scene it was left on, with the same file underneath — so
        everybody arrives holding what they held when it was put down.
        """
        path = self._path(story_id)
        if not path.is_file():
            raise FileNotFoundError(f"no story {story_id}")
        store = EventStore(str(path))
        row = store.get_story()
        if row is None:
            store.close()
            raise ValueError(f"{path} is not a story")
        # How the story is played is the story's, not the caller's. A CLI
        # run with `--open-ended` must not quietly convert a story that
        # was started with its endings intact, and a run without the flag
        # must not put endings back into one that never had them.
        kwargs.pop("open_ended", None)
        # The same person they made when they started it. The look is not
        # replayed — it was said once, at the beginning — but the name
        # goes back through every line of authored prose.
        session = Session.open(
            self.worlds_root / row["world"],
            row["scene"],
            llm=llm,
            store=store,
            player=Player(name=row["character_name"], look=row["character_look"]),
            # How it was being played, not how the caller happens to be
            # opening it: a story you left running does not quietly
            # acquire endings because you resumed it from somewhere else.
            open_ended=bool(row["open_ended"]),
            **kwargs,
        )
        session.steer(row["steering"])
        return session

    def delete(self, story_id: str) -> bool:
        path = self._path(story_id)
        if not path.is_file():
            return False
        path.unlink()
        return True
