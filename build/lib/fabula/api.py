"""FastAPI service + SSE streaming (spec §12, M6).

The native protocol is a POV, not a transcript. Every event on the wire
has already been through the world model: it carries the requesting
character's *perceived* content and states its own visibility. An event
that character could not perceive is not sent-and-flagged for a client
to hide — it never enters the stream at all, which is the only version
of this that survives a careless client.

The engine knows nothing about this module. Everything here goes through
`Session`, the same surface the CLI uses.
"""
from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from fabula.env import load_env
from fabula.concurrency import DEFAULT_WORKERS
from fabula.library import DEFAULT_ROOT, Library
import shutil

from fabula.art import authored, cover, portrait
from fabula.cards import NotACard
from fabula.cards import as_world as card_world
from fabula.cards import png as card_png
from fabula.cards import read as cards_read
from fabula.inspect import complaints
from fabula.discovery import invents_a_fact
from fabula.loader import catalogue, load_characters, load_scene, load_world
from fabula.player import Player
from fabula.settings import DEFAULT_SETTINGS, Settings, describe, remember_player
from fabula.shelf import stocked
from fabula.invent import CannotInvent, invent
from fabula.llm import (
    LiteLLMClient,
    LLMClient,
    ModelUnavailable,
    Routed,
    get_default_llm,
    missing_credentials,
)
from fabula.models import ProjectedEvent
from fabula.openai_shim import add_openai_shim
from fabula.reveal import build_reveal, render
from fabula.session import Session


class StreamEvent(BaseModel):
    """One scene event as one character perceived it."""

    seq: int
    kind: str
    speaker: str | None
    speaker_name: str | None
    location: str
    location_name: str
    visibility: Literal["full", "degraded"]
    content: str
    story_time: datetime
    detail_level: str
    # The scene's own first words, addressed to this character. Not
    # something that happened in the room, so a client should not render
    # it as though somebody did it.
    opening: bool = False


class Named(BaseModel):
    id: str
    name: str


class SceneState(BaseModel):
    session_id: str
    # Which world this is, so a client can ask for its art. Not a
    # disclosure: it is the directory the player picked off the shelf.
    world: str
    scene: str
    mode: str
    character: str
    character_name: str
    location: str
    location_name: str
    # Only who this character can see is here with them — never where the
    # rest of the cast is.
    present: list[Named]
    rooms: list[Named]
    story_time: datetime
    cast: list[str]
    pending_skip_minutes: int | None
    # Whether the scene has reached its declared end condition.
    ended: bool
    # The standing note the player gave the director, if any.
    steering: str = ""
    # The scene the story goes to from here, given what happened in this
    # one. None means the story ends here — and always, when the story is
    # being played open-ended, because there are no seams to cross.
    next_scene: str | None
    open_ended: bool = False


class MissedOut(BaseModel):
    location: str
    speaker: str | None
    content: str


class HalfHeardOut(BaseModel):
    heard: str
    truth: str


class RegardOut(BaseModel):
    who: str
    toward: str
    trust: float
    started: float


class RevealOut(BaseModel):
    """Deliberately omniscient, and only produced on request."""

    character: str
    missed: list[MissedOut]
    half_heard: list[HalfHeardOut]
    knowledge: dict[str, dict[str, bool]]
    # Only the regard that moved during the scene, and never the player's
    # own — how they feel is not the engine's to report back to them.
    regard: list[RegardOut]
    text: str


class StoryOut(BaseModel):
    """A story in the library, as a card to show without opening it."""

    id: str
    title: str
    world: str
    scene: str
    character: str = ""
    open_ended: bool = False
    created_at: datetime
    played_at: datetime
    turns: int


class NewCharacter(BaseModel):
    """Who the player is playing, when it is somebody they made."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    # Asked for as what anyone can see, because it becomes something the
    # room perceives — a private truth put here would be handed to
    # everybody standing there.
    look: str = ""


class NewWorld(BaseModel):
    """A story somebody thought of, rather than one from the shelf."""

    model_config = ConfigDict(extra="forbid")

    premise: str
    character: "NewCharacter | None" = None
    open_ended: bool = False


class NewStory(BaseModel):
    world: str
    scene: str
    title: str | None = None
    character: NewCharacter | None = None
    # Played to stay in rather than to finish: no ending is reported, no
    # seam is crossed, and the engine writes what happens next once the
    # authored complications are spent.
    open_ended: bool = False


class NewSettings(BaseModel):
    """Deliberately closed: an unknown field is refused rather than
    stored, so no client can talk this into keeping a credential."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str = ""
    api_base: str = ""
    # A second, cheaper model for the calls nobody reads.
    fast_model: str = ""
    fast_api_base: str = ""
    interpret: bool = True
    direct: bool = True
    workers: int = DEFAULT_WORKERS
    # Who you usually are. A suggestion for the box, never the story's
    # own copy of who you played.
    player_name: str = ""
    player_look: str = ""


class NewSession(BaseModel):
    world: str
    scene: str


class Say(BaseModel):
    text: str


class Steer(BaseModel):
    """What the player wants out of this story, said to the director."""

    note: str = ""


class Rewind(BaseModel):
    """Take the scene back to just after this event."""

    seq: int


class Move(BaseModel):
    room: str


class Wait(BaseModel):
    consent: bool = False


class _LiveSession:
    """A `Session` plus the subscribers watching it.

    Subscriber queues live here rather than on `Session` so the engine
    stays free of transport concerns.
    """

    def __init__(self, session: Session):
        self.session = session
        self.subscribers: list[asyncio.Queue[StreamEvent]] = []
        # The loop the queues belong to, learned when somebody subscribes.
        self.loop: asyncio.AbstractEventLoop | None = None
        # Whether the engine is working on something nobody asked for —
        # a scene opening itself. Set before the response goes out, so a
        # client that connects in the gap is told to keep waiting rather
        # than told the room has finished.
        self.busy = False

    @property
    def story_id(self) -> str | None:
        saved = self.session.store.get_story()
        return saved["story_id"] if saved else None

    def publish(self, events: list) -> None:
        for queue in self.subscribers:
            for event in events:
                queue.put_nowait(event)

    def publish_soon(self, frame) -> None:
        """Publish from the worker thread the engine is running on.

        `asyncio.Queue` is not thread-safe and a turn runs in a
        threadpool, so this hops back to the loop rather than touching
        the queues where it stands. A frame that arrives after the
        listener has gone is dropped, which is what a closed tab is.
        """
        loop = self.loop
        if loop is None:
            return
        for queue in list(self.subscribers):
            try:
                loop.call_soon_threadsafe(queue.put_nowait, frame)
            except RuntimeError:
                pass


class Working(BaseModel):
    """Which agent is being asked something, right now.

    A key and a name, and deliberately nowhere to put a sentence: what a
    character is *writing* has not been checked yet, and half of what
    this engine checks after the fact exists to stop particular words
    reaching the player. See `fabula.llm.Watched`.
    """

    who: str = ""
    name: str = ""


class Trouble(BaseModel):
    """Something went wrong where nobody was waiting on a reply.

    Every other failure in this service answers whoever caused it: a turn
    that cannot reach the model is a 502 on the line that asked for it.
    An opening is the exception — it runs after its response has already
    gone — so without this it would be a room that quietly never says
    anything, which reads as the engine being slow rather than as the
    model being unreachable.
    """

    detail: str


class Settled(BaseModel):
    """Background work has stopped.

    `Working` says who is being asked; nothing ever said when the asking
    was over, and nothing needed to — every turn was somebody's request
    and ended when its response did. An opening has no response to end
    on, so without this the room finishes taking its turn and the client
    is still showing that it is thinking.
    """


class Retake(BaseModel):
    """Sent on its own SSE event type when a take is played again: drop
    everything from `from_seq` onward, then render what follows. A
    separate frame rather than a field, because a retake can legitimately
    produce no perceived events at all and the client still has to
    truncate."""

    from_seq: int


def to_stream_events(session: Session, perceived: list[ProjectedEvent]) -> list[StreamEvent]:
    out = []
    for projected in perceived:
        event = projected.event
        speaker = session.characters.get(event.actor_id) if event.actor_id else None
        out.append(
            StreamEvent(
                seq=event.seq,
                kind=event.kind,
                speaker=event.actor_id,
                speaker_name=speaker.name if speaker else None,
                location=event.location_id,
                location_name=session.world.room_name(event.location_id),
                visibility=projected.perception,
                # Perceived content only. The raw event.content never
                # reaches this module.
                content=projected.perceived_content,
                story_time=event.story_time,
                detail_level=event.detail_level,
                opening=bool(event.metadata.get("opening")),
            )
        )
    return out


def _world_dir(worlds_root: Path, name: str) -> Path:
    """Resolve a world by name, refusing anything that escapes the root.

    The service takes a world *name*, never a path: accepting a caller's
    filesystem path would make this an arbitrary-directory reader.
    """
    candidate = (worlds_root / name).resolve()
    if candidate.parent != worlds_root.resolve() or not candidate.is_dir():
        raise HTTPException(status_code=404, detail=f"no world named {name!r}")
    return candidate


def _warm_litellm() -> None:
    try:
        import litellm  # noqa: F401
    except Exception:  # pragma: no cover - depends on the install
        pass


def create_app(
    worlds_root: Path | None = None,
    llm: LLMClient | None = None,
    db_path: str = ":memory:",
    library_root: Path | None = None,
    workers: int | None = None,
    settings_path: Path | None = None,
) -> FastAPI:
    # Not an argument default: `stocked` makes a directory and copies
    # files into it, and that is a thing an app does when it starts, not
    # a thing that happens because somebody imported this module.
    worlds_root = Path(worlds_root) if worlds_root is not None else stocked()
    live: dict[str, _LiveSession] = {}
    by_token: dict[str, str] = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Importing litellm takes several seconds, and it happens on the
        # first thing that needs a model — which was the first save in
        # the settings panel, and the first line of the first story.
        # Doing it on a thread while somebody is still reading the shelf
        # costs nothing and takes that wait off the front of the app.
        threading.Thread(target=_warm_litellm, daemon=True).start()
        yield
        # The turn a session is in the middle of is uncommitted, which is
        # what lets a player take it again. Somebody has to say when play
        # is over, or on a file database the last exchange of every
        # session is lost.
        for entry in live.values():
            entry.session.close()

    app = FastAPI(
        title="Fabula",
        description="A multi-agent story engine.",
        lifespan=lifespan,
    )
    library = Library(root=library_root or DEFAULT_ROOT, worlds_root=worlds_root)
    # Which model answers is the player's choice, and a choice that lives
    # only in the flags of whoever started the process is not theirs. An
    # explicit `llm` — a flag, or a test — still wins for that run, and
    # the panel says so rather than pretending to be in charge.
    settings_file = Path(settings_path) if settings_path else DEFAULT_SETTINGS
    settings = Settings.load(settings_file)
    pinned = llm is not None
    app.state.sessions = live
    app.state.sessions_by_token = by_token
    app.state.library = library
    app.state.settings = settings

    def opening() -> dict:
        """What to open the next story with."""
        return {
            "llm": llm if pinned else settings.client(),
            "workers": workers if workers is not None else settings.workers,
            "interpret_beliefs": settings.interpret,
            "direct_beats": settings.direct,
        }

    def get_live(session_id: str) -> _LiveSession:
        if session_id not in live:
            raise HTTPException(status_code=404, detail="no such session")
        return live[session_id]

    def open_session(world_name: str, scene_name: str) -> tuple[str, Session]:
        world_dir = _world_dir(worlds_root, world_name)
        try:
            session = Session.open(world_dir, scene_name, db_path=db_path, **opening())
        except ModelUnavailable as failure:
            raise HTTPException(status_code=502, detail=f"the model did not answer: {failure}")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no scene named {scene_name!r}")
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        return _remember(session)

    def _remember(session: Session) -> tuple[str, Session]:
        session_id = uuid.uuid4().hex
        live[session_id] = _LiveSession(session)
        return session_id, session

    def remember(session: Session) -> SceneState:
        """Hold a session open for a client and describe it back."""
        session_id, session = _remember(session)
        return state_of(session_id, session)

    def opened(session: Session, background: BackgroundTasks) -> SceneState:
        """The same, for a story that has just begun: hand it back now
        and let the room open it afterwards.

        A scene's opening costs a narration, a bid from everybody
        standing there and a line from whoever wanted to speak — eight
        model calls in a full room, which is a long time to look at a
        disabled button. The story is playable the moment it exists, and
        what is missing arrives on the stream the client opens anyway, so
        the wait is spent watching the room take its turn instead of
        waiting to see the room at all.
        """
        session_id, session = _remember(session)
        live[session_id].busy = True
        background.add_task(curtain_up, session_id)
        return state_of(session_id, session)

    def curtain_up(session_id: str) -> None:
        """Open the scene, after the client has been told where it is.

        Runs in a worker thread once the response has gone, so frames hop
        back to the loop rather than touching the queues where they
        stand. Nobody may be listening yet — the catch-up every stream
        begins with is what makes that safe, and the client drops a seq
        it has already rendered.

        A model that fails here has left the scene at its authored first
        words rather than half-opened, which is a story somebody can
        still play. Anyone who caught the rolled-back lines mid-flight is
        told to drop them, through the same retake frame `/again` uses.
        """
        entry = live.get(session_id)
        if entry is None:
            return
        entry.busy = True
        from_seq = entry.session.store.next_seq(entry.session.scene.id)
        try:
            for frame in to_stream_events(entry.session, entry.session.raise_curtain()):
                entry.publish_soon(frame)
        except ModelUnavailable as failure:
            entry.publish_soon(Retake(from_seq=from_seq))
            entry.publish_soon(Trouble(detail=f"the model did not answer: {failure}"))
        except Exception as failure:
            entry.publish_soon(Retake(from_seq=from_seq))
            entry.publish_soon(Trouble(detail=f"the scene did not open: {failure}"))
        finally:
            # Last, and on every way out: a client left showing that the
            # room is still thinking is worse than one that never saw it
            # thinking at all.
            entry.busy = False
            entry.publish_soon(Settled())

    def scene_tokens() -> list[str]:
        if not worlds_root.is_dir():
            return []
        return [
            f"{world.name}/{scene.stem}"
            for world in sorted(worlds_root.iterdir())
            if (world / "world.yaml").is_file()
            for scene in sorted((world / "scenes").glob("*.yaml"))
        ]

    def resolve_token(token: str | None) -> tuple[str, Session]:
        """A scene token names a scene; the same token keeps returning the
        same session, which is how a stateless client holds a continuing
        story. A session id works too, so a client can attach to a scene
        opened through the native API."""
        if not token:
            raise HTTPException(
                status_code=401,
                detail="set the API key to a scene token, e.g. 'ashgrove/the_dinner'",
            )
        if token in live:
            return token, live[token].session
        if token in by_token and by_token[token] in live:
            session_id = by_token[token]
            return session_id, live[session_id].session

        world_name, separator, scene_name = token.replace(":", "/").partition("/")
        if not separator or not scene_name:
            raise HTTPException(
                status_code=404,
                detail=f"{token!r} is not a scene token; expected '<world>/<scene>'",
            )
        session_id, session = open_session(world_name, scene_name)
        by_token[token] = session_id
        return session_id, session

    add_openai_shim(app, resolve_token, scene_tokens)

    def state_of(session_id: str, session: Session) -> SceneState:
        here = session.here()
        events = session.store.get_events(session.scene.id)
        return SceneState(
            session_id=session_id,
            world=session.world.id,
            scene=session.scene.id,
            mode=session.scene.mode,
            character=session.user_character.id,
            character_name=session.user_character.name,
            location=here,
            location_name=session.world.room_name(here),
            present=[Named(id=c.id, name=c.name) for c in session.present()],
            rooms=[Named(id=r.id, name=r.name) for r in session.world.rooms.values()],
            story_time=events[-1].story_time if events else session.scene.start_time,
            cast=list(session.scene.cast),
            pending_skip_minutes=session.pending_skip(),
            ended=session.ended(),
            steering=session.steering,
            next_scene=session.next_scene(),
            open_ended=session.director.open_ended,
        )

    async def act(session_id: str, operation) -> list[StreamEvent]:
        """Run one blocking engine operation and fan its POV out to
        anyone streaming."""
        entry = get_live(session_id)
        try:
            perceived = await run_in_threadpool(operation, entry.session)
        except ModelUnavailable as failure:
            # The take was rolled back, so the scene is exactly where it
            # was and the client can send the same line again. 502: the
            # thing upstream of us failed, not the request.
            raise HTTPException(status_code=502, detail=f"the model did not answer: {failure}")
        events = to_stream_events(entry.session, perceived)
        entry.publish(events)
        return events

    def _card(card) -> StoryOut:
        # A story open here has a turn still uncommitted — that is what
        # makes the last take discardable — so the file on disk is one
        # turn behind what the player can see. The live session is the
        # newer truth, and reading a card must not contradict the screen.
        open_here = next(
            (entry.session for entry in live.values() if entry.story_id == card.id), None
        )
        return StoryOut(
            id=card.id,
            title=card.title,
            character=card.character,
            open_ended=card.open_ended,
            world=card.world,
            scene=open_here.scene.id if open_here else card.scene,
            created_at=card.created_at,
            played_at=card.played_at,
            turns=open_here.turns_played if open_here else card.turns,
        )

    @app.get("/stories", response_model=list[StoryOut])
    def stories() -> list[StoryOut]:
        """Everything on this machine, most recently played first."""
        return [_card(card) for card in library.list()]

    @app.post("/stories", response_model=SceneState)
    def start_story(body: NewStory, background: BackgroundTasks) -> SceneState:
        player = Player(
            name=(body.character.name if body.character else ""),
            look=(body.character.look if body.character else ""),
        )
        complaint = player.complaint()
        if complaint:
            raise HTTPException(status_code=400, detail=complaint)
        if player.look.strip():
            # A description naming one of the world's own facts would hand
            # a secret to everybody in the room before a word was spoken.
            # Said plainly here rather than quietly dropped: it is the
            # player's sentence, and they should know it did not land.
            try:
                world = load_world(_world_dir(worlds_root, body.world))
            except Exception:
                world = None
            named = world and invents_a_fact(player.look, world)
            if named:
                raise HTTPException(
                    status_code=400,
                    detail="that mentions something the story turns on — "
                           "say it in the scene rather than before it",
                )
        try:
            session = library.start(
                body.world, body.scene, title=body.title, player=player,
                open_ended=body.open_ended, curtain=False, **opening(),
            )
            remember_player(settings, settings_file, player.called, player.look.strip())
        except FileNotFoundError as missing:
            raise HTTPException(status_code=404, detail=str(missing))
        except ModelUnavailable as failure:
            # Starting a story asks the model nothing — the curtain goes
            # up afterwards — so this is a world that could not be loaded
            # rather than a model that would not answer. Kept because the
            # story file is already gone if it fires: a start that fails
            # leaves no card.
            raise HTTPException(status_code=502, detail=f"the model did not answer: {failure}")
        return opened(session, background)

    @app.post("/stories/{story_id}/resume", response_model=SceneState)
    def resume_story(story_id: str, background: BackgroundTasks) -> SceneState:
        """Pick a story up where it was left, on the scene it was left on.

        A story already open here is handed back rather than opened
        again: a second connection to a file whose turn is still open
        would sit waiting on its write lock, and two sessions over one
        story would disagree about what had happened in it.
        """
        for session_id, entry in live.items():
            if entry.story_id == story_id:
                return state_of(session_id, entry.session)
        try:
            session = library.resume(story_id, curtain=False, **opening())
        except (FileNotFoundError, ValueError) as missing:
            raise HTTPException(status_code=404, detail=str(missing))
        except ModelUnavailable as failure:
            raise HTTPException(status_code=502, detail=f"the model did not answer: {failure}")
        # Ordinarily nothing: a scene in progress is not re-described, and
        # there is no opening line for the room to answer. It matters for
        # the story whose curtain never went up — a model that was
        # unreachable at the first click — which opens properly on the
        # next one rather than staying a room nobody ever described.
        return opened(session, background)

    @app.delete("/stories/{story_id}")
    def delete_story(story_id: str) -> dict:
        """Delete a story, and let go of it if it is open.

        Otherwise the session outlives the file it was reading from, and
        the next resume hands back a story the player just deleted.
        """
        for session_id in [sid for sid, entry in live.items() if entry.story_id == story_id]:
            live.pop(session_id).session.store.close()
        try:
            gone = library.delete(story_id)
        except ValueError as bad:
            raise HTTPException(status_code=400, detail=str(bad))
        if not gone:
            raise HTTPException(status_code=404, detail=f"no story {story_id}")
        return {"deleted": story_id}

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def client() -> str:
        """The POV client (spec §12, M7). It renders what the character
        perceived — it has no access to the world log to render instead."""
        page = Path(__file__).parent / "web" / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="client not installed")
        return page.read_text(encoding="utf-8")

    @app.get("/worlds")
    def list_worlds() -> dict[str, list[str]]:
        if not worlds_root.is_dir():
            return {}
        return {
            world.name: sorted(p.stem for p in (world / "scenes").glob("*.yaml"))
            for world in sorted(worlds_root.iterdir())
            if (world / "world.yaml").is_file()
        }

    # A shelf of prose is a terminal with a stylesheet. The category this
    # app is in is one people browse with their eyes, so every world has a
    # cover and everybody in it has a face — the authored file when there
    # is one, and a plate drawn from the id when there is not, which is
    # every world here and every world the generator writes.
    #
    # One URL per thing, always answering, so the client never has to know
    # which of the two it is getting.
    def _art(svg: str) -> Response:
        return Response(
            content=svg,
            media_type="image/svg+xml",
            # Long enough that the shelf does not refetch a dozen plates
            # on every render, short enough that an author who drops a
            # real file in sees it without explaining the cache to
            # themselves.
            headers={"Cache-Control": "public, max-age=60"},
        )

    def _served(world_dir: Path, named: str) -> Response | None:
        found = authored(world_dir, named)
        if found is None:
            return None
        path, kind = found
        return Response(
            content=path.read_bytes(),
            media_type=kind,
            headers={"Cache-Control": "public, max-age=60"},
        )

    @app.get("/worlds/{world}/cover", include_in_schema=False)
    def world_cover(world: str) -> Response:
        world_dir = _world_dir(worlds_root, world)
        try:
            loaded = load_world(world_dir)
        except Exception:
            raise HTTPException(status_code=404, detail=f"no world named {world!r}")
        return _served(world_dir, loaded.image) or _art(cover(loaded.id, loaded.title))

    @app.get("/worlds/{world}/faces/{character}", include_in_schema=False)
    def character_face(world: str, character: str) -> Response:
        world_dir = _world_dir(worlds_root, world)
        try:
            loaded = load_world(world_dir)
            cast = load_characters(world_dir)
        except Exception:
            raise HTTPException(status_code=404, detail=f"no world named {world!r}")
        person = cast.get(character)
        if person is None:
            raise HTTPException(status_code=404, detail=f"nobody called {character!r}")
        # Their place in the cast, which is what keeps a room of five
        # from being one colour in five shades.
        order = list(cast)
        return _served(world_dir, person.image) or _art(
            portrait(loaded.id, person.id, person.name, among=(order.index(character), len(order)))
        )

    @app.get("/settings")
    def read_settings() -> dict:
        """What a story started now would run on, and what the
        environment can and cannot reach. Never a credential: the keys
        this reports are names, and their values are not read here."""
        return dict(describe(settings), pinned=pinned)

    @app.put("/settings")
    def write_settings(body: NewSettings) -> dict:
        """Change it, for the stories opened from now on and for the ones
        already open.

        A model the environment cannot reach is refused rather than
        saved: accepting it would mean every story from here on failing
        at its first turn, several clicks away from the screen that
        caused it.
        """
        if pinned:
            raise HTTPException(
                status_code=409,
                detail="the model was set on the command line for this run; "
                       "restart without --model to choose it here",
            )
        wanted = Settings(
            model=body.model.strip(),
            api_base=body.api_base.strip(),
            fast_model=body.fast_model.strip(),
            fast_api_base=body.fast_api_base.strip(),
            player_name=body.player_name.strip(),
            player_look=body.player_look.strip(),
            interpret=body.interpret,
            direct=body.direct,
            workers=max(1, min(body.workers, 32)),
        )
        unreachable = wanted.unreachable()
        if unreachable:
            raise HTTPException(status_code=400, detail=unreachable)

        settings.model = wanted.model
        settings.api_base = wanted.api_base
        settings.fast_model = wanted.fast_model
        settings.fast_api_base = wanted.fast_api_base
        settings.player_name = wanted.player_name
        settings.player_look = wanted.player_look
        settings.interpret = wanted.interpret
        settings.direct = wanted.direct
        settings.workers = wanted.workers
        settings.save(settings_file)
        # A story already open changes model too. Nothing about it moves:
        # the log, the beliefs and the trust are the engine's; the model
        # is only who gets asked next.
        for entry in live.values():
            entry.session.use(
                settings.client(), settings.workers, settings.interpret, settings.direct
            )
        return dict(describe(settings), pinned=pinned)

    @app.post("/cards", response_model=SceneState)
    async def play_a_card(
        request: Request, background: BackgroundTasks, name: str = ""
    ) -> SceneState:
        """Play a character card from somewhere else.

        The file arrives as the raw body rather than as a form, which
        keeps `python-multipart` out of the install for one upload. What
        the card is allowed to say — and what it is not, which is
        anything shaped like an instruction — is `fabula.cards`.
        """
        data = await request.body()
        try:
            world_dir, scene = card_world(
                cards_read(data), worlds_root, player_name=name
            )
            remember_player(settings, settings_file, name.strip(), "")
        except NotACard as refused:
            raise HTTPException(status_code=422, detail=str(refused))
        complaint = complaints(world_dir)
        if complaint:
            shutil.rmtree(world_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail="; ".join(complaint[:3]))
        return opened(
            library.start(
                world_dir.name, scene, player=Player(name=name),
                curtain=False, **opening(),
            ),
            background,
        )

    @app.get("/worlds/{world}/cards/{character}", include_in_schema=False)
    def character_card(world: str, character: str) -> Response:
        """One of this world's people, as a card the rest of the shelf
        reads — the plate we already draw for them, with the description
        inside it."""
        world_dir = _world_dir(worlds_root, world)
        try:
            loaded, cast = load_world(world_dir), load_characters(world_dir)
        except Exception:
            raise HTTPException(status_code=404, detail=f"no world named {world!r}")
        person = cast.get(character)
        if person is None or person.is_user:
            # The player is whoever is holding them; there is nobody to
            # hand over.
            raise HTTPException(status_code=404, detail=f"nobody to export called {character!r}")
        scenes = sorted((world_dir / "scenes").glob("*.yaml"))
        scene = load_scene(world_dir, scenes[0].stem) if scenes else None
        return Response(
            content=card_png(loaded, person, scene, cast),
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{person.id}.png"',
            },
        )

    @app.post("/invent", response_model=SceneState)
    def invent_world(body: NewWorld, background: BackgroundTasks) -> SceneState:
        """Make a world from a sentence and start a story in it.

        The world is written to the same directory the shipped ones live
        in, as the same YAML, so the shelf lists it beside them and the
        engine plays it without knowing where it came from. Nothing
        reaches disk that `fabula.inspect` has not read.
        """
        chosen = opening()
        if chosen["llm"] is None:
            chosen = dict(chosen, llm=get_default_llm())
        player = Player(
            name=(body.character.name if body.character else ""),
            look=(body.character.look if body.character else ""),
        )
        complaint = player.complaint()
        if complaint:
            raise HTTPException(status_code=400, detail=complaint)
        try:
            made = invent(body.premise, chosen["llm"], worlds_root=worlds_root)
        except CannotInvent as refused:
            raise HTTPException(status_code=422, detail=str(refused))
        except ModelUnavailable as failure:
            raise HTTPException(
                status_code=502, detail=f"the model did not answer: {failure}"
            )
        try:
            session = library.start(
                made.world_dir.name, made.scene, player=player,
                open_ended=body.open_ended, curtain=False, **chosen,
            )
            remember_player(settings, settings_file, player.called, player.look.strip())
        except ModelUnavailable as failure:
            raise HTTPException(
                status_code=502, detail=f"the model did not answer: {failure}"
            )
        return opened(session, background)

    @app.get("/catalogue")
    def shelf() -> list[dict]:
        """Every world and its scenes, with the ones a story starts from
        marked — what a person picking something to play needs to see,
        as opposed to `/worlds`, which is the flat list a tool wants."""
        if not worlds_root.is_dir():
            return []
        return catalogue(worlds_root)

    @app.post("/sessions", response_model=SceneState)
    def create_session(body: NewSession) -> SceneState:
        session_id, session = open_session(body.world, body.scene)
        return state_of(session_id, session)

    @app.get("/sessions/{session_id}", response_model=SceneState)
    def get_state(session_id: str) -> SceneState:
        return state_of(session_id, get_live(session_id).session)

    @app.get("/sessions/{session_id}/events", response_model=list[StreamEvent])
    def history(session_id: str) -> list[StreamEvent]:
        """The scene so far, as this character experienced it."""
        entry = get_live(session_id)
        return to_stream_events(entry.session, entry.session.perceived_so_far())

    @app.post("/sessions/{session_id}/steer")
    def steer(session_id: str, body: Steer) -> dict:
        """Tell the director what kind of story this is.

        No model call and nothing to fail: it is a standing note read by
        the two things that decide what happens to a room and how it is
        described. It never reaches a character, which is the whole
        design — `fabula.steering` says why.
        """
        entry = get_live(session_id)
        return {"note": entry.session.steer(body.note)}

    @app.post("/sessions/{session_id}/rewind", response_model=list[StreamEvent])
    def rewind(session_id: str, body: Rewind) -> list[StreamEvent]:
        """Take the scene back, and hand back what is left of it.

        No model call and nothing to fail upstream, so this is not an
        `act`: it is bookkeeping that happens to be the most useful
        control in the app. The whole history comes back rather than a
        delta, because the client's job here is to redraw rather than to
        append.
        """
        entry = get_live(session_id)
        return to_stream_events(entry.session, entry.session.rewind_to(body.seq))

    @app.post("/sessions/{session_id}/say", response_model=list[StreamEvent])
    async def say(session_id: str, body: Say) -> list[StreamEvent]:
        return await act(session_id, lambda session: session.say(body.text))

    @app.post("/sessions/{session_id}/move", response_model=list[StreamEvent])
    async def move(session_id: str, body: Move) -> list[StreamEvent]:
        entry = get_live(session_id)
        if body.room not in entry.session.world.rooms:
            raise HTTPException(status_code=400, detail=f"no room {body.room!r} in this world")
        return await act(session_id, lambda session: session.move(body.room))

    @app.post("/sessions/{session_id}/wait", response_model=list[StreamEvent])
    async def wait(session_id: str, body: Wait = Body(default=Wait())) -> list[StreamEvent]:
        # Consent travels in the request: a large skip spends the user's
        # character's time, so a client has to have asked first. GET
        # /sessions/{id} reports how long the next skip would be.
        return await act(session_id, lambda session: session.wait(lambda _m: body.consent))

    @app.post("/sessions/{session_id}/look", response_model=list[StreamEvent])
    async def look(session_id: str) -> list[StreamEvent]:
        return await act(session_id, lambda session: session.look())

    @app.post("/sessions/{session_id}/regenerate", response_model=list[StreamEvent])
    async def regenerate(session_id: str) -> list[StreamEvent]:
        """Throw the last take away and play it again.

        The engine never stops to ask whether a change to the world is
        wanted; this is how it gets rejected instead. Everything the
        discarded take wrote is gone — events, beliefs, readings, trust —
        and the retake reuses the sequence numbers it vacated.
        """
        entry = get_live(session_id)
        if not entry.session.can_regenerate():
            raise HTTPException(status_code=409, detail="nothing has been played yet")
        perceived = await run_in_threadpool(lambda s: s.regenerate(), entry.session)
        events = to_stream_events(entry.session, perceived)
        entry.publish([Retake(from_seq=entry.session.turn_started_at), *events])
        return events

    @app.post("/sessions/{session_id}/next", response_model=SceneState)
    async def go_on(session_id: str) -> SceneState:
        """Move the story to the scene this one leads to.

        The session id is stable across the seam: a client holds a story,
        not a scene. Everybody arrives carrying what the last scene did to
        them, because it is the same store underneath.
        """
        entry = get_live(session_id)
        following = await run_in_threadpool(lambda s: s.go_on(), entry.session)
        if following is None:
            raise HTTPException(status_code=409, detail="the story ends here")
        entry.session = following
        entry.publish([Retake(from_seq=0)])  # a new scene: the transcript starts over
        return state_of(session_id, following)

    @app.post("/sessions/{session_id}/reveal", response_model=RevealOut)
    def reveal(session_id: str) -> RevealOut:
        """What the player did not know, once they ask to be told.

        A spoiler by design, and the one endpoint that steps outside the
        requesting character's POV — which is why it is a POST the client
        must deliberately make, not part of scene state. Read-only: it
        appends nothing, so play can continue afterwards.
        """
        session = get_live(session_id).session
        built = build_reveal(session)
        return RevealOut(
            character=built.character.name,
            missed=[
                MissedOut(
                    location=session.world.room_name(event.location_id),
                    speaker=(
                        session.characters[event.actor_id].name
                        if event.actor_id in session.characters
                        else None
                    ),
                    content=event.content,
                )
                for event in built.missed
            ],
            half_heard=[
                HalfHeardOut(heard=heard, truth=event.content) for heard, event in built.half_heard
            ],
            knowledge=built.knowledge,
            regard=[
                RegardOut(
                    who=r.who, toward=r.toward, trust=r.trust, started=r.started
                )
                for r in built.regard
            ],
            text=render(built, session),
        )

    @app.get("/sessions/{session_id}/stream")
    async def stream(session_id: str, follow: bool = True) -> StreamingResponse:
        """The scene as this character experiences it.

        Sends everything they have already perceived, then follows live.
        `follow=false` closes after the catch-up instead, for clients that
        would rather poll than hold a connection open.
        """
        entry = get_live(session_id)
        queue: asyncio.Queue[StreamEvent | Retake | Trouble | Settled] = asyncio.Queue()
        if follow:
            entry.loop = asyncio.get_running_loop()
            entry.subscribers.append(queue)
            # Only while somebody is looking: a session nobody is
            # streaming pays nothing for this.
            entry.session.watch(
                lambda key: entry.publish_soon(
                    Working(
                        who=key,
                        name=(
                            entry.session.characters[key].name
                            if key in entry.session.characters
                            else ""
                        ),
                    )
                )
            )

        async def frames():
            try:
                for event in to_stream_events(entry.session, entry.session.perceived_so_far()):
                    yield f"data: {event.model_dump_json()}\n\n"
                yield ": caught up\n\n"
                # A scene that opened itself before anybody was listening
                # published its "done" to nobody. Said again here, so a
                # client that shows the room thinking while it waits is
                # never left showing it forever.
                if follow and not entry.busy:
                    yield f"event: settled\ndata: {Settled().model_dump_json()}\n\n"
                while follow:
                    event = await queue.get()
                    if isinstance(event, Retake):
                        yield f"event: retake\ndata: {event.model_dump_json()}\n\n"
                    elif isinstance(event, Settled):
                        yield f"event: settled\ndata: {event.model_dump_json()}\n\n"
                    elif isinstance(event, Trouble):
                        yield f"event: trouble\ndata: {event.model_dump_json()}\n\n"
                    elif isinstance(event, Working):
                        yield f"event: working\ndata: {event.model_dump_json()}\n\n"
                    else:
                        yield f"data: {event.model_dump_json()}\n\n"
            finally:
                if queue in entry.subscribers:
                    entry.subscribers.remove(queue)
                if not entry.subscribers:
                    entry.session.watch(None)

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def free_port(host: str, wanted: int) -> int:
    """`wanted` if it is free, otherwise one the operating system picks.

    Somebody who has left a story open in another window should not be
    told to go and find it and close it: the second one takes another
    port and says which.
    """
    import socket

    with socket.socket() as probe:
        try:
            probe.bind((host, wanted))
            # What was actually bound, which for port 0 is the one the
            # operating system chose — otherwise the address printed on
            # the way up would be a lie.
            return probe.getsockname()[1]
        except OSError:
            pass
    with socket.socket() as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def serve(
    worlds_root: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    model: str | None = None,
    api_base: str | None = None,
    workers: int | None = None,
    library_root: Path | None = None,
    settings_path: Path | None = None,
    open_browser: bool = False,
    fast_model: str | None = None,
    fast_api_base: str | None = None,
) -> None:
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - depends on the install
        raise SystemExit("serving needs uvicorn: pip install uvicorn")
    llm = None
    if fast_model and not model:
        raise SystemExit(
            "fabula-serve: --fast-model needs --model: it is for the calls the "
            "first model would otherwise make, so on its own it changes nothing"
        )
    if model:
        # Both models, before anything opens. A second model is a second
        # way for a story to end in a provider traceback several turns in.
        for named, base in (
            (model, api_base), (fast_model, fast_api_base or api_base)
        ):
            unreachable = named and missing_credentials(named, base)
            if unreachable:
                raise SystemExit(f"fabula-serve: {unreachable}")
        llm = LiteLLMClient(model=model, api_base=api_base)
        if fast_model:
            llm = Routed(
                llm,
                LiteLLMClient(model=fast_model, api_base=fast_api_base or api_base),
            )
    port = free_port(host, port)
    where = f"http://{host}:{port}"
    if open_browser:
        # After the server is listening, in a thread, and never fatal: a
        # machine with no browser to open (a container, a server over
        # ssh) still gets a running app and the address to reach it at.
        import threading
        import webbrowser

        threading.Timer(0.7, lambda: webbrowser.open(where)).start()
    print(f"Fabula is at {where}   (ctrl-c to stop)")
    uvicorn.run(
        create_app(
            worlds_root,
            llm=llm,
            workers=workers,
            library_root=library_root,
            settings_path=settings_path,
        ),
        host=host,
        port=port,
        log_level="warning",
    )


def main(argv: list[str] | None = None) -> None:
    # Read a local .env first, so a key in the file is available to
    # everything below. Only in an entry point: importing a library
    # should never mutate the process environment.
    load_env()
    import argparse

    parser = argparse.ArgumentParser(prog="fabula-serve", description="Run the Fabula service.")
    parser.add_argument(
        "--worlds", type=Path, default=None,
        help="Where worlds live (default ~/.fabula/worlds)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None, help="Any model id litellm understands")
    parser.add_argument("--api-base", default=None, help="An OpenAI-compatible endpoint")
    parser.add_argument(
        "--fast-model", default=None, metavar="MODEL",
        help="A cheaper model for the calls nobody reads — most of a turn",
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
    args = parser.parse_args(argv)
    serve(
        args.worlds, args.host, args.port, args.model, args.api_base, args.workers,
        fast_model=args.fast_model, fast_api_base=args.fast_api_base,
    )


if __name__ == "__main__":
    main()
