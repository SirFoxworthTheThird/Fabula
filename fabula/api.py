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
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from fabula.env import load_env
from fabula.llm import LiteLLMClient, LLMClient
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


class Named(BaseModel):
    id: str
    name: str


class SceneState(BaseModel):
    session_id: str
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


class NewSession(BaseModel):
    world: str
    scene: str


class Say(BaseModel):
    text: str


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

    def publish(self, events: list) -> None:
        for queue in self.subscribers:
            for event in events:
                queue.put_nowait(event)


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


def create_app(
    worlds_root: Path = Path("worlds"),
    llm: LLMClient | None = None,
    db_path: str = ":memory:",
) -> FastAPI:
    app = FastAPI(title="Fabula", description="A multi-agent story engine.")
    live: dict[str, _LiveSession] = {}
    by_token: dict[str, str] = {}
    app.state.sessions = live
    app.state.sessions_by_token = by_token

    def get_live(session_id: str) -> _LiveSession:
        if session_id not in live:
            raise HTTPException(status_code=404, detail="no such session")
        return live[session_id]

    def open_session(world_name: str, scene_name: str) -> tuple[str, Session]:
        world_dir = _world_dir(worlds_root, world_name)
        try:
            session = Session.open(world_dir, scene_name, db_path=db_path, llm=llm)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no scene named {scene_name!r}")
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        session_id = uuid.uuid4().hex
        live[session_id] = _LiveSession(session)
        return session_id, session

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
        )

    async def act(session_id: str, operation) -> list[StreamEvent]:
        """Run one blocking engine operation and fan its POV out to
        anyone streaming."""
        entry = get_live(session_id)
        perceived = await run_in_threadpool(operation, entry.session)
        events = to_stream_events(entry.session, perceived)
        entry.publish(events)
        return events

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
        queue: asyncio.Queue[StreamEvent | Retake] = asyncio.Queue()
        if follow:
            entry.subscribers.append(queue)

        async def frames():
            try:
                for event in to_stream_events(entry.session, entry.session.perceived_so_far()):
                    yield f"data: {event.model_dump_json()}\n\n"
                yield ": caught up\n\n"
                while follow:
                    event = await queue.get()
                    if isinstance(event, Retake):
                        yield f"event: retake\ndata: {event.model_dump_json()}\n\n"
                    else:
                        yield f"data: {event.model_dump_json()}\n\n"
            finally:
                if queue in entry.subscribers:
                    entry.subscribers.remove(queue)

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def serve(
    worlds_root: Path = Path("worlds"),
    host: str = "127.0.0.1",
    port: int = 8000,
    model: str | None = None,
    api_base: str | None = None,
) -> None:
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - depends on the install
        raise SystemExit("serving needs uvicorn: pip install uvicorn")
    llm = LiteLLMClient(model=model, api_base=api_base) if model else None
    uvicorn.run(create_app(worlds_root, llm=llm), host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    # Read a local .env first, so a key in the file is available to
    # everything below. Only in an entry point: importing a library
    # should never mutate the process environment.
    load_env()
    import argparse

    parser = argparse.ArgumentParser(prog="fabula-serve", description="Run the Fabula service.")
    parser.add_argument("--worlds", type=Path, default=Path("worlds"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None, help="Any model id litellm understands")
    parser.add_argument("--api-base", default=None, help="An OpenAI-compatible endpoint")
    args = parser.parse_args(argv)
    serve(args.worlds, args.host, args.port, args.model, args.api_base)


if __name__ == "__main__":
    main()
