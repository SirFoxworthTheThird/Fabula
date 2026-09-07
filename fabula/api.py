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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from fabula.llm import LLMClient
from fabula.models import ProjectedEvent
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


class SceneState(BaseModel):
    session_id: str
    scene: str
    mode: str
    character: str
    character_name: str
    location: str
    location_name: str
    cast: list[str]
    pending_skip_minutes: int | None


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

    def publish(self, events: list[StreamEvent]) -> None:
        for queue in self.subscribers:
            for event in events:
                queue.put_nowait(event)


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
    app.state.sessions = live

    def get_live(session_id: str) -> _LiveSession:
        if session_id not in live:
            raise HTTPException(status_code=404, detail="no such session")
        return live[session_id]

    def state_of(session_id: str, session: Session) -> SceneState:
        here = session.here()
        return SceneState(
            session_id=session_id,
            scene=session.scene.id,
            mode=session.scene.mode,
            character=session.user_character.id,
            character_name=session.user_character.name,
            location=here,
            location_name=session.world.room_name(here),
            cast=list(session.scene.cast),
            pending_skip_minutes=session.pending_skip(),
        )

    async def act(session_id: str, operation) -> list[StreamEvent]:
        """Run one blocking engine operation and fan its POV out to
        anyone streaming."""
        entry = get_live(session_id)
        perceived = await run_in_threadpool(operation, entry.session)
        events = to_stream_events(entry.session, perceived)
        entry.publish(events)
        return events

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
        world_dir = _world_dir(worlds_root, body.world)
        try:
            session = Session.open(world_dir, body.scene, db_path=db_path, llm=llm)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"no scene named {body.scene!r}")
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))

        session_id = uuid.uuid4().hex
        live[session_id] = _LiveSession(session)
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

    @app.get("/sessions/{session_id}/stream")
    async def stream(session_id: str, follow: bool = True) -> StreamingResponse:
        """The scene as this character experiences it.

        Sends everything they have already perceived, then follows live.
        `follow=false` closes after the catch-up instead, for clients that
        would rather poll than hold a connection open.
        """
        entry = get_live(session_id)
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        if follow:
            entry.subscribers.append(queue)

        async def frames():
            try:
                for event in to_stream_events(entry.session, entry.session.perceived_so_far()):
                    yield f"data: {event.model_dump_json()}\n\n"
                yield ": caught up\n\n"
                while follow:
                    event = await queue.get()
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


def serve(worlds_root: Path = Path("worlds"), host: str = "127.0.0.1", port: int = 8000) -> None:
    try:
        import uvicorn
    except ImportError:  # pragma: no cover - depends on the install
        raise SystemExit("serving needs uvicorn: pip install uvicorn")
    uvicorn.run(create_app(worlds_root), host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="fabula-serve", description="Run the Fabula service.")
    parser.add_argument("--worlds", type=Path, default=Path("worlds"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    serve(args.worlds, args.host, args.port)


if __name__ == "__main__":
    main()
