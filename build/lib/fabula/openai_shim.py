"""OpenAI-compatible shim (spec §12, M8).

Lets a client built for `/v1/chat/completions` — SillyTavern is the one
this exists for — drive a scene. It is deliberately *lossy*: a chat
completion is a flat exchange between one user and one assistant, so a
whole turn of several characters at graded perception collapses into a
single block of text on the way out.

Two things it will not do, both load-bearing:

- **Incoming history is ignored except the last user message.** The
  client's transcript, its system prompt, its character card: all
  discarded. The scene's memory is the event log, and a client that
  could prepend context would be able to hand a character knowledge the
  world model never gave them.
- **It answers with the user character's POV**, the same projection every
  other client gets, so the shim cannot become the one door that leaks.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Iterable

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fabula.llm import ModelUnavailable
from fabula.models import ProjectedEvent
from fabula.session import Session

# Resolves a scene token (or None) to (session_id, Session).
Resolver = Callable[[str | None], tuple[str, Session]]


class ChatMessage(BaseModel):
    role: str = "user"
    content: Any = ""


class ChatRequest(BaseModel):
    """Only `messages` is really read. Everything an OpenAI client sends
    besides — temperature, max_tokens, penalties, the lot — is accepted
    and ignored rather than rejected, or no real client would connect."""

    model: str | None = None
    messages: list[ChatMessage] = []
    stream: bool = False


def message_text(content: Any) -> str:
    """OpenAI content is a string, or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def last_user_message(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            text = message_text(message.content).strip()
            if text:
                return text
    return ""


def render_turn(session: Session, perceived: list[ProjectedEvent]) -> str:
    """Flatten a turn's POV into the one text blob the protocol allows.

    This is where the fidelity goes: speaker, location and visibility
    become typography. A half-heard event stays unattributed, because the
    character still does not know who it was.
    """
    lines: list[str] = []
    for projected in perceived:
        event = projected.event
        if event.actor_id == session.user_character.id:
            continue  # the client already shows what the user sent
        if projected.perception == "degraded" or event.kind != "utterance":
            lines.append(f"*{projected.perceived_content}*")
        else:
            speaker = session.characters.get(event.actor_id)
            name = speaker.name if speaker else (event.actor_id or "")
            lines.append(f"{name}: {projected.perceived_content}")
    return "\n\n".join(lines) or "*(no one answers)*"


def _completion(model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        # Reported as zero rather than guessed: the tokens this scene
        # actually spent are the engine's business, not this exchange's.
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _chunks(model: str, pieces: Iterable[str]) -> Iterable[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def frame(delta: dict, finish: str | None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    yield frame({"role": "assistant"}, None)
    for piece in pieces:
        yield frame({"content": piece}, None)
    yield frame({}, "stop")
    yield "data: [DONE]\n\n"


def scene_label(session: Session) -> str:
    """The scene token form, `<world>/<scene>` — safe to show a client."""
    return f"{session.scene.world}/{session.scene.id}"


def bearer_token(authorization: str | None) -> str | None:
    """The API-key field doubles as the scene token (spec §12)."""
    if not authorization:
        return None
    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[len("bearer ") :].strip()
    return token or None


def add_openai_shim(
    app: FastAPI,
    resolve: Resolver,
    list_scene_tokens: Callable[[], list[str]],
) -> None:
    @app.get("/v1/models", include_in_schema=False)
    def models() -> dict:
        """Every playable scene, as a model id. A client's model picker
        becomes the scene picker."""
        return {
            "object": "list",
            "data": [
                {"id": token, "object": "model", "created": 0, "owned_by": "fabula"}
                for token in list_scene_tokens()
            ],
        }

    def run(token: str | None, body: ChatRequest):
        _session_id, session = resolve(token)
        text = last_user_message(body.messages)
        if not text:
            raise HTTPException(status_code=400, detail="no user message to act on")

        try:
            perceived = session.say(text)
        except ModelUnavailable as failure:
            raise HTTPException(
                status_code=502, detail=f"the model did not answer: {failure}"
            )
        content = render_turn(session, perceived)
        # Always the scene token, never the session id: a session id is
        # accepted as a bearer token, so echoing one into every response
        # would hand out a working credential.
        model = scene_label(session)

        if body.stream:
            pieces = [line + "\n\n" for line in content.split("\n\n")]
            return StreamingResponse(
                _chunks(model, pieces),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return _completion(model, content)

    @app.post("/v1/chat/completions", include_in_schema=False)
    def chat_completions(
        body: ChatRequest, authorization: str | None = Header(default=None)
    ):
        # Identity comes from the API-key field, falling back to the model
        # id so a client that only offers a model picker still works.
        return run(bearer_token(authorization) or body.model, body)

    @app.post("/v1/{world}/{scene}/chat/completions", include_in_schema=False)
    def chat_completions_for_scene(
        world: str,
        scene: str,
        body: ChatRequest,
        authorization: str | None = Header(default=None),
    ):
        """Per-scene URL path, for clients that cannot set a key."""
        return run(f"{world}/{scene}", body)
