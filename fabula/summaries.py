"""Per-character summaries: the one place the memory projector is allowed
to call a model (spec §4).

Two properties matter here and both are structural rather than prompted:

1. A summary is built only from `ProjectedEvent.perceived_content`. The
   raw `Event.content` is never read, so a summary cannot reintroduce
   what the projection already excluded or degraded.
2. A summary is computed once and stored, keyed by a hash of the exact
   perceived content it covers (spec §6 rule 5). Re-summarizing on every
   turn would let a character's memory drift on its own and would destroy
   the reproducibility the leak tests depend on.
"""
from __future__ import annotations

import hashlib
import json

from fabula.db import EventStore
from fabula.llm import LLMClient
from fabula.models import Character, ProjectedEvent


def span_key(character_id: str, span: list[ProjectedEvent]) -> str:
    """Stable identity for a span of one character's perceived events."""
    payload = json.dumps(
        [[p.event.id, p.event.seq, p.perceived_content, p.perception] for p in span],
        sort_keys=True,
    )
    return hashlib.sha256(f"{character_id}\n{payload}".encode()).hexdigest()


def get_or_create_summary(
    character: Character,
    span: list[ProjectedEvent],
    scene_id: str,
    store: EventStore,
    llm: LLMClient,
) -> str:
    key = span_key(character.id, span)
    stored = store.get_summary(character.id, scene_id, key)
    if stored is not None:
        return stored

    text = _summarize(character, span, llm)
    store.put_summary(character.id, scene_id, key, text)
    return text


def _summarize(character: Character, span: list[ProjectedEvent], llm: LLMClient) -> str:
    lines = "\n".join(f"- {p.perceived_content}" for p in span)
    system = (
        f"You compress one character's memory. Everything you are given is what "
        f"{character.name} personally perceived — some of it may have been unclear "
        "to them. Compress it into two or three sentences in the third person, "
        "preserving who did what. Never add a detail that is not in the lines given, "
        "and never resolve something they perceived as unclear into something specific."
    )
    prompt = f"What {character.name} perceived:\n{lines}\n\nCompress this."
    return llm.complete(system=system, prompt=prompt, key=f"summary:{character.id}")
