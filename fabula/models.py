"""Pydantic models for the fabula (event log), character state, and beliefs.

Vocabulary note: `fabula` is the canonical, append-only event log. A
`syuzhet` is a single character's filtered, ordered projection of it.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EventKind = Literal[
    "utterance", "action", "narration", "arrival",
    "departure", "time_skip", "state_change",
]
Audibility = Literal["private", "room", "adjacent", "building"]
PerceptionLevel = Literal["full", "degraded", "none"]


class Event(BaseModel):
    id: int | None = None
    scene_id: str
    seq: int
    story_time: datetime
    kind: EventKind
    actor_id: str | None = None
    location_id: str
    content: str
    audibility: Audibility = "room"
    addressed_to: list[str] = Field(default_factory=list)
    salience_base: float = 0.5
    detail_level: Literal["full", "summary"] = "full"
    metadata: dict = Field(default_factory=dict)


class Goal(BaseModel):
    id: str
    description: str
    priority: float = 0.5
    resolved: bool = False


class Traits(BaseModel):
    talkativeness: float = 0.5
    reactivity: dict[str, float] = Field(default_factory=dict)
    salience_bias: dict[str, float] = Field(default_factory=dict)
    reticence: float = 0.3


class Character(BaseModel):
    id: str
    name: str
    persona: str
    traits: Traits = Field(default_factory=Traits)
    goals: list[Goal] = Field(default_factory=list)
    location_id: str
    is_user: bool = False


class Belief(BaseModel):
    character_id: str
    subject_id: str
    content: str
    confidence: float
    source_event_id: int | None
    formed_at: datetime
    last_rehearsed: datetime
    salience: float


class Pressure(BaseModel):
    id: str
    intent: str                   # director-only: authored, never enters a projection
    trigger: dict = Field(default_factory=dict)
    effect: dict = Field(default_factory=dict)
    weight: float = 0.5           # how hard this competes against character bids
    cooldown_turns: int = 0
    max_fires: int = 1


class Bid(BaseModel):
    character_id: str
    desire: float
    one_line_reason: str


class ProjectedEvent(BaseModel):
    """One event as seen through a single character's perception."""
    event: Event
    perceived_content: str
    perception: PerceptionLevel
