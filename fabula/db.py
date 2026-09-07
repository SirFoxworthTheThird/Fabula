"""SQLite-backed storage: the append-only event log, plus the belief store.

The `events` table is the fabula. It is never updated or deleted from —
enforced both by this module's API (no update/delete methods exist) and,
as a second line of defense, by SQLite triggers that abort any attempt.

A sqlite3 connection is bound to its creating thread unless told
otherwise, and the HTTP service touches the store from both the event
loop and a worker thread. The connection is therefore opened for shared
use and every method holds a lock for the whole statement — cursors are
drained inside the lock, never handed out to be read later.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime

from fabula.models import Belief, Event, Relationship

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    story_time TEXT NOT NULL,
    kind TEXT NOT NULL,
    actor_id TEXT,
    location_id TEXT NOT NULL,
    content TEXT NOT NULL,
    audibility TEXT NOT NULL,
    addressed_to TEXT NOT NULL,
    salience_base REAL NOT NULL,
    detail_level TEXT NOT NULL,
    metadata TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS events_no_update
BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events table is append-only: updates are forbidden');
END;

CREATE TRIGGER IF NOT EXISTS events_no_delete
BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events table is append-only: deletes are forbidden');
END;

-- Durable across scenes, and keyed by character rather than scene. A
-- belief store is allowed to hold things that are false; nothing here
-- ever reconciles it against world state.
CREATE TABLE IF NOT EXISTS beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    character_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_event_id INTEGER,
    formed_at TEXT NOT NULL,
    last_rehearsed TEXT NOT NULL,
    salience REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS beliefs_one_per_source
ON beliefs (character_id, source_event_id);

CREATE TABLE IF NOT EXISTS relationships (
    character_id TEXT NOT NULL,
    toward_id TEXT NOT NULL,
    affinity REAL NOT NULL,
    trust REAL NOT NULL,
    note TEXT NOT NULL,
    interactions INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (character_id, toward_id)
);

CREATE TABLE IF NOT EXISTS resolved_goals (
    character_id TEXT NOT NULL,
    goal_id TEXT NOT NULL,
    PRIMARY KEY (character_id, goal_id)
);

-- Derived, per-character, and keyed by a hash of the exact perceived
-- content it covers: the same span always resolves to the same stored
-- row, so a summary is computed once and never recomputed on the fly.
CREATE TABLE IF NOT EXISTS summaries (
    character_id TEXT NOT NULL,
    scene_id TEXT NOT NULL,
    span_key TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    PRIMARY KEY (character_id, scene_id, span_key)
);

CREATE TABLE IF NOT EXISTS rehearsals (
    character_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    last_rehearsed_seq INTEGER NOT NULL,
    PRIMARY KEY (character_id, event_id)
);
"""


class EventStore:
    def __init__(self, path: str = ":memory:"):
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def next_seq(self, scene_id: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT MAX(seq) FROM events WHERE scene_id = ?", (scene_id,)
            ).fetchone()
            return (row[0] or 0) + 1

    def append_event(self, event: Event) -> Event:
        with self._lock:
            if event.id is not None:
                raise ValueError("events are append-only; do not pass a pre-assigned id")
            cur = self.conn.execute(
                """INSERT INTO events
                   (scene_id, seq, story_time, kind, actor_id, location_id, content,
                    audibility, addressed_to, salience_base, detail_level, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.scene_id,
                    event.seq,
                    event.story_time.isoformat(),
                    event.kind,
                    event.actor_id,
                    event.location_id,
                    event.content,
                    event.audibility,
                    json.dumps(event.addressed_to),
                    event.salience_base,
                    event.detail_level,
                    json.dumps(event.metadata),
                ),
            )
            self.conn.commit()
            return event.model_copy(update={"id": cur.lastrowid})

    def get_events(self, scene_id: str) -> list[Event]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM events WHERE scene_id = ? ORDER BY seq ASC", (scene_id,)
            ).fetchall()
            return [_row_to_event(r) for r in rows]

    def add_belief(self, belief: Belief) -> None:
        with self._lock:
            # One belief per character per source event: encoding the same
            # moment twice would let a memory quietly gain weight on replay.
            self.conn.execute(
                """INSERT OR IGNORE INTO beliefs
                   (character_id, subject_id, content, confidence, source_event_id,
                    formed_at, last_rehearsed, salience)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    belief.character_id,
                    belief.subject_id,
                    belief.content,
                    belief.confidence,
                    belief.source_event_id,
                    belief.formed_at.isoformat(),
                    belief.last_rehearsed.isoformat(),
                    belief.salience,
                ),
            )
            self.conn.commit()

    def get_beliefs(self, character_id: str) -> list[Belief]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM beliefs WHERE character_id = ?", (character_id,)
            ).fetchall()
            return [
                Belief(
                    id=r["id"],
                    character_id=r["character_id"],
                    subject_id=r["subject_id"],
                    content=r["content"],
                    confidence=r["confidence"],
                    source_event_id=r["source_event_id"],
                    formed_at=datetime.fromisoformat(r["formed_at"]),
                    last_rehearsed=datetime.fromisoformat(r["last_rehearsed"]),
                    salience=r["salience"],
                )
                for r in rows
            ]


    def set_belief_salience(self, belief_id: int, salience: float) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE beliefs SET salience = ? WHERE id = ?", (salience, belief_id)
            )
            self.conn.commit()

    def get_relationships(self, character_id: str) -> dict[str, Relationship]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM relationships WHERE character_id = ?", (character_id,)
            ).fetchall()
            return {
                r["toward_id"]: Relationship(
                    affinity=r["affinity"],
                    trust=r["trust"],
                    note=r["note"],
                    interactions=r["interactions"],
                )
                for r in rows
            }

    def seed_relationship(
        self, character_id: str, toward_id: str, relationship: Relationship
    ) -> None:
        """Authored starting state. Idempotent: once a relationship has
        been carried into the store it is the character's own, and
        re-seeding must not reset what play has done to it."""
        with self._lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO relationships
                   (character_id, toward_id, affinity, trust, note, interactions)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    character_id,
                    toward_id,
                    relationship.affinity,
                    relationship.trust,
                    relationship.note,
                    relationship.interactions,
                ),
            )
            self.conn.commit()

    def bump_interaction(self, character_id: str, toward_id: str) -> None:
        with self._lock:
            self.conn.execute(
                """UPDATE relationships SET interactions = interactions + 1
                   WHERE character_id = ? AND toward_id = ?""",
                (character_id, toward_id),
            )
            self.conn.commit()

    def resolve_goal(self, character_id: str, goal_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO resolved_goals (character_id, goal_id) VALUES (?, ?)",
                (character_id, goal_id),
            )
            self.conn.commit()

    def get_resolved_goals(self, character_id: str) -> set[str]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT goal_id FROM resolved_goals WHERE character_id = ?", (character_id,)
            ).fetchall()
            return {r["goal_id"] for r in rows}

    def get_summary(self, character_id: str, scene_id: str, span_key: str) -> str | None:
        with self._lock:
            row = self.conn.execute(
                """SELECT summary_text FROM summaries
                   WHERE character_id = ? AND scene_id = ? AND span_key = ?""",
                (character_id, scene_id, span_key),
            ).fetchone()
            return row["summary_text"] if row else None

    def put_summary(self, character_id: str, scene_id: str, span_key: str, text: str) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO summaries
                   (character_id, scene_id, span_key, summary_text) VALUES (?, ?, ?, ?)""",
                (character_id, scene_id, span_key, text),
            )
            self.conn.commit()

    def get_rehearsals(self, character_id: str) -> dict[int, int]:
        """event_id -> seq of the most recent event that re-mentioned it."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT event_id, last_rehearsed_seq FROM rehearsals WHERE character_id = ?",
                (character_id,),
            ).fetchall()
            return {r["event_id"]: r["last_rehearsed_seq"] for r in rows}

    def record_rehearsal(self, character_id: str, event_id: int, seq: int) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT INTO rehearsals (character_id, event_id, last_rehearsed_seq)
                   VALUES (?, ?, ?)
                   ON CONFLICT (character_id, event_id)
                   DO UPDATE SET last_rehearsed_seq = max(last_rehearsed_seq, excluded.last_rehearsed_seq)""",
                (character_id, event_id, seq),
            )
            self.conn.commit()


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
        id=row["id"],
        scene_id=row["scene_id"],
        seq=row["seq"],
        story_time=datetime.fromisoformat(row["story_time"]),
        kind=row["kind"],
        actor_id=row["actor_id"],
        location_id=row["location_id"],
        content=row["content"],
        audibility=row["audibility"],
        addressed_to=json.loads(row["addressed_to"]),
        salience_base=row["salience_base"],
        detail_level=row["detail_level"],
        metadata=json.loads(row["metadata"]),
    )
