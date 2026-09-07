"""SQLite-backed storage: the append-only event log, plus the belief store.

The `events` table is the fabula. It is never updated or deleted from —
enforced both by this module's API (no update/delete methods exist) and,
as a second line of defense, by SQLite triggers that abort any attempt.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from fabula.models import Belief, Event

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

CREATE TABLE IF NOT EXISTS beliefs (
    character_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    content TEXT NOT NULL,
    confidence REAL NOT NULL,
    source_event_id INTEGER,
    formed_at TEXT NOT NULL,
    last_rehearsed TEXT NOT NULL,
    salience REAL NOT NULL
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
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def next_seq(self, scene_id: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(seq) FROM events WHERE scene_id = ?", (scene_id,)
        ).fetchone()
        return (row[0] or 0) + 1

    def append_event(self, event: Event) -> Event:
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
        rows = self.conn.execute(
            "SELECT * FROM events WHERE scene_id = ? ORDER BY seq ASC", (scene_id,)
        ).fetchall()
        return [_row_to_event(r) for r in rows]

    def add_belief(self, belief: Belief) -> None:
        self.conn.execute(
            """INSERT INTO beliefs
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
        rows = self.conn.execute(
            "SELECT * FROM beliefs WHERE character_id = ?", (character_id,)
        ).fetchall()
        return [
            Belief(
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


    def get_summary(self, character_id: str, scene_id: str, span_key: str) -> str | None:
        row = self.conn.execute(
            """SELECT summary_text FROM summaries
               WHERE character_id = ? AND scene_id = ? AND span_key = ?""",
            (character_id, scene_id, span_key),
        ).fetchone()
        return row["summary_text"] if row else None

    def put_summary(self, character_id: str, scene_id: str, span_key: str, text: str) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO summaries
               (character_id, scene_id, span_key, summary_text) VALUES (?, ?, ?, ?)""",
            (character_id, scene_id, span_key, text),
        )
        self.conn.commit()

    def get_rehearsals(self, character_id: str) -> dict[int, int]:
        """event_id -> seq of the most recent event that re-mentioned it."""
        rows = self.conn.execute(
            "SELECT event_id, last_rehearsed_seq FROM rehearsals WHERE character_id = ?",
            (character_id,),
        ).fetchall()
        return {r["event_id"]: r["last_rehearsed_seq"] for r in rows}

    def record_rehearsal(self, character_id: str, event_id: int, seq: int) -> None:
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
