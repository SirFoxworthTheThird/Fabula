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
    interpretation TEXT NOT NULL DEFAULT '',
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

-- Derived and keyed the same way a summary is: a hash of the exact
-- perceived lines it was read from. The same run-up always resolves to
-- the same reading, so a character's memory of a moment is written once
-- and never drifts underneath them.
CREATE TABLE IF NOT EXISTS interpretations (
    character_id TEXT NOT NULL,
    span_key TEXT NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (character_id, span_key)
);

-- Rooms the player walked into that the author did not write. Kept per
-- world rather than per scene: a corridor found on the first evening is
-- still there on the second, which is the whole reason to store it
-- rather than regenerate it.
CREATE TABLE IF NOT EXISTS discovered_rooms (
    world_id TEXT NOT NULL,
    room_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    reached_from TEXT NOT NULL,
    PRIMARY KEY (world_id, room_id)
);

-- One row, describing the story this file holds. Kept inside the file
-- rather than in a separate index, so a library is a directory you can
-- browse, back up, copy between machines and delete with `rm` — and
-- there is no second place to fall out of sync with.
CREATE TABLE IF NOT EXISTS story (
    only_row INTEGER PRIMARY KEY CHECK (only_row = 1),
    story_id TEXT NOT NULL,
    title TEXT NOT NULL,
    world TEXT NOT NULL,
    scene TEXT NOT NULL,
    created_at TEXT NOT NULL,
    played_at TEXT NOT NULL,
    turns INTEGER NOT NULL DEFAULT 0
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
        # A turn is an open transaction, which holds a write lock. Another
        # holder of the same file should wait for it rather than fail on
        # the spot — a story mid-turn is a normal thing to walk into.
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=5.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self._turn_open = False
        self.conn.commit()

    # --- One turn, one unit of work -------------------------------------
    #
    # A turn writes far more than events: beliefs, rehearsals, summaries,
    # readings, interaction counts, trust. All of it lives in this one
    # connection, so a savepoint around a turn is a complete undo with no
    # per-subsystem bookkeeping — and the discarded turn leaves no trace,
    # so `next_seq` hands the retake the same slot.
    #
    # This is why writes no longer commit one at a time: a commit ends the
    # transaction and takes the savepoint with it. The turn's writes stay
    # uncommitted until the next turn opens, which is also what makes them
    # discardable.

    def _commit(self) -> None:
        if not self._turn_open:
            self.conn.commit()

    def begin_turn(self) -> None:
        """Open a turn, making whatever came before it permanent."""
        with self._lock:
            self.commit_turn()
            self.conn.execute("SAVEPOINT fabula_turn")
            self._turn_open = True

    def commit_turn(self) -> None:
        """Settle the open turn. After this it can no longer be discarded."""
        with self._lock:
            if self._turn_open:
                self.conn.execute("RELEASE fabula_turn")
                self._turn_open = False
            self.conn.commit()

    def rollback_turn(self) -> bool:
        """Discard everything the open turn wrote. False if none is open."""
        with self._lock:
            if not self._turn_open:
                return False
            self.conn.execute("ROLLBACK TO fabula_turn")
            self.conn.execute("RELEASE fabula_turn")
            self._turn_open = False
            self.conn.commit()
            return True

    def _migrate(self) -> None:
        """Bring a database written by an older build up to date.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        exists, so a scene file from before interpretations were added
        would otherwise fail on the first read. Characters are meant to
        be durable across runs; that has to survive an upgrade too.
        """
        columns = {
            row["name"] for row in self.conn.execute("PRAGMA table_info(beliefs)").fetchall()
        }
        if "interpretation" not in columns:
            self.conn.execute(
                "ALTER TABLE beliefs ADD COLUMN interpretation TEXT NOT NULL DEFAULT ''"
            )

    def close(self) -> None:
        with self._lock:
            self.commit_turn()  # never lose a finished turn to a close
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
            self._commit()
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
                   (character_id, subject_id, content, interpretation, confidence,
                    source_event_id, formed_at, last_rehearsed, salience)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    belief.character_id,
                    belief.subject_id,
                    belief.content,
                    belief.interpretation,
                    belief.confidence,
                    belief.source_event_id,
                    belief.formed_at.isoformat(),
                    belief.last_rehearsed.isoformat(),
                    belief.salience,
                ),
            )
            self._commit()

    def get_beliefs(self, character_id: str) -> list[Belief]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM beliefs WHERE character_id = ?", (character_id,)
            ).fetchall()
            return [_row_to_belief(r) for r in rows]


    def set_belief_salience(self, belief_id: int, salience: float) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE beliefs SET salience = ? WHERE id = ?", (salience, belief_id)
            )
            self._commit()

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
            self._commit()

    def set_trust(self, character_id: str, toward_id: str, trust: float) -> None:
        """How far one character now believes another. Written only from
        that character's own projection — see `fabula.persistence`."""
        with self._lock:
            self.conn.execute(
                """UPDATE relationships SET trust = ?
                   WHERE character_id = ? AND toward_id = ?""",
                (max(0.0, min(1.0, trust)), character_id, toward_id),
            )
            self._commit()

    def bump_interaction(self, character_id: str, toward_id: str) -> None:
        with self._lock:
            self.conn.execute(
                """UPDATE relationships SET interactions = interactions + 1
                   WHERE character_id = ? AND toward_id = ?""",
                (character_id, toward_id),
            )
            self._commit()

    def resolve_goal(self, character_id: str, goal_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO resolved_goals (character_id, goal_id) VALUES (?, ?)",
                (character_id, goal_id),
            )
            self._commit()

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
            self._commit()

    def get_beliefs_from_other_scenes(
        self, character_id: str, scene_id: str, limit: int = 5
    ) -> list[Belief]:
        """What this character carries in from before tonight, most
        salient first.

        Joined to the source event to find its scene, so a belief is
        "from before" by where it was formed rather than by when it was
        written. Every row was encoded from this character's own
        projection, which is what makes reading them back safe.
        """
        with self._lock:
            rows = self.conn.execute(
                """SELECT b.* FROM beliefs b
                   JOIN events e ON e.id = b.source_event_id
                   WHERE b.character_id = ? AND e.scene_id != ?
                   ORDER BY b.salience DESC, b.id DESC
                   LIMIT ?""",
                (character_id, scene_id, limit),
            ).fetchall()
            return [_row_to_belief(r) for r in rows]

    def get_interpretation(self, character_id: str, span_key: str) -> str | None:
        """The stored reading, or None if this span has never been read.

        An empty string is a real answer — a reading that failed its
        guard — so absence and refusal are deliberately distinguishable.
        """
        with self._lock:
            row = self.conn.execute(
                """SELECT text FROM interpretations
                   WHERE character_id = ? AND span_key = ?""",
                (character_id, span_key),
            ).fetchone()
            return row["text"] if row else None

    def put_interpretation(self, character_id: str, span_key: str, text: str) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO interpretations (character_id, span_key, text)
                   VALUES (?, ?, ?)""",
                (character_id, span_key, text),
            )
            self._commit()

    def get_discovered_rooms(self, world_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT room_id, name, description, reached_from
                   FROM discovered_rooms WHERE world_id = ? ORDER BY rowid""",
                (world_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def add_discovered_room(
        self, world_id: str, room_id: str, name: str, description: str, reached_from: str
    ) -> None:
        with self._lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO discovered_rooms
                   (world_id, room_id, name, description, reached_from)
                   VALUES (?, ?, ?, ?, ?)""",
                (world_id, room_id, name, description, reached_from),
            )
            self._commit()

    def get_story(self) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM story WHERE only_row = 1").fetchone()
            return dict(row) if row else None

    def start_story(self, story_id: str, title: str, world: str, scene: str) -> None:
        # Full precision: two stories started or played in the same
        # second are ordered by when they happened, not by filename.
        now = datetime.now().isoformat()
        with self._lock:
            self.conn.execute(
                """INSERT OR IGNORE INTO story
                   (only_row, story_id, title, world, scene, created_at, played_at, turns)
                   VALUES (1, ?, ?, ?, ?, ?, ?, 0)""",
                (story_id, title, world, scene, now, now),
            )
            self._commit()

    def touch_story(self, scene: str, turns: int) -> None:
        """Where the story is now and how far it has got.

        Written inside the turn like everything else, so a take that is
        thrown away does not leave the library claiming it happened.
        """
        with self._lock:
            self.conn.execute(
                """UPDATE story SET scene = ?, turns = ?, played_at = ?
                   WHERE only_row = 1""",
                (scene, turns, datetime.now().isoformat()),
            )
            self._commit()

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
            self._commit()


def _row_to_belief(row: sqlite3.Row) -> Belief:
    return Belief(
        id=row["id"],
        character_id=row["character_id"],
        subject_id=row["subject_id"],
        content=row["content"],
        interpretation=row["interpretation"],
        confidence=row["confidence"],
        source_event_id=row["source_event_id"],
        formed_at=datetime.fromisoformat(row["formed_at"]),
        last_rehearsed=datetime.fromisoformat(row["last_rehearsed"]),
        salience=row["salience"],
    )


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
