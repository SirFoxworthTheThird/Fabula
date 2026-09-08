"""Making the independent calls at the same time (CLAUDE.md).

A turn is mostly waiting: every character bids, every character reads
back the moment they just perceived, and each of those is a round trip.
They were made one at a time, so the fuller the room, the longer it took
to speak in it — which is the wrong way round for a roleplay app.

What these tests hold down is that going parallel bought only time: the
same story, the same store, the same order, with nothing durable written
off the turn's own thread.
"""
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from fabula.concurrency import in_parallel
from fabula.llm import FakeLLM
from fabula.session import Session

from tests.conftest import ASHGROVE

# Three people, one room, nowhere to hide: the case the wait was worst in.
CROWDED = "the_reckoning"
SCRIPT = ["Pass the bread.", "Where is the music box?", "Somebody broke it."]


class Recording(FakeLLM):
    """A model that takes time to answer, and remembers how many callers
    were waiting on it at once."""

    def __init__(self, delay: float = 0.01):
        super().__init__()
        self.delay = delay
        self._lock = threading.Lock()
        self._now = 0
        self.peak = 0
        self.threads: set[str] = set()

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        with self._lock:
            self._now += 1
            self.peak = max(self.peak, self._now)
            self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        try:
            return super().complete(system, prompt, key)
        finally:
            with self._lock:
                self._now -= 1


# --- The helper -------------------------------------------------------


def test_results_come_back_in_the_order_they_were_given():
    """Not the order they finished in. Arbitration reads bids in order
    and the store is written in cast order, so a turn that finished its
    calls in a different order every time would tell a different story
    every time."""
    delays = [0.05, 0.0, 0.03, 0.0]

    def work(index: int, delay: float):
        return lambda: (time.sleep(delay), index)[1]

    assert in_parallel([work(i, d) for i, d in enumerate(delays)], workers=4) == [0, 1, 2, 3]


def test_one_worker_runs_on_the_calling_thread():
    """`--workers 1` is the old engine exactly: no pool, no thread, and
    the same stack to debug."""
    here = threading.current_thread().name
    ran = in_parallel([lambda: threading.current_thread().name] * 3, workers=1)

    assert ran == [here, here, here]


def test_the_calls_really_do_overlap():
    """The barrier is the proof: it only opens once all four callers are
    inside it at the same time, so this cannot pass sequentially."""
    gate = threading.Barrier(4, timeout=5)

    results = in_parallel([lambda: gate.wait() is not None] * 4, workers=4)

    assert results == [True] * 4


def test_a_failure_surfaces_rather_than_being_swallowed():
    def boom():
        raise RuntimeError("the provider said no")

    with pytest.raises(RuntimeError, match="the provider said no"):
        in_parallel([lambda: 1, boom, lambda: 3], workers=4)


# --- A whole turn -----------------------------------------------------


def play(workers: int, llm) -> Session:
    session = Session.open(ASHGROVE, CROWDED, llm=llm, workers=workers)
    for line in SCRIPT:
        session.say(line)
    return session


def test_a_turn_has_several_calls_in_flight_at_once():
    parallel = Recording()
    play(8, parallel).close()

    sequential = Recording()
    play(1, sequential).close()

    assert parallel.peak > 1, "the whole point"
    assert sequential.peak == 1, "--workers 1 must still be one at a time"
    # Concurrency is not meant to change what is asked, only when.
    assert len(parallel.calls) == len(sequential.calls)


def test_the_same_story_either_way():
    """The story a player gets must not depend on how many sockets were
    open while it was told."""
    def told(workers: int) -> dict:
        session = play(workers, FakeLLM())
        events = session.store.get_events(session.scene.id)
        told = {
            "events": [(e.seq, e.kind, e.actor_id, e.location_id, e.content) for e in events],
            "beliefs": {
                cid: [(b.content, b.interpretation, round(b.salience, 6))
                      for b in session.store.get_beliefs(cid)]
                for cid in session.characters
            },
            "regard": {
                cid: {
                    toward: (r.trust, r.affinity)
                    for toward, r in session.store.get_relationships(cid).items()
                }
                for cid in session.characters
            },
        }
        session.close()
        return told

    assert told(8) == told(1)


def test_nothing_durable_is_written_off_the_turns_own_thread():
    """The rule that makes a parallel turn safe: only the model call runs
    in a worker. Everything that changes the store happens afterwards, on
    the calling thread, in order — which is what keeps a turn one unit of
    work that can still be thrown away whole.
    """
    llm = Recording()
    session = Session.open(ASHGROVE, CROWDED, llm=llm, workers=8)
    main = threading.current_thread().name
    offences: list[tuple[str, str]] = []

    def watch(sql: str) -> None:
        # SQLite reports each statement on the thread that runs it.
        if sql.lstrip()[:6].upper() in ("INSERT", "UPDATE", "DELETE"):
            if threading.current_thread().name != main:
                offences.append((threading.current_thread().name, " ".join(sql.split())[:60]))

    session.store.conn.set_trace_callback(watch)
    for line in SCRIPT:
        session.say(line)
    session.store.conn.set_trace_callback(None)
    session.close()

    assert llm.threads != {main}, "nothing ran in a worker, so this proves nothing"
    assert offences == []


def test_a_story_on_disk_survives_a_parallel_turn(tmp_path):
    """Threads plus one SQLite connection is the classic way to corrupt a
    file, so play a real one and read it back with a fresh connection."""
    path = tmp_path / "story.sqlite"
    session = Session.open(ASHGROVE, CROWDED, db_path=str(path), llm=FakeLLM(), workers=8)
    for line in SCRIPT:
        session.say(line)
    session.close()
    session.store.close()

    fresh = sqlite3.connect(path)
    integrity = fresh.execute("PRAGMA integrity_check").fetchone()[0]
    written = fresh.execute("SELECT count(*) FROM events").fetchone()[0]
    beliefs = fresh.execute("SELECT count(*) FROM beliefs").fetchone()[0]
    fresh.close()

    assert integrity == "ok"
    assert written >= len(SCRIPT), "every line the player typed is in the file"
    assert beliefs > 0, "and what the room made of them"
