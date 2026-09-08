"""Making the independent model calls at the same time.

A turn is mostly waiting. Every character bids, every character reads
back the moment they just perceived, and each of those is a round trip
to somebody's model — measured at 33 calls for a two-agent scene and 51
for a four-agent one, over three player lines. They were made one at a
time, so a turn cost the *sum* of every round trip in it, and the wait
grew with the size of the cast: the fuller the room, the slower it got
to speak in.

Two of those loops are independent by construction. Nobody's bid can
see anybody else's — that is what makes them separate agents rather than
one model with a cast list — and neither can anybody's private reading
of a moment. Independent work made to queue is just waiting, so it goes
out together.

What that costs, and why it is safe:

* **Order is preserved.** Results come back in the order the work was
  given, not the order it finished, so arbitration sees the same bids in
  the same sequence and the store is written in cast order. A parallel
  turn produces the same story as a sequential one; only the waiting is
  different.
* **Nothing writes concurrently.** Only the model call runs in a thread.
  Anything that changes durable state is applied afterwards, in order,
  on the calling thread — so a turn is still one unit of work that can
  be thrown away whole.
* **The threads are waiting, not computing.** These are network round
  trips, so the GIL is released for the whole of each one and threads are
  the right tool. No process pool, no async rewrite of the engine.

`workers` is a knob for somebody else's endpoint, not a tuning
parameter: a hosted provider with a rate limit and a single-slot llama
server on a laptop want different numbers, and `--workers 1` restores
the old sequential behaviour exactly.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

T = TypeVar("T")

# Enough to cover a whole cast bidding at once — the batch that matters —
# without opening more sockets on somebody's endpoint than a turn can use.
DEFAULT_WORKERS = 8


def in_parallel(work: list[Callable[[], T]], workers: int = DEFAULT_WORKERS) -> list[T]:
    """Run independent calls at the same time; results in the order given.

    One item, or one worker, runs inline on this thread — no pool, no
    thread, and an identical stack to the sequential path, which is what
    `--workers 1` is for.
    """
    if workers <= 1 or len(work) <= 1:
        return [call() for call in work]
    with ThreadPoolExecutor(max_workers=min(workers, len(work))) as pool:
        # `map` yields in submission order and re-raises the first
        # failure, so a provider error surfaces as it would have done
        # sequentially rather than being swallowed by a worker.
        return list(pool.map(lambda call: call(), work))
