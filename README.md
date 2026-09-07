# Fabula

A story engine where every character in a scene is a separate AI agent with its own
private knowledge, and a central orchestrator decides who speaks when. You are a
character in the story, not an observer of it.

In narrative theory, the *fabula* is the raw chronological set of events that make up a
story; the *syuzhet* is the arranged, partial telling of it. That distinction is the
architecture: the append-only event log is the fabula, and each character's projection
is their own syuzhet.

## The point

Existing character-chat apps ask a model nicely not to reveal things. Fabula makes the
revelation impossible.

If Tomás says something in the kitchen while Maria is in the study, Maria does not know
it — not because a prompt discouraged it, but because the engine cannot give it to her.
Her context is assembled by deterministic code that dropped the event before any model
saw it. Characters can hold false beliefs, keep secrets, lie, and be lied to, including
by you.

Perception is graded, not binary:

| | |
|---|---|
| **full** | same room — the actual content |
| **degraded** | next door — *"muffled voices from the kitchen, words unclear"* |
| **none** | the event never enters their projection at all |

Degradation is a template substitution, not a model call. A character's inference about
what they half-heard happens naturally when they speak.

## Status

Milestones M0–M8 of [`spec.md`](spec.md) are implemented, with 97 tests passing.

**One thing is unverified, and it is the important one.** Without a provider API key the
engine runs on `FakeLLM`, which emits `(a considered pause) [gen:8334793e]` in place of
every line. All the machinery is exercised and tested that way — turn-taking, perception
grading, pressures firing on schedule, time skips, persistence. But the spec's first
milestone is *"prove the fun exists"*, and whether the scene is actually compelling to
play has never been tested. Set a key and spend ten minutes in the kitchen before
trusting any of the rest.

## Quickstart

```bash
pip install -e ".[dev]"
```

Python 3.11+. Set a provider key to get real dialogue — without one you get placeholder
text and a warning on stderr:

```bash
export OPENAI_API_KEY=...     # ANTHROPIC_API_KEY, GEMINI_API_KEY etc. are also detected
```

### Play in a terminal

```bash
python -m fabula.cli worlds/ashgrove the_dinner
```

You are Elena, in the kitchen with your brother Tomás. Your sister Maria is in the
study. Tomás is sitting on something he does not want Maria to hear.

```
Elena> Tomás, you've been strange all evening.
/go study      move to another room
/wait          let time pass, resolving what happens off-screen
/look          take in the room — off-screen events here expand into detail
/quit
```

Pass `--db scene.sqlite` to keep a scene on disk. Characters are durable: run it again
against the same file and they arrive remembering the last one.

### Playtest a scene

Whether a scene *reads well* is a judgment a person has to make, but it shouldn't need
playing by hand after every change. This runs a fixed script and prints the transcript,
then dumps what each character came away believing:

```bash
fabula-playtest worlds/ashgrove the_dinner --model gpt-4o-mini
```

The belief dump is deliberately omniscient — it is how you check the asymmetry landed,
that Maria really doesn't know. It is an author's tool; no player-facing client may show
it. Pass `--script mine.txt` for your own sequence, `--no-beliefs` for the transcript
alone.

### Play in a browser

```bash
fabula-serve --worlds worlds        # then open http://127.0.0.1:8000
```

The web client renders *your character's projection*. Dialogue is attributed; things you
half-hear are dimmed behind an `unclear` tag and deliberately left unattributed, because
your character does not know who that was.

## Invariants

These define the product. If a change makes one harder to guarantee, the change is wrong.

1. **A character can never be given information their character could not perceive.**
   Filtering is deterministic code, never a model deciding what someone should know.
2. **World truth and character belief are separate.** A belief store may hold false
   things. Nothing ever reconciles it against reality.
3. **Everything that happens is an event in an append-only log** — including time skips
   and off-screen action. The `events` table has triggers that abort UPDATE and DELETE.
4. **The orchestrator is the only omniscient component, and its output is narrow.** The
   director emits a speaker id or a pressure id, never prose that reaches a character.
5. **Projections are reproducible.** Same log and character produce the same projection.
   Summaries are computed once and stored, never recomputed on the fly.

Invariant 1 is the one that erodes silently under refactoring, so it is covered by tests
from the first commit — including through summarization, across scene boundaries, and at
the HTTP wire.

## Architecture

| Component | Responsibility | Model calls |
|---|---|---|
| `world.py` | Rooms, adjacency, who perceives what | **None, ever** |
| `memory.py` | Projection, tiering, decay, retrieval | Summarization only |
| `director.py` | Speaker selection, pressures, time, turn budget | No — arbitration is argmax over bids |
| `narrator.py` | Describes perceivable action; renders pressures | Yes |
| `agents.py` | Bids to speak; generates utterances | Yes |

`world.py` has zero model calls by design — it is the thing that makes invariant 1
enforceable rather than aspirational.

Supporting modules: `session.py` (the surface both clients use), `chronology.py` (time
skips, off-screen intentions), `pressures.py`, `persistence.py`, `summaries.py`,
`db.py`, `loader.py`.

### The turn loop

1. Your input is appended as an event.
2. The world model resolves who perceived it, at what fidelity.
3. A cheap heuristic prefilters candidate speakers — only those who perceived it.
4. Each candidate bids. A bid reads *exactly* what a reply would read, so it cannot leak.
5. The narrator bids on different triggers: a lull, an arrival, a scene needing setting.
6. The director picks a speaker, fires a pressure, or yields to you.
7. Guards: a turn budget, and a hard cap on consecutive non-player turns.

Model calls are spent on bids only for ambiguous candidates; obvious ones resolve by
heuristic.

## Authoring

Everything authored is plain YAML on disk — diffable, shareable, legible to a coding
agent asked to change it.

```
worlds/ashgrove/
  world.yaml          rooms, adjacency, facts the story can turn on
  characters/*.yaml   persona, traits, goals, intentions, relationships
  pressures.yaml      authored complications
  scenes/*.yaml       cast, starting positions, mode, turn budget
```

Personality is mechanical, not only prose. `traits` feed the bid function and encoding
salience; `persona` feeds the voice. Both are required — if traits live only in the
prompt, every character bids alike and the scene converges into polite round-robin.

```yaml
traits:
  talkativeness: 0.45
  reactivity: {secret_mentioned: 0.9, insulted: 0.6}
  salience_bias: {utterance: 1.0, action: 0.9}   # what this character finds memorable
  reticence: 0.75
protects: [music_box]        # asked about this, he deflects instead of answering
intentions:
  - id: check_the_glue
    description: checks the seam where he glued it
    location_id: kitchen
    ready_after_minutes: 20
    private: true            # only when the room is empty
```

A character with high `reticence` who is pressed on something in `protects` bids to
**withhold** — the narrator renders them visibly not answering, which the room can see.
That matters because a reticent character who merely loses the bid reads exactly like
someone who isn't there. The narrator is never told *what* is being withheld, and is
never given the persona (that is where the secret is written down).

Pressures are a trigger plus an intent, never a script. The director chooses among
*authored* pressures and the narrator renders the chosen one; it can never invent a
complication, because a director allowed to improvise produces generic beats — a knock,
thunder, a stranger — with no stake in the world.

```yaml
- id: maria_comes_through
  intent: >
    Maria runs out of patience with the letters and comes through to the kitchen,
    half-expecting to interrupt something.
  trigger: {turns_elapsed: "> 4", character_at: {maria: study}}
  effect: {kind: arrival, actor: maria, location: kitchen, audibility: adjacent}
  cooldown_turns: 20
  max_fires: 1
```

Scene `mode` changes how pressures are *selected*, not what they are: `arc` escalates as
the scene runs on, `sandbox` stays out of the way until the characters go quiet and then
reseeds tension.

## HTTP API

The protocol is a POV, not a transcript. Every event on the wire carries the requesting
character's perceived content plus structured `speaker`, `location` and `visibility`
fields. An event they could not perceive is not sent-and-flagged for a client to hide —
it never enters the stream.

| | |
|---|---|
| `GET /` | the web client |
| `GET /worlds` | worlds and their scenes |
| `POST /sessions` | open a scene → session id, your character, location |
| `GET /sessions/{id}` | location, who is present, story time, pending skip |
| `GET /sessions/{id}/events` | the scene as your character experienced it |
| `POST /sessions/{id}/say\|move\|wait\|look` | act |
| `GET /sessions/{id}/stream` | SSE — catch-up, then live (`?follow=false` to poll) |

Large time skips require consent in the request; `pending_skip_minutes` in scene state
tells a client how long the next one would be, so it can ask first.

### SillyTavern

An OpenAI-compatible shim lets any `/v1/chat/completions` client drive a scene.

* **API Base URL** — `http://127.0.0.1:8000/v1`
* **API Key** — a scene token, e.g. `ashgrove/the_dinner`

`GET /v1/models` lists every scene, so the model picker becomes the scene picker. Clients
that cannot set a key can POST to `/v1/{world}/{scene}/chat/completions` instead. The
same token keeps returning the same session, which is how a stateless client holds a
continuing story.

It is deliberately lossy: a whole turn of several characters at graded perception
collapses into one block of text. And **incoming history is ignored except the last user
message** — the client's transcript, system prompt and character card are all discarded,
because a client that could prepend context could hand a character knowledge the world
model never gave them.

## Tests

```bash
python -m pytest
```

The suite is the regression net *and* the clearest description of the product:

* **Leak tests** — a secret spoken in the kitchen never reaches the character in the
  study: not in her projection, her generated reply, her stored summaries after 60 turns,
  her beliefs across a scene boundary, the CLI's output, or the HTTP stream. No bid
  rationale from one character ever reaches another's context.
* **Determinism** — same log and character produce byte-identical projections; summaries
  are computed once and never drift.
* **Loop safety** — agent-to-agent exchange always terminates; you always get the turn back.
* **Divergence** — two characters with different `salience_bias` form measurably
  different beliefs about the same event, and a degraded perceiver forms a belief
  consistent with what they half-heard rather than with the truth.

## Known rough edges

* `LiteLLMClient` defaults to `gpt-4o-mini`. Any model id litellm understands works, and
  `fabula-playtest --model` selects one, but the interactive CLI and the service still
  have no flag — construct the client directly to change it there.
* Prompt adherence is the soft spot. Structural rules hold regardless of model (the
  narrator cannot narrate the player, because it does not bid), but the ones that live in
  prompts — invent no props, never speak for a character, don't raise what you guard —
  are only as good as the model reading them. Measured on a 1.5B local model, the
  "what you guard" reminder made no difference at all: 4/16 openings blurted the secret
  with it, 3/16 without. Treat every prompt-level rule here as unproven until measured
  on the model you actually ship.
* Relationship affinity does not move during play. Interaction counts accumulate and
  authored affinity/trust persist, but nothing shifts affinity from the *content* of what
  is said; that needs judgment that could not be made deterministically without risking
  invariant 2.
* Sessions live in memory, so restarting the service drops them. Scene state survives if
  you point sessions at a database file.

## License

None yet.
