# Fabula — a multi-agent story engine

**Audience:** an implementing agent (Claude Code). Read this whole document before writing code.

**On the name.** In narrative theory, the *fabula* is the raw chronological set of events that make up a story; the *syuzhet* is the arranged, partial telling of it. That distinction is this system's architecture: the append-only event log is the fabula, and each character's projection is their own syuzhet. Use this vocabulary in the code — `fabula` for the canonical log, `syuzhet` for a per-character projection — it is precise and it keeps the central invariant legible at a glance.

---

## 1. What we are building

A story engine where every character in a scene is a separate AI agent with its own private knowledge, and a central orchestrator decides who speaks when. The user is also a character in the story, not an outside observer.

The point of difference from existing character-chat apps is **genuine information asymmetry**. If two characters are in the kitchen and one is in the study, the one in the study does not know what was said in the kitchen — not because a prompt asked nicely, but because the engine cannot give them that information. Characters can hold false beliefs, keep secrets, lie, and be lied to, including by the user.

This is a headless service. Clients (a CLI first, a GUI later) talk to it over an API. The engine never assumes a particular frontend.

---

## 2. Non-negotiable invariants

These are the properties that define the product. If an implementation choice makes any of these harder to guarantee, the choice is wrong.

1. **A character can never be given information their character could not perceive.** Filtering is done by deterministic code, never by asking a model to decide what someone should know.
2. **World truth and character belief are separate.** A character's belief store is allowed to contain things that are false. No component ever "corrects" a character's beliefs against world state.
3. **Everything that happens is an event in an append-only log.** Nothing is implicit — not time skips, not off-screen actions, not narration.
4. **The orchestrator is the only omniscient component, and its output is narrow.** It emits control decisions (who speaks, what pressure fires, how much time passes). It never emits prose that reaches a character's context.
5. **Projections are reproducible.** Given the same log and the same character, the projection is the same. Derived artifacts like summaries are computed once and stored, never recomputed on the fly.

Invariant 1 is the one that will silently erode under refactoring. It must be covered by tests from the very first milestone.

---

## 3. Stack

- **Python 3.12+**
- **FastAPI** — HTTP + SSE streaming
- **SQLite** — single file, append-only `events` table plus derived tables
- **Pydantic** — all events, bids, and config are validated models
- **litellm** — provider-agnostic model calls
- **pytest** — the invariant suite

No ORM beyond what SQLite needs directly. No message broker. No background worker in early milestones.

---

## 4. Architecture

Five components, separately testable.

| Component | Responsibility | Uses a model? |
|---|---|---|
| **World model** | Rooms, presence, adjacency, who can perceive what | No — pure code |
| **Memory projector** | Builds a character's context from the log: filter, decay, retrieve | Only for summarization |
| **Director** | Speaker selection, pressure selection, time advancement, scene pacing | Yes |
| **Narrator** | Describes perceivable action; renders pressures as prose | Yes |
| **Character agents** | Bid to speak; generate utterances and actions | Yes |

The world model must have zero model calls. It is the thing that makes invariant 1 enforceable.

---

## 5. Data model

### 5.1 Events

The core table. Append-only — rows are never updated or deleted.

```python
class Event(BaseModel):
    id: int
    scene_id: str
    seq: int                      # monotonic within scene
    story_time: datetime          # in-world time, not wall clock
    kind: Literal["utterance", "action", "narration", "arrival",
                  "departure", "time_skip", "state_change"]
    actor_id: str | None          # character who caused it; None for world events
    location_id: str
    content: str
    audibility: Literal["private", "room", "adjacent", "building"]
    addressed_to: list[str] = []  # character ids, empty = whole room
    salience_base: float = 0.5    # author/narrator hint, 0..1
    detail_level: Literal["full", "summary"] = "full"
    metadata: dict = {}
```

`detail_level: "summary"` is how coarsely-resolved off-screen time is stored (see §8).

### 5.2 Perception

Perception is graded, not binary. The world model resolves, for a given event and a given character, one of:

- `full` — same room, event is audible to them
- `degraded` — adjacent space; content is replaced with a degraded descriptor ("raised voices next door, words unclear")
- `none` — not perceived; the event does not enter their projection at all

Degradation is a **transformation of the event content**, produced deterministically from `kind` + `audibility` + distance. Do not ask a model to degrade content at perception time; use templates. The character's own inference about what they heard happens naturally in their generation.

### 5.3 Characters

```python
class Character(BaseModel):
    id: str
    name: str
    persona: str                  # prose, voice and background
    traits: Traits                # mechanical, see below
    goals: list[Goal]
    location_id: str
    is_user: bool = False

class Traits(BaseModel):
    talkativeness: float          # 0..1, base bid threshold
    reactivity: dict[str, float]  # trigger -> bid modifier
                                  # e.g. {"insulted": 0.9, "secret_mentioned": 0.8,
                                  #       "unexplained_noise": 0.6}
    salience_bias: dict[str, float]  # topic/kind -> salience multiplier at encode time
    reticence: float              # resistance to volunteering information
```

**Personality must be mechanical, not only prose.** If traits live purely in the prompt, every character bids alike and the voices converge into polite round-robin. `traits` feed the bid function and the encoding salience; `persona` feeds the prose voice. Both are required.

### 5.4 Beliefs

Per-character, durable across scenes.

```python
class Belief(BaseModel):
    character_id: str
    subject_id: str               # another character, an object, a fact key
    content: str
    confidence: float
    source_event_id: int | None
    formed_at: datetime
    last_rehearsed: datetime
    salience: float
```

Beliefs are derived from perceived events but stored independently, because they persist after the source events have decayed to gist. They may contradict world state. That is correct.

### 5.5 Pressures

Authored per world. Trigger condition + intent, never a script.

```yaml
- id: secret_surfaces
  intent: "Someone who knows part of the secret arrives and probes"
  trigger:
    fact_unspoken: buried_body
    for_turns: "> 12"
  effect:
    kind: arrival
    actor: neighbour_ana
    location: kitchen
  cooldown_turns: 20
  max_fires: 1
```

The director selects among *eligible* pressures based on scene state. The narrator renders the chosen one into situational prose. **Improvisation lives in the wording and timing, never in the invention.** Do not let the director invent complications from nothing — it produces generic beats (a knock, thunder, a stranger) with no stake in the world.

---

## 6. Memory and context assembly

**The agent's memory store is not its context.** The store is durable: events, cached summaries, beliefs, relationships. The context is assembled fresh each turn from the store, sized to a token budget.

Tiered by a combination of recency, salience, and rehearsal:

- **Verbatim** — recent events, full content
- **Summarized** — mid-range, compacted into stored per-character summaries
- **Gist** — old, one-line traces

Rules:

1. Salience is scored **at encode time**, from `salience_base` × the perceiving character's `salience_bias`. The jealous character rates an ambiguous glance higher and keeps it verbatim long after a placid character has let it blur.
2. **High-salience events are exempt from demotion.** A betrayal from twenty scenes ago stays sharp.
3. **Rehearsal refreshes recency.** An event re-mentioned in conversation has its `last_rehearsed` updated and climbs back up the tiers.
4. **A retrieval pass** can pull an old detail back into context when the current scene cues it (semantic match against the character's own event history only — never the global log).
5. **Summaries are computed once and stored.** Re-summarizing on every turn causes beliefs to drift on their own and destroys reproducibility, which is what makes the leak tests meaningful.

---

## 7. The turn loop

For each user input:

```
1. Append user's utterance/action as an event.
2. World model resolves who perceived it, at what fidelity.
3. Cheap heuristic prefilter → candidate speakers:
     - addressed by name
     - present in the scene
     - own last turn ended unresolved
     - trait-triggered (reactivity keys matched)
4. For each candidate: assemble their projection, request a BID.
     Bid = {desire: float, one_line_reason: str}
     Same filtered context as a real reply — a bid can never leak,
     because it reads exactly what the reply would read.
5. Narrator also bids, on different triggers: a lull, an undescribed
   physical action, a scene needing establishing.
6. Director arbitrates: picks a speaker, or fires a pressure, or
   yields to the user, or advances time.
7. Selected agent generates. Result is appended as an event.
8. Loop from 2, until turn budget exhausted or yield-to-user.
```

**Guards:**
- **Turn budget** per user input, or agent-to-agent ping-pong never terminates.
- **Explicit yield-to-user condition**, or the user is starved out of their own scene.
- **Bid rationales never enter another character's context.** The director sees all bids and is the omniscience risk. Its output must be a speaker id, not prose.
- Only poll characters who perceived the triggering event. Someone in another room is never asked, so there is nothing to leak.

Spend model calls on bids only for ambiguous candidates; resolve the obvious ones by heuristic.

---

## 8. Time

Story time advances in narrative jumps, not wall clock. There is no background job ticking the world.

- The **director derives** the jump: it looks at pending character intentions and skips to the next moment something worth discovering is ready. Never an arbitrary "three hours pass" that lands on nothing.
- Every skip is **committed to the log as a `time_skip` event with an explicit duration**. Projections replay over it and decay reads off it. An unlogged jump is the same class of bug as an unlogged utterance.
- **Off-screen action is resolved coarsely.** Characters have intentions and schedules; elapsed time produces a few `detail_level: "summary"` events per character per interval. Detail is **materialized lazily** — when the user finds the kitchen ransacked, *that* is when "Tomás searched the house" expands into specifics. The log remains the single source of truth; only resolution varies.
- **Large skips require user consent.** Advancing to morning when the user's character sleeps is fine. Silently swallowing three days is not — the user is a character with agency, and time is something they should never lose without noticing.
- **Elapsed time is perceived non-uniformly.** Someone asleep experiences the skip as a discontinuity; someone awake and waiting experienced every hour. This is a projection concern like any other.

---

## 9. Scene modes

Same machinery, different director objective. Declared per scene.

- **Arc** — bounded. Escalates pressures toward a declared end condition.
- **Sandbox** — unbounded. Maintains equilibrium, reseeds tension as it resolves.

This changes how pressures are *selected*, not what they are.

---

## 10. Persistence across scenes

Characters are durable entities. Between scenes they carry:

- their belief store, aged and decayed
- unresolved goals
- relationship state toward every other character, **including the user's character**

Scenes reference characters by id. A character who remembers last week's betrayal is the point; resetting them each session is not acceptable.

---

## 11. Authoring

Everything authored lives in **plain YAML files on disk** — worlds, characters, rooms, pressures, scenes. The GUI (later milestone) is an editor *over those files*, not a database that hides them. This keeps scenes diffable, shareable, and legible to a coding agent asked to change them.

```
worlds/
  ashgrove/
    world.yaml          # rooms, adjacency, audibility rules
    characters/
      tomas.yaml
      maria.yaml
    pressures.yaml
    scenes/
      the_dinner.yaml   # cast, starting positions, mode, end condition
```

---

## 12. Milestones

**M0 — Prove the fun exists. No UI, no API, no persistence layer beyond SQLite.**
CLI that loads a scene from YAML and prints turns to stdout. Three characters, two rooms, one secret held by exactly one character. The user plays a character via stdin.
*If this scene is not compelling in a terminal, nothing built later will save it.*

**M1 — Invariant test suite.** (Write alongside M0, not after.)
See §13.

**M2 — Memory tiers and decay.** Long scenes stop blowing the context budget. Summaries stored, salience scoring live, retrieval pass working.

**M3 — Pressures and the director's arc/sandbox objectives.** Drama stops being accidental.

**M4 — Time skips and off-screen resolution.**

**M5 — Cross-scene persistence.** Belief aging, relationship state.

**M6 — FastAPI service + SSE streaming.** Native protocol: a stream of scene events with structured speaker, location, and visibility fields.

**M7 — GUI client.** Renders *the user's character's projection*, not the world log. The UI is a POV, not a transcript.

**M8 — OpenAI-compatible shim** (`/v1/chat/completions`) so SillyTavern can be used as a lossy client. Session identity via the API-key field as a scene token, or a per-scene URL path. Incoming history is ignored except the last user message.

Do not start M6 before M0 reads well.

---

## 13. Required tests

The invariant suite is the regression net *and* the clearest description of the product. Write these before the code they cover.

**Leak tests (highest priority):**
- Build a scene where Tomás states a secret in the kitchen while Maria is in the study. Assert Maria's assembled projection contains zero tokens of the secret. Assert her generated reply does not reference it.
- Same, after 50 turns and a summarization pass. Summaries must not leak what projection excluded.
- Same, across a scene boundary with persistence.
- Assert no bid rationale from character A ever appears in character B's context.
- Assert narration never describes anything the POV character cannot perceive.

**Determinism:**
- Same log + same character + same seed → byte-identical projection.
- Summaries are stable: running the loop twice does not produce different stored summaries for the same events.

**Loop safety:**
- Agent-to-agent exchange always terminates within the turn budget.
- The user always regains the turn within N exchanges.

**Divergence (the feature working):**
- Two characters with different `salience_bias` perceiving the same event produce measurably different belief records.
- A degraded-perception character forms a belief consistent with the degraded descriptor, not the true content.

---

## 14. Explicitly out of scope for now

- Any background/real-time world simulation
- Multi-user shared scenes
- Voice, images, or any non-text modality
- SillyTavern support before M8

---

## 15. Guidance for the implementing agent

- The spec owns the design; you own the implementation. Where this document is silent on a detail, choose the simplest thing that preserves §2.
- If a requested change would weaken an invariant in §2, say so before implementing it.
- Prefer boring, inspectable code over cleverness. The world model in particular should be readable by someone checking it for leaks.
- Keep the engine free of frontend assumptions from day one, including during M0.
