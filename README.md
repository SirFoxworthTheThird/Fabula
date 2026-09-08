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

Milestones M0–M8 of [`spec.md`](spec.md) are implemented, with 233 tests passing.

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

**On Windows / PowerShell**, the environment variable is set differently, and the
`fabula-*` console scripts only exist if `pip install` ran *after* they were added and
your Python `Scripts\` directory is on `PATH`. Running them as modules sidesteps both
problems and always works:

```powershell
$env:OPENAI_API_KEY = "..."
python -m fabula.playtest worlds/ashgrove the_dinner
python -m fabula.cli worlds/ashgrove the_dinner
python -m fabula.api --port 8000          # the service
python -m fabula.measure worlds/ashgrove the_dinner
```

If those report `No module named fabula`, the install itself did not take — check with
`python -c "import fabula; print(fabula.__file__)"` and re-run `pip install -e .` from
the repository root, using the same interpreter you are invoking.

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
/reveal        when you are done: what you could not perceive (a spoiler)
/quit
```

Pass `--db scene.sqlite` to keep a scene on disk. Characters are durable: run it again
against the same file and they arrive remembering the last one.

`the_reckoning` is the same three people with nowhere to hide — one room, and an ending
the director escalates toward. Swap the scene name to play it.

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

### The reveal

Information asymmetry is invisible while you play. A character with nothing to say looks
exactly like a character saying nothing, and a scene where the secret held reads — from
the inside — like a scene with no secret in it. So the payoff comes afterwards: `/reveal`
in the terminal, or **What I missed** in the browser, shows what happened out of your
sight, what you half-heard set against what it actually was, who ended the scene knowing
the secret, and whose regard for whom moved while you were not looking. In a scene with an `end_condition` you are offered it at the moment
the scene reaches its end, rather than having to remember it exists.

It is the one part of the engine that deliberately steps outside a point of view, which
is why it lives in its own module rather than on `Session` (where everything is POV with
no exceptions), is only ever produced on request, and is read-only — looking behind the
curtain appends nothing, so you can look and keep playing.

### Memory

A belief used to be an echo — the perceived line, stored verbatim. Nobody remembers a
conversation as a transcript; they remember what they took it to mean, and two people in
the same room take it to mean different things. So a belief now carries both halves:

```
[0.50   sure] Tomás, you have been quiet all evening.
                → Tomás thinks his sister has noticed and is working up to asking.
```

`content` is the verbatim projection — the half the leak tests assert against, and the
half that survives with no model at all. `interpretation` is the reading, written from
that character's own perceived lines and nothing else. Three things keep the softest
component in the engine from being its weakest point:

* It is built from `perceived_content`, exactly like a summary, so it cannot reintroduce
  what the projection excluded or blurred.
* It is **checked before it is stored**. A reading that names a world fact the character
  has no business knowing is thrown away and the belief keeps its echo — keyword
  matching against authored facts, never a model judging a model. A memory is durable and
  crosses scenes, so a hallucination here would not be a bad line; it would be a false
  memory a character carries for good.
* It is **additive, never substituted**. In context the echo comes first and the reading
  is appended to it. Letting a reading stand in place of what was perceived would let a
  weak model quietly delete a memory instead of colouring it — which is exactly what
  happened the first time this was built the other way.

What a character carries in from earlier scenes is now read back into their context,
which is what makes durability visible in play at all. Everything in that block was
encoded from their own projection, so a memory Maria never formed cannot appear, and one
she formed from half-hearing something says what she half-heard.

Readings are also the bulk of a scene's model calls — one per remembered moment per
character. The player's are skipped (nothing reads their memory back, and writing down
what they privately think is the engine deciding their inner life), and `--no-interpret`
turns off the rest. On the sample scene that is 23 calls → 15 → 5.

### Point it at your own model

Every entry point takes `--model` (any id litellm understands) and `--api-base` (any
OpenAI-compatible endpoint — a hosted proxy, an aggregator, a local server). Credentials
come from the environment, never a flag, so they stay out of shell history.

```bash
export OPENAI_API_KEY=...
fabula-playtest worlds/ashgrove the_dinner --model gpt-4.1-nano

# against an OpenAI-compatible proxy: prefix the id so litellm speaks that dialect
export OPENAI_API_KEY=...                       # whatever key the proxy expects
fabula-playtest worlds/ashgrove the_dinner \
    --model openai/<their-model-id> --api-base https://<host>/v1
```

The same two flags work on `fabula`, `fabula-serve` and `fabula-measure`.

Rather than exporting it every session, put it in a `.env` beside the repo — copy
`.env.example`. Every entry point reads it at startup, and `.gitignore` already covers it
(`.env.example` is the only one that belongs in the repo).

```bash
cp .env.example .env && chmod 600 .env     # then fill it in
```

**An exported variable always wins over the file.** A dotfile that silently shadowed a key
you set in your shell would make it impossible to say which credential a run actually
used. Nothing prints, logs, or returns a value from it — names only — and on POSIX you get
one line of warning if the file is readable by other users on the machine.

There is deliberately no `--api-key` flag anywhere: a flag puts the key in shell history
and in every `ps` on the machine. There is a test that walks the syntax tree of every
module to keep it that way.

<details>
<summary>Worked example: nano-gpt.com</summary>

```bash
export OPENAI_API_KEY=<your nano-gpt key>     # yes, that variable — see below
fabula-playtest worlds/ashgrove the_dinner \
    --model openai/z-ai/glm-5.3-flash \
    --api-base https://nano-gpt.com/api/v1
```

Three things that are easy to get wrong:

* **The `openai/` prefix is required** and is *not* part of the model name. It tells
  litellm which dialect to speak; everything after it is sent to the provider verbatim.
  Their ids contain slashes of their own, and those survive:
  `openai/z-ai/glm-5.3-flash` arrives as `z-ai/glm-5.3-flash`.
* **The key lives in `OPENAI_API_KEY`** even though it is not an OpenAI key. That is the
  variable litellm's OpenAI-compatible path reads. It is sent as a bearer token, which is
  what they expect.
* `https://nano-gpt.com/api/v1` is the base — `/chat/completions` is appended for you.

`curl https://nano-gpt.com/api/v1/models` lists what is available without needing a key.

</details>

### Measure a prompt rule before believing it

The mechanical guarantees hold whatever model is behind them. The rules that live in
prompts are worth exactly what the model reading them makes of them, which differs by
model. `fabula-measure` runs one scene opening N times under three variants and counts
how often the character raises what they guard:

```bash
fabula-measure worlds/ashgrove the_dinner --model gpt-4.1-nano --samples 16
```

On a 1.5B local model the three variants landed at 3/16, 4/16 and 3/16 — the shipped
guard line does nothing there. Whether it earns its place on a capable model is exactly
what this command is for.

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

## Two worlds

Everything here was built against one world, and code fitted to one shape looks general
until a second one arrives. So there are two, deliberately unalike:

| | **ashgrove** | **winterlight** |
|---|---|---|
| | a house after a funeral | a station on the plateau, nine months in |
| rooms | 2, mutually audible | 4 in a chain, one of them one-way |
| cast | 2 agents + you | 3 agents + you |
| secrets | one, one holder | **two, two holders, from each other** |
| ends | when the music box is said | when *both* things are finally in the room |

`winterlight` is the test. Ilse has known for eleven days that the first flight out has
slipped by two months and has not said so — not from shame but as policy, which is a
different reason to reach for the same `protects` mechanic. Yusuf bled six hundred litres
out of the reserve tank in June and has been quietly making the whiteboard add up ever
since. Neither knows about the other. Nadia is the one who asks, and you are the doctor
they both need on side.

The engine ran it unchanged — the service, the shim and the browser client picked the
world up with no code at all — but building it found four things one world had hidden:

* **A trigger could only name one fact.** `end_condition: {fact_spoken: the_flight}` cannot
  say "over when both are out", which is the entire shape of a two-secret story. Triggers
  now take a list: `fact_spoken` means all of them, `fact_unspoken` means none of them,
  and a bare string still means one, so nothing already authored changes meaning.
* **Edges are directed, and nothing had ever used it.** The mess lists the generator shed,
  the shed lists nothing back — so from the mess you hear Yusuf working, and inside the
  shed the engine is all there is to hear. He can be overheard and cannot overhear. The
  perception model already supported this; ashgrove's two rooms could not ask for it.
* **A keyword has to be a phrase that cannot mean anything else.** Matching is substring
  and deterministic — a model is never asked whether a subject came up — so the cost lands
  on the author. `the reserve` was in the fuel fact's keywords until it matched *"the
  reserve of patience in this room is thin"*.
* **A room description reaches the log as narration.** So a fact keyword in one lets the
  narrator raise the subject just by describing the room, firing pressures and ending
  arcs nobody spoke about. The shed's description named the reserve drums until a test
  caught it.

The last two are now guarded for **every** world in the repo, present and future — along
with the one that would have caught an early ashgrove bug, where Maria's persona named
the music box in the same breath as saying she did not know about it. Authored prose is
the one place a leak can be written by hand, because it never passes through a projection.

### Then playing it found the real one

Running `winterlight` on a local 1.5B model, the arc **ended itself**. Nobody had
confessed anything. A pressure whose intent read *"counted down the days to the first
flight"* produced narration carrying that phrase, and the scene's `fact_spoken` condition
matched it. The guard above covered room descriptions and not pressure intents; it now
covers both, and intention descriptions too.

Chasing that turned up a live bug in **ashgrove**, which had been there since the arc
scene shipped:

```python
# Tomás's authored off-screen intention:
"takes the music box down and checks the seam where he glued it"
```

He performs that alone, off-screen, by design. It names the fact in its own action text —
so checking the glue in an empty kitchen satisfied `fact_spoken: music_box` and ended
`the_reckoning` with nobody in the room and nothing said.

A word in the log is not a subject in the room. `fact_spoken` now means **somebody other
than the actor perceived that event in full**. Half-hearing it through a wall does not
count either: the degraded descriptor carries no words, so a listener in the next room
did not catch what the subject was. One world could not have surfaced this — it needs
somewhere to be alone, and ashgrove's two rooms are always within earshot.

## Authoring

Everything authored is plain YAML on disk — diffable, shareable, legible to a coding
agent asked to change it.

```
worlds/<name>/
  world.yaml          rooms, adjacency, facts the story can turn on
  characters/*.yaml   persona, traits, goals, intentions, relationships
  pressures.yaml      authored complications
  scenes/*.yaml       cast, starting positions, mode, turn budget
```

Drop a directory in `worlds/` and every client finds it: the terminal, the playtest
harness, `/worlds`, the browser picker and the shim's model list all enumerate the
directory rather than a registry. `winterlight` needed no code.

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

And it costs them. Watching somebody refuse to answer is the one thing in the engine that
deterministically moves how they are regarded: every witness who saw it **at full
fidelity** loses a fraction of their trust in the withholder, and low trust raises their
bid the next time that person speaks — distrust is attention. Two things make that safe.
It is computed from each witness's own projection, so a character in the next room loses
nothing and neither does one who only half-heard it (the degraded descriptor carries no
name, so they cannot know who that was — a number that moved on an unperceived event
would be invariant 1 leaking through arithmetic instead of prose). And it never moves the
*player's* trust: deciding that Elena believes her brother less tonight is telling the
person holding her how they feel, which is the same overreach as narrating her actions
for her.

Trust that nothing reads is trust that does not exist, so the reveal shows what moved —
and it is a good example of what the reveal is for. Elena, standing in the study, has no
way to know her sister stopped believing her brother tonight.

```
What tonight changed:
  Maria trusts Tomás less than at the start (0.50 → 0.42)
```

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

An arc can also say what it is escalating *toward*. `end_condition` is written in the
same vocabulary as a pressure trigger — deliberately, because an author saying "this is
over once the music box is finally said out loud" should not need a second condition
language to say it:

```yaml
mode: arc
end_condition: {fact_spoken: music_box}
```

The ending is **advisory**. The engine reports that the thing the author was building
toward has happened; it does not lock the scene, refuse input, or stop the cast. The
terminal prints a line and the browser marks **What I missed**, and if you want to sit
in the kitchen afterwards and ask how long he has known, you can. `the_dinner` has no
`end_condition` at all — a sandbox is an evening, not a story with a shape — while
`the_reckoning` is the same house and the same three people with all of them at the
table and nowhere to slip off to.

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
* **Two-way asymmetry** — in `winterlight`, neither secret-holder ever learns the other's,
  through projection or a played turn loop.
* **Authoring guards**, parametrised over every world on disk: no persona names a fact
  that is not that character's own, no room description names a fact at all, every scene
  has exactly one player, and every authored room, fact, actor and cast id resolves — a
  typo otherwise fails silently as a pressure that never fires.

## Known rough edges

* `Room.adjacent` values are not read. The field maps a neighbour to how well sound
  crosses that doorway, which would let an author seal one; perception uses only the
  presence of an edge and the *event's* own audibility. Until it is wired up, a one-way
  edge is how you make a room you cannot hear out of.
* **A character the narrator invents is a ghost.** It is told to introduce no people who
  have not appeared, and a small model ignores that — the `winterlight` run produced an
  interrogator and an elderly man who are not on the station. They stay in prose and
  cannot become anything more, because an event needs an `actor_id` the director assigns
  from the cast, so an invented person cannot speak, act, be perceived as an actor, or
  hold a belief. It is a quality bug rather than a leak, and nothing promotes them.
  Bringing somebody genuinely new into a scene wants an authored latent cast and a
  pressure that fetches them, which does not exist yet.
* Prompt adherence is still the soft spot, and playing `winterlight` on a 1.5B model made
  that vivid: Yusuf recited his own persona aloud three times ("I fix the transfer valve.
  It's been a quarter turn open"), and the narrator invented an interrogator and an
  elderly man who are not on the station. Nothing structural gave way — see below — but
  every prompt-level rule here is worth exactly what the model makes of it.
* Prompt adherence is the soft spot. Structural rules hold regardless of model (the
  narrator cannot narrate the player, because it does not bid), but the ones that live in
  prompts — invent no props, never speak for a character, don't raise what you guard —
  are only as good as the model reading them. Measured on a 1.5B local model, the
  "what you guard" reminder made no difference at all: 4/16 openings blurted the secret
  with it, 3/16 without. Treat every prompt-level rule here as unproven until measured
  on the model you actually ship.
* Relationship **affinity** still does not move during play; only trust does, and only on
  a witnessed refusal to answer. Moving either from the *content* of what is said needs a
  model deciding whether someone was being sincere, and a wrong call there quietly
  rewrites a character's inner life — which is the correction invariant 2 forbids.
* Sessions live in memory, so restarting the service drops them. Scene state survives if
  you point sessions at a database file.

## License

None yet.
