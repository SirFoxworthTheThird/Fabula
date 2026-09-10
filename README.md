# Fabula

A story engine where every character in a scene is a separate AI agent with its own
private knowledge, and a central orchestrator decides who speaks when. You are a
character in the story, not an observer of it.

In narrative theory, the *fabula* is the raw chronological set of events that make up a
story; the *syuzhet* is the arranged, partial telling of it. That distinction is the
architecture: the append-only event log is the fabula, and each character's projection
is their own syuzhet.

## The point

You play a character going through a story — not a conversation, and not a puzzle about
who knows what. A story here is a sequence of scenes, and **what happened in one chooses
what follows it**:

```yaml
end_condition: {fact_spoken: music_box}
next:
  - {scene: the_morning_after,  when: {character_at: {maria: kitchen}}}
  - {scene: nobody_said_a_word}
```

The same confession leads to two different mornings, and which one you get depends on who
was standing in the room. Everyone crosses that seam carrying what the last scene did to
them: what they came to believe, whose word they stopped taking, the thing they were
trying to do that is no longer worth doing.

## What holds it up

Existing character-chat apps ask a model nicely not to reveal things. Fabula makes the
revelation impossible. That is the foundation rather than the product — it is what makes
the fiction trustworthy, not what anybody plays for.

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

Milestones M0–M8 of [`spec.md`](spec.md) are implemented, with 458 tests passing.

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

Name a model with `--model` and the key for it is checked *before* the story opens, so a
key that was never set is one sentence rather than a provider traceback several turns in.
A local server usually ignores the key's value but its client still insists on one — set
it to any string at all.

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

### Play

```bash
fabula
```

That starts the app and opens it in a browser: your stories on top, worlds underneath,
and a scene to click. It is the whole app — the shelf, play, the reveal — and it needs no
build step, because the client is one HTML file the service hands you.

If the port is busy (another copy is already open), it takes a free one and says which.
`--no-browser` starts it without opening anything, `--port` picks the port.

### Play in a terminal

```bash
fabula --terminal                      # your stories, in words
python -m fabula.cli worlds/ashgrove the_dinner
```

You are Elena, in the kitchen with your brother Tomás. Your sister Maria is in the
study. Tomás is sitting on something he does not want Maria to hear.

A scene opens by telling you what you have walked into — an authored paragraph, the same
every time, addressed to you and perceived by nobody else in the room, which is what lets
it be written in the second person and mention what only your character would know coming
in. Then the room is described, and whoever is standing there gets to speak first if they
want to: a story that opens on a bare prompt puts the whole burden of
starting it on you — you arrive somewhere, nobody says anything, and the only way to find
out you are not alone is to talk to the air. One beat, and no pressures: a greeting is the
room noticing you, while a pressure is the director escalating, and a story whose first
move is its own complication has started without you. And when nobody *is* there to
answer, the room answers — a player alone used to speak into "No one answers." and nothing
else, which is not a story.

```
Elena> Tomás, you've been strange all evening.
/go study      move to another room
/wait          let time pass, resolving what happens off-screen
/look          take in the room — off-screen events here expand into detail
/again         throw the last moment away and play it again
/reveal        when you are done: what you could not perceive (a spoiler)
/quit
```

`the_reckoning` is the same three people with nowhere to hide — one room, and an ending
the director escalates toward. Swap the scene name to play it.

### Start a story you thought of

```bash
fabula --invent "a heist that goes wrong in a hotel kitchen"
```

or the box at the top of the shelf in the browser. A world is written — three or four
rooms, two to four people, something one of them is not saying — as the same YAML an
author would have written, into the same directory the shipped worlds live in. The shelf
then lists it beside them and the engine plays it without knowing where it came from. Edit
it afterwards like anything else.

What makes it more than a wish is that **the engine disposes**. Ids are made here, not
taken from the model; references are dropped unless they resolve; exactly one character is
the player whatever it said, and they protect nothing, because the player is the one who
does not know. Nothing reaches disk that `inspect.py` has not read, and a world that
cannot be made playable is deleted rather than offered.

The rule a generator breaks constantly is the one about authored prose naming a fact — it
writes a room description that says "the second key" and hands the secret to the scenery,
where it satisfies `fact_spoken` before anybody has spoken. Those come back as complaints,
each gets one rewrite with the forbidden words spelled out, and what still fails is dropped
rather than shipped. What it had to rewrite or lose is reported, not buried.

It also writes the parts that make a scene **escalate rather than converse**: pressures,
which are the room having its own opinion about how long this can go on, and intentions,
which are what somebody does while nobody is watching. Both are asked for in plain words —
a sentence, a room, a number of turns, which secret it is about, whether they wait to be
alone — and the machinery is built here:

* **The trigger vocabulary never leaves `invent.py`.** The evaluator raises on a key it
  does not recognise, so a made-up one is not a pressure that misfires but a world that
  cannot be played at all. What the model says is `after_turns: 5`; what is written is
  `{"turns_elapsed": "> 5"}`.
* **A generated pressure only ever narrates.** An arrival needs somebody the scene said
  might turn up and a state change needs a scene written around it; either from a
  generator fires into a story nobody wrote.
* **`while_unsaid` is matched to a fact that exists** — by id, by its id read as words, or
  by anything it is recognised by out loud, because the model is naming it from memory two
  calls later. A name that resolves to nothing is dropped rather than written, since a
  trigger waiting on a fact this world does not have reads exactly like a pressure nobody
  wrote.
* **The player is never given an intention**, and nobody is put to sleep by a generator:
  an intention is what happens while nobody is watching, and somebody put under with
  nothing written to wake them perceives nothing for the rest of the scene.
* **The numbers are clamped, not taken.** A pressure that fires on turn one, forever, is
  worse than one that never fires.
* **A failed complications call loses the complications, not the world.** The scene still
  runs on the people in it, and the CLI says so: `dropped: the complications — nothing
  happens on its own here`.

And it writes **where the story goes from there**, which is the difference between a story
and a menu of scenes. The first scene is over the moment the secret is finally said out
loud; what follows is the morning after, and there are two of them:

```yaml
next:
  - scene: what_the_freezer_knew
    when: {character_at: {ines_cardoso: the_kitchen}}
  - scene: service_as_usual
```

The branch is built here and never asked for — a condition is a thing this file can build
and a morning is not. The engine picks the **witness** (somebody who is neither the player
nor the person keeping it, preferably somebody who does not start in the room, so their
being there at the end is something that *happened* rather than something that was set up),
names both situations to the model in plain words, and the model writes the two mornings.
Which is the one place the asymmetry pays a *story* back rather than only a projection:
the same sentence lands differently depending on who was standing there.

`character_at` reads where somebody is when the scene is over, which is a proxy for who
heard it — the same proxy the hand-written worlds use, and the only one the condition
language can see. If only one morning survives the prose repair it becomes the
unconditional successor, because a branch with one side is a condition that decides
nothing; if the call fails, the story stops after one scene and the CLI says so. The
shelf marks both mornings as chapters rather than starting points, derived from the
`next` that names them, so nobody is offered chapter three cold.

While generated worlds were getting branches, `inspect.py` learned to check every authored
condition — a pressure trigger, a scene's ending, a branch — in one place: that its keys
are ones the evaluator knows, and that the people and rooms it names exist. The unknown key
is the sharp one, because `evaluate_trigger` raises on it rather than quietly passing, so
an author who invents `after_turns` had written a world that ends the scene it is in with a
traceback instead of a morning.

Cost is a handful of calls, once, when the world is made: not per turn, not per character.

Measured on local models, because a generator is exactly the kind of thing that works on
paper:

* **Qwen2.5-1.5B cannot do it, and says so.** Shown an example of the shape, it hands the
  example back — Ashgrove, Elena, the music box, in answer to a heist in a hotel kitchen.
  It is asked once more, plainly; the second copy is refused rather than shipped, which is
  the right failure.
* **Qwen2.5-3B can.** The Grand Mercure: a pantry, a main kitchen, a walk-in freezer and a
  bar, a chef, a thief, and a recipe nobody will name. Six of its descriptions mentioned
  the secret and were rewritten; none had to be dropped.
* **The example leaks.** Even when the 3B designed its own world it put one of the
  example's characters in the cast and borrowed its scene title. Both are stripped by name
  now. An example is still the only thing that reliably fixes the shape — a model handed
  `"title": "two or three words"` writes a scene called *two or three words*, measured
  first time out.

### A cover for every story, a face for everybody in it

The client was prose on a page, which is what a story is and also what a terminal is. The
shelf this app sits on is one people browse with their eyes, and being the one that is all
text is not principled, it is bare.

So a world has cover art and everybody in it has a picture, and both are optional:

```yaml
# world.yaml
image: art/cover.svg      # relative to the world directory

# characters/tomas.yaml
image: art/tomas.jpg
```

**What matters is what happens when nobody drew anything**, because that is every world
the generator writes and three of the four shipped here. The authored file is the
*override*; the fallback is a plate drawn from the id — deterministic SVG, no files to
ship, nothing fetched from anywhere, and no world without a picture. One URL per thing
(`/worlds/{id}/cover`, `/worlds/{id}/faces/{character}`), always answering, so the client
never branches on whether art exists.

Two decisions make the drawn plates look deliberate rather than like broken avatars:

* **A cast is spaced around the wheel, not hashed.** Hashing each hue independently clumps
  — measured, on `winterlight`, which came out as five pinks. Each character takes a slot
  in a band anchored to the world's own hue, so five people are five colours *and*
  Ashgrove's cast looks like Ashgrove's cast.
* **No initials, and no faces.** A letter on a coloured square reads as a placeholder for
  a picture that never arrives. And a *face* nobody wrote would be a claim about somebody's
  appearance — this app asks the player for their own line about how they come across
  rather than inventing one for them, so a figure is the honest amount to say.

`worlds/ashgrove/art/cover.svg` is hand-drawn and is the one in the repo that exercises the
authored path. SVG because it is the one image format that is text, so it belongs in a repo
and diffs like everything else here.

A world directory is content, not code — generated here, copied off another machine,
downloaded from somebody — so the path inside it gets treated the way any other path from
outside would: resolved, and required to still be inside the world afterwards.
`image: ../../../etc/passwd` is why `art.authored()` exists rather than a bare
`world_dir / named`, and only types a browser can render are served. An `image:` that
points at nothing is a complaint from `inspect.py`, because otherwise it looks exactly like
a world that never had a picture and the author is left wondering why theirs is not
showing.

Faces appear beside spoken lines only. Narration has no speaker, and a half-heard line
through a wall is never given one — attaching a face would claim the listener knows who
that was, which is the whole thing the perception grading exists to avoid. The row of who
is here rides on `present`, which is already filtered to what this character can perceive.

### Living a story instead of finishing one

Scenes, end conditions and an "onward" button are game furniture, and not every story
wants them. Meeting a childhood friend after twenty years is not something you complete;
a school year is somewhere you stay. So **"Just let it run"**, next to the name you pick
where you choose a story, or `--open-ended`:

```bash
fabula ardenhall arrival --open-ended
```

Nothing ends, no seam is crossed, and the engine writes what happens next.

That last part is the whole feature, because the alternative is measurably worse than it
looks. Pressures have `max_fires`. On `ashgrove/the_dinner` over twenty-four player lines,
all three authored ones are spent by the seventh — and every turn from the eighth on is
*identically* two people talking with a narration between them. The story does not break
and nothing reports it. It goes slack, and no amount of authoring fixes that, because the
twentieth complication is the one nobody wrote.

```
                  authored pressures    something happens on
the_dinner, 24 lines   spent by turn 7        5 of 24 turns
  ... open-ended       spent by turn 7        9 of 24 turns, 4 model calls
```

**An invented situation is an ordinary `Pressure`.** Not a new event kind, not a special
case in the director, not a second path through perception — the same object an author
writes in `pressures.yaml`, built in `situations.py` instead of read off disk, and from
there `_fire` renders it and the world model filters it exactly as it does the authored
ones. So it is one function that returns a `Pressure`, and everything downstream is
untouched.

What constrains it:

* **Narration only, in a room that exists.** An arrival would need somebody the scene said
  might turn up and a state change would need a scene written around it; neither is a thing
  to decide mid-turn on a model's say-so.
* **It may not name a fact.** Same guard the generated room descriptions get — a situation
  using a secret's own words would raise the subject as scenery, satisfying `fact_spoken`
  before anybody in the story had said it.
* **The writer is never told the secrets.** Narrower than the director is allowed to be, on
  purpose: the deterministic guard can only catch a fact's actual keywords, and a writer
  that had been told could paraphrase around them — *"he looks at the empty space on the
  mantel"* names nothing and gives everything away. One that was never told cannot allude
  to it. It gets the rooms, who is present, and lines that were said out loud.
* **It competes rather than interrupts.** Scored like any other pressure, so somebody with
  something to say still wins the turn.

The trigger is drift: the room has gone quiet by the same threshold the sandbox pressures
use, *and* nothing has happened for four things the player said. Counted in the player's
lines rather than log positions, because a scene with four rooms of people racks up events
fast and still feels like nothing is happening — and deliberately not "no narration for a
while", since the atmosphere beats never run out and would mask the very state this has to
detect.

Cost: one model call per drift, none otherwise. Four across twenty-four player lines,
against thirty-five for every three.

How a story is played is stored with the story, not with the world — the same house can be
one you finish or one you stay in — so resuming it cannot silently change it, and a story
saved before this existed comes back with its scenes and endings intact.

### Taking it back

Every app on this shelf has some version of it — swipe for another answer, edit the line,
delete back to a point — because it is how you survive a model having a bad turn. This one
had `/again`, which throws away the last take and nothing else. Three transcripts against a
local 3B produced a character announcing the same intention eight times, another handing
over the secret he is keeping on the first line, and a third repeating somebody else's line
word for word. Deterministic guards catch some of that and will never catch all of it.

So: **take back to here** on any line in the browser, or `/back` (`/back 3`) in the
terminal. The scene resumes from the moment before it, and you play on.

What makes it more than a delete is that a scene is not only its log. Three kinds of state,
and only one of them is a delete:

* **The log is append-only and stays that way.** `events` has two triggers that abort any
  update or delete, and that is the foundation the rest of the engine reads to decide what
  is true. So a rewind marks events *withdrawn* rather than removing them — the record of
  what was played is intact on disk, and `get_events` stops returning them. Filtering at
  that one funnel is what makes "never happened" true for the projection, the bidding, the
  pressures, the endings and the reveal at the same time.
* **What was made of them is deleted outright** — beliefs, rehearsals, the private
  readings. A belief is not the record of a moment, it is somebody's impression of one, and
  the impression of a moment that has been taken back is nothing.
* **What they moved has to be replayed.** Trust drifts toward a floor each time somebody is
  seen refusing to answer; interactions count up; goals close when their subject is finally
  heard. None of that is a row a delete can reach. It is reset to the authored values and
  played forward again over what is left of the log, using the same deterministic rules
  that moved it — and costing nothing, because the expensive half of taking an event in is
  the private reading, and the readings that survive are already written down.

The visible proof is the header. Rewind past the beat where Maria walks into the kitchen
and the room says *with Tomás* again: her arrival is undone, not just her line.

The next take occupies the sequence numbers the withdrawn one had, so nothing downstream
sees a gap.

### Who is answering

A turn is thirty to ninety seconds, and until now all of it was one
undifferentiated dot. Every other application on this shelf fixes that by streaming the
reply token by token as it is written.

**This one cannot, and the reason is the guards.** Everything a character says is checked
*after* it is written and before it becomes an event: a line naming the secret its speaker
is keeping is refused, a narration that invents a fact or plays the player is dropped, a
line repeating one already said is thrown away and asked for again. Streaming would put the
words on screen ahead of all of that — and the sharpest of those guards exists precisely to
stop *the music box* reaching a player who has not earned it. Streaming and then retracting
tells them anyway. A guard you can read around is not a guard.

So what crosses the wire is a key and a job:

```
event: working
data: {"who":"tomas","name":"Tomás"}
event: working
data: {"who":"__narrator__","name":""}
event: working
data: {"who":"maria","name":"Maria"}
```

and the line under the scene reads *Tomás is answering*, then *the room is answering*, then
*Maria is answering*. The safety is structural rather than a promise about callers: there
is nowhere in a `Working` frame to put a sentence, and there is a test that asserts the
type has exactly two fields.

`fabula.llm.Watched` wraps the client the way `Routed` does, so no call site changes and
nothing below it knows anybody is looking. It reports only for keys that are somebody
taking a turn — the readings, summaries and classifications are two thirds of a turn's
calls and none of them is a person speaking in a room. It is installed only while somebody
is streaming, and removed when the last listener goes.

Which is also the truest picture of what this engine is doing: the room taking its turn,
one agent at a time. No application with a single model behind it could honestly draw it.

### Character cards, in and out

The category trades in cards: a PNG with a character's description hidden in a text chunk,
read by SillyTavern, Chub, Risu and most of the rest. Until now a world here was YAML only
this app understood, so nothing made with it could be given to anybody and nothing from
anywhere else could be played in it.

```bash
fabula --card ~/Downloads/ruth_vale.png      # play somebody else's character
fabula --cards ashgrove                      # write this world's cast out as cards
```

or the file picker under the invent box, and `GET /worlds/{world}/cards/{character}` for
the download. The picture on an exported card is the plate this app already draws for that
character — the same figure and the same colour, rasterised by evaluating the two shapes
per pixel rather than by adding an imaging library to an install story that is already the
weak part.

**Import is a file off the internet, and that is where the care goes.** Three rules:

* **Size and shape.** Capped before it is read, every field taken by name with a type
  check, and a `zTXt` chunk decompressed against a limit — a few bytes of deflate can be a
  great many in memory.
* **It cannot reshape the world.** Card prose becomes authored YAML that is then parsed, so
  the punctuation `fabula.player` refuses in a name is stripped here too. `{{user}}` and
  `{{char}}` are removed *before* that, because `{` is itself forbidden and stripping first
  would leave the bare words sitting in the prose.
* **The instructions are dropped on the floor.** A v2 card can carry `system_prompt`,
  `post_history_instructions` and a `character_book`: text whose entire purpose is to reach
  a model as *instructions*, and a lorebook is a mechanism for injecting text into a prompt
  on a keyword, which is a description of the attack. Honouring any of it would mean
  anybody who can get you to open a file can rewrite what the narrator may do and what a
  character may say. **The import takes the character and never their instructions** — the
  fields are absent from `KEPT`, so there is nothing to forget to check.

An imported card becomes one room with the two of you in it, the card's `first_mes` as the
opening, and **no model call at all**. Every other route into a world here generates one;
generating around a card would put words in a stranger's character's mouth before the
player had met them. It also means no facts and therefore no secrets — this engine's whole
mechanism is a thing one character knows and another does not, and a card does not say what
that would be. The world is playable and shallow, and the honest place to deepen it is the
YAML it just became.

**On the way out, one thing travels that is worth a warning.** A persona here may name the
secret its *own* character is keeping — `inspect` allows exactly that and nothing else — so
Tomás's card says he broke the music box, because his persona does. That is right for a
card, which is read by an application that needs to play him, and ruinous for a person
hoping to find out. So it travels, and the card's `creator_notes` says so. Maria's card
carries nothing of his, because a persona that named a fact which is not its own would not
have passed `inspect` in the first place.

### Play as somebody of your own

Everything in a world is authored, which is right for the parts a story turns on and
wrong for the one person you are: being handed Rook, or Elena, is being handed somebody
else's character to wear. So the last thing between picking a story and playing it is who
you are in it — skippable in one click, because sometimes you just want to start.

```bash
fabula ardenhall arrival --as "Wren Halloway" --look "A tall girl in a coat two sizes too big."
```

Two fields, and each has to reach the fiction or it is decoration:

* **The name** replaces the authored one *everywhere the author wrote it* — the other
  characters' personas, their notes about you, the pressures, the room descriptions. A
  name only the interface uses is worse than none, because then the Director's own
  description still says he read Rook's file on Tuesday. It is a substitution over
  authored text before it is parsed, so it can rename and nothing else: the ids
  underneath (`rook:` in a relationship map) are untouched, and a bare first name later
  in the same prose becomes the new first name. A name that could reshape the YAML it
  goes into — quotes, colons, brackets — is refused with the reason; any script is fine,
  and an apostrophe is quietly typeset as `’` so it cannot close a quoted scalar.

* **How you come across** becomes the first thing the room perceives about you: an
  ordinary event in your own room, filtered like any other, so the people standing there
  can react to it and the people elsewhere never see it. Which is why it is asked for as
  what *anyone can see* — a private truth put here would be handed to everybody in
  earshot, which is the one thing this engine exists not to do. A line that names one of
  the world's own facts is refused rather than quietly dropped: it would hand a secret to
  the room before a word was spoken.

Both are kept with the story, so resuming it is still your character.

### Your stories

A story is saved the moment you start it, and it is yours: one SQLite file, in
`~/.fabula/stories`, with no account anywhere near it.

```bash
fabula                              # what you have, most recently played first
fabula --resume a521c8c057dc        # pick one back up
fabula --delete a521c8c057dc
fabula ardenhall arrival --title "Tuesday"
```

```
Your stories  (/home/you/.fabula/stories)

  a521c8c057dc  Tuesday        the_interview     7 turns   08 Sep 17:49
  8062050d24f6  Ashgrove — the dinner   the_dinner    unplayed   08 Sep 16:12
```

Resuming lands on the scene the story was *left* on, not the one it began on — a story
that ran on into `the_interview` is picked up there, with everybody holding what they
held when it was put down. The turn count is the story's, not the take's: `/again`
rewinds it along with everything else.

The file is the whole story. Copy it to another machine and the story goes with it,
characters and grudges and all; `rm` is a supported way to delete one. The library is
just the directory they sit in — each file describes itself, so there is no index to
rebuild and nothing to fall out of sync. `--library ~/elsewhere` points at another one.

`--db scene.sqlite` still plays against a file you name, outside the library, which is
what the tests and the playtest harness use.

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

### Taking it again

A take you don't like is not something to live with. `/again` in the terminal, **Again**
in the browser: the last moment is thrown away and played once more from the same point.

It is deliberately not a confirmation prompt. The engine never stops mid-scene to ask
whether you want what just happened — a dialogue box is not a story — so it acts, and
this is how a take gets rejected. You are a director calling *again*, not a player being
asked to approve the world.

What makes it cheap is that everything durable lives in one SQLite connection. A savepoint
around a turn undoes all of it at once — the events, the beliefs encoded from them, their
readings, the rehearsals, the interaction counts, and any trust that moved when somebody
declined to answer. No per-subsystem bookkeeping, and no tombstones in an append-only log:
a discarded take was never committed, so it is not history. The retake even reuses the
sequence numbers the discarded one vacated, which is what lets a streaming client replace
it in place rather than showing both.

The cost of a write path that no longer commits one row at a time is that a turn is only
made permanent when the next one opens, or when the session is closed — so every client
closes its session on the way out, and `Session` is a context manager for the purpose.
Getting that wrong is silent and expensive: for a while it was, and the last exchange of
every session on a file database never reached disk. And a re-roll is a spoiler channel:
you can take a moment again until somebody confesses, and learn what the scene was holding.
That is not an invariant-1 break — that is about characters, not you — and in a
single-player story it is your story to spoil.

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

### What a turn costs in waiting

A turn is mostly waiting. Every character in earshot bids, every character reads back the
moment they just perceived, the winner speaks — each of those a round trip to whichever
model you pointed it at. They used to be made one at a time, so a turn cost the *sum* of
every round trip in it and the wait grew with the size of the cast: the fuller the room,
the slower it got to speak in, which is the wrong way round.

Bids are independent by construction — nobody's bid can see anybody else's, which is what
makes these separate agents rather than one model with a cast list — and so are the
private readings. Those two batches now go out together. At 300 ms a round trip, three
player lines:

| scene | calls | one at a time | together | |
|---|---|---|---|---|
| `ashgrove/the_dinner` (2 agents, split rooms) | 24 | 7.0s | 5.1s | 1.4× |
| `ashgrove/the_reckoning` (2 agents, one room) | 37 | 10.8s | 7.6s | 1.4× |
| `winterlight/the_manifest` (4 agents, one room) | 52 | 15.4s | 9.4s | 1.6× |

The win scales with how many characters are in the room with you, which is the case that
was worst. What is left is the part that cannot be parallel: each character has to hear
the last line before deciding to answer it, so the replies themselves are a queue.

`--workers N` sets how many calls a turn may have in flight (default 8; `--workers 1` is
the old engine exactly, and is what a single-slot local server or a tight rate limit
wants). Only the model call runs in a worker — every durable write happens afterwards on
the turn's own thread, in cast order, so a parallel turn tells the same story as a
sequential one and can still be thrown away whole by `/again`. There is a test that plays
the same scene both ways and compares every event, belief and relationship, and another
that fails if anything writes to the store from a worker thread.

### When the model does not answer

Everything past the model boundary is somebody else's machine: a key that was never set,
a rate limit, a laptop that went to sleep. None of it is a bug in the story, and none of
it costs you the story.

A take that fails is thrown away whole and the scene is left exactly where it stood —
the same savepoint `/again` uses, because half a turn is the worst outcome available:
a model that failed on the third of five characters would otherwise leave two of them
having heard something the others never will, permanently, in the file. The terminal says
so in a sentence and hands the prompt back; the service answers `502` and the session
stays open on the same scene. Retype the line and play on.

### Point it at your own model

There is a panel at the bottom of the shelf: the model id, an optional endpoint, how many
of a turn's calls may be in flight, and whether characters read back what they remember.
It saves to `~/.fabula/settings.json`, so the choice survives a restart, and it applies to
stories already open — the log, the beliefs and the trust are the engine's; the model is
only who gets asked next. A model the environment cannot reach is refused with the reason
rather than saved, because saving it would mean every story from then on failing at its
first line, several clicks from the screen that caused it.

**The panel will not take your key, on purpose.** Credentials reach the engine through
the environment and nowhere else — a browser form posting an API key into a JSON file
would be a worse place for it than the environment, dressed up as a better one. So the
panel *looks* instead: it names which provider variables are set (names only; the values
are never read by the app) and tells you the exact file to put a missing one in.

Every entry point also takes `--model` (any id litellm understands) and `--api-base` (any
OpenAI-compatible endpoint — a hosted proxy, an aggregator, a local server). A flag is
for that run and wins over the file; the panel says so rather than quietly disagreeing
with the process it is running in.

```bash
export OPENAI_API_KEY=...
fabula-playtest worlds/ashgrove the_dinner --model gpt-4.1-nano

# against an OpenAI-compatible proxy: prefix the id so litellm speaks that dialect
export OPENAI_API_KEY=...                       # whatever key the proxy expects
fabula-playtest worlds/ashgrove the_dinner \
    --model openai/<their-model-id> --api-base https://<host>/v1
```

The same two flags work on `fabula`, `fabula-serve` and `fabula-measure`.

### Two models: one that writes, one that files

Most of a turn is text nobody ever reads. Measured on `ashgrove`, three player lines:

```
35 model calls
  24  the model nobody reads   interpret (23), beat (1)
  11  the model you are paying for   a character's line (6), narration (5)
```

The 24 are the per-memory readings — one sentence of "what does she now think is going
on", stored as a belief and surfaced only by `/reveal` — plus the summaries that feed
context, the classifications whose every referent is checked afterwards, and the
director's beat, which is one id out of a list the engine handed it and refuses anything
else from. The 11 are the product.

So `--fast-model` names a cheaper model for the first group, and `--fast-api-base` says
where it lives when that is somewhere else — a hosted model writing, a local one filing:

```bash
fabula --model anthropic/claude-sonnet-5        --fast-model openai/qwen2.5-3b-instruct        --fast-api-base http://127.0.0.1:8090/v1
```

Or the two extra boxes in the Model panel, saved to `~/.fabula/settings.json` like the
rest. Leaving them empty means what it meant before: one model for everything.

Note the seam is what a call is *for*, not who makes it. Splitting by "the director versus
the characters" would move one call in thirty-five, because a character makes both the
expensive call and most of the cheap ones — and the director's single call is already
confined to a closed vocabulary, so a better model there has little to be better at.
Making a world stays on the good model for the same reason in reverse: it is a handful of
calls once, and the one job measured to *need* the better model.

Two things worth knowing before turning it on:

* **The readings feed the context the writing model gets.** A cheap model filing badly
  does not show up as bad filing; it shows up as a worse scene one turn later. `/reveal`
  is where to look — it is exactly the readable artefact for judging whether the second
  model is filing sensibly.
* **Invariant 1 does not depend on how many models there are.** The projection is
  deterministic and happens before every call, so a second model cannot be told anything
  the first could not. There is a test that asserts it on a real turn.

Both models are checked against the environment before a story opens, for the same reason
one was: a second model is a second way to lose a scene to a provider traceback several
turns in.

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

Played against the same 1.5B (`llama-cpp-python` serving Qwen2.5-1.5B-Instruct, pointed at
with `--model openai/… --api-base http://127.0.0.1:8090/v1`), three things came back worth
writing down:

* **The beat choice survives a small model.** Asked which of two beats a moment wanted, it
  answered `lull` — the bare id, taken as given. The answer space is three labels, which
  is small enough that a model this size can hit it. That is the argument for a closed
  vocabulary rather than an instruction.
* **The first run of the chooser never fired at all.** Only one beat was ever on offer, so
  there was nothing to choose between; `the_room` is now always offered as an alternative
  when something else already is.
* **Characters repeat themselves.** Maria said one sentence twice, word for word, inside
  four lines — with her own prior line in the context she was given — and then said
  Tomás's line back at him. The engine catches an exact repeat now (their own last line,
  the line just said in front of them, or anything of their own from earlier in the scene
  once it is longer than a few words), tries once more, and lets them say nothing rather
  than say it twice. Short lines are left alone: "No." twice is a person.
* **The narrator played the player.** "Elena's finger brushes the dusty glass of the photo
  album" — which Elena never did. The prompt has said never to describe the protagonist
  since M0. A narration naming them is now dropped, the same way one naming a fact is:
  playing the one character somebody else is holding is the worst thing the narrator can
  do, so it gets the deterministic version of the rule rather than the asked-nicely one.
* **What is left is what cannot be caught structurally.** The same run put a laptop in a
  1990s kitchen and had Tomás announce his own secret in his second line. "Introduce
  nothing that is not already established" cannot be checked by keyword, because the set
  of things that do not exist cannot be enumerated. Treat it as unproven on any model you
  have not measured.

### What the browser client does

`fabula` opens it; `fabula-serve` runs the same thing without a browser, for a machine
you reach over ssh.

It opens on your stories rather than on a menu of things to begin. Underneath them are
the worlds, each with what kind of story it is, and the scenes a story can *start* from —
a scene another scene leads to is a chapter, and offering it cold is how a menu of scenes
reads. What you play, who you play, and who is in the room with you are on the card.

In the scene it renders *your character's projection*. Dialogue is attributed; things you
half-hear are dimmed behind an `unclear` tag and deliberately left unattributed, because
your character does not know who that was. Your own line appears the moment you send it,
held back until the turn lands — and vanishes again if the turn failed, because then it
did not happen. While the room is answering it says so: a turn is several model calls and
a blank screen for ten seconds reads as a crash.

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
| `director.py` | Speaker selection, pressures, time, turn budget | Only to pick a beat from the offered ids |
| `beats.py` | What the room could use next, from a closed set | **None, ever** |
| `narrator.py` | Describes perceivable action; renders pressures | Yes |
| `agents.py` | Bids to speak; generates utterances | Yes |

`world.py` has zero model calls by design — it is the thing that makes invariant 1
enforceable rather than aspirational.

Supporting modules: `session.py` (the surface both clients use), `chronology.py` (time
skips, off-screen intentions), `pressures.py`, `persistence.py`, `summaries.py`,
`db.py`, `loader.py`, `inspect.py`.

`inspect.py` is everything that has to be true of a world before anybody plays it: no
authored prose naming a fact, no persona naming a secret that is not its own, exactly one
player per scene, every id resolving, nobody asleep with nothing to wake them, no scene
leading somewhere that does not exist. Those rules used to be four tests parametrised over
the four worlds in the repo, which enforced them exactly as often as somebody remembered
to run pytest. `complaints(world_dir)` returns them as a list, so a world that arrives
from anywhere — copied in, generated, edited by somebody who has never read this file —
gets the same reading the shipped ones get. Every rule has a test that breaks a world in
exactly that way; a validator that never says no is a promise nobody checked.

### What gets narrated, and who decides

The narrator used to be handed the triggering event and asked for "one or two sentences
of scene-setting narration". It knew what had just happened and nothing about what the
scene needed, so it fired rarely and wrote whatever the last line suggested. A scene
played that way is people talking in a white room.

The director says what to narrate now. It is the only omniscient component, which is the
whole design problem: an instruction written freely from what it knows would be a channel
from the world log straight into prose everybody in the room perceives — "narrate that
Tomás is nervous about the music box" is a leak with extra steps. So a **beat** is one id
from a closed set, plus something anybody standing there can already see:

| beat | when | what it carries |
|---|---|---|
| `the_room` | the player walked in | the room's authored description |
| `lull` | three straight lines of talk | — |
| `after_deflection` | somebody visibly did not answer | — |
| `held_back` | somebody present has said nothing for a while | their name |
| `object` | an authored item nobody has looked at | its name |
| `alone` | the player spoke and nobody is there | — |
| `arrival` / `departure` / `time_skip` | the story moved | — |

Which beats are *available* is decided by code in `beats.py` from the event log and the
world — never from beliefs, trust, goals or what anybody protects. Which one the moment
wants is a judgement, so the director asks the model: it answers with one id, the id has
to be one it was offered, and anything else falls back to the engine's order. It cannot
invent a beat, write an instruction, or reach past the closed vocabulary, and what it is
shown is what the room can see, because its whole job is picking between three labels.
That call is made only once the narrator has won the turn and only when more than one
beat is available — 1 to 2 extra calls across four player lines, measured. `--no-direct`,
or the checkbox in the settings panel, takes them in the engine's order instead.

Atmosphere beats wait for `COOLDOWN` events of quiet in *that room*; an arrival never
waits, because losing it is worse than one paragraph too many. And every narration is
checked before it lands: one that names a world fact is dropped, because it would raise
the subject in front of the room and end an arc nobody had spoken in.

And a turn always answers. If nobody bids and no beat is due, the room takes the turn
rather than nobody having it — *perceived*, not merely appended, because a pressure firing
two rooms away is the story moving and still silence where the player is standing. The
beat it falls back on is anchored on the last line they actually heard, in the room they
are actually in: `last_event` can be two rooms away, and putting that in the prompt would
narrate their room out of words they never perceived.

Measured on the shipped scenes, four player lines each: 6 narrations to 12 spoken lines in
`the_dinner`, 7 to 10 in `arrival`, 8 to 10 in `the_manifest` — against 4, 5 and 6 before.
The extra prose costs a model call each, which is the bill for asking for it.

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

## Stories

A scene declares where the story goes from it. Entries are tried in order, first match
winning; one without a `when` is the fallback, and an empty list ends the story. The
conditions are the same `evaluate_trigger` as endings and pressures — an author should
not need a third language to say *"if she was in the room"*.

`/next` in the terminal, **Go on** in the browser, `POST /sessions/{id}/next` on the
wire. The session id survives the seam, because a client holds a story rather than a
scene. Underneath it is one database for the whole run, which is what makes the durable
state — beliefs, trust that moved, goals that closed — finally have somewhere to land.
Going on settles the turn that was open: a seam is not a take you can ask to have again.

## Four worlds

Everything here was built against one world, and code fitted to one shape looks general
until a second one arrives. So there are two, deliberately unalike:

| | **ashgrove** | **winterlight** | **vilamar** | **ardenhall** |
|---|---|---|---|---|
| | a house after a funeral | a station on the plateau | a house by the sea | a school, on the first day |
| rooms | 2, mutually audible | 4 in a chain, one one-way | 2 | 2 authored, **the rest found** |
| cast | 2 agents + you | 3 agents + you, **and one who may walk in** | 2 agents + you | 1 agent + you |
| secrets | one, one holder | **two, two holders, from each other** | one, one holder | one he keeps from you |
| language | English | English | **Portuguese** | English |
| story | — | — | — | **four scenes, branching** |
| items | — | — | — | **two, one of them about you** |

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

## The engine speaks no language of its own

`vilamar` is in Portuguese, and it exists to prove that. Everything the engine can put
inside a character's perception used to be an English constant in Python — a Portuguese
scene got English injected into it, and two of its systems silently stopped working
altogether:

```
Portuguese  "Ele quebrou a caixa de música da avó"  ->  ['caixa', 'passado', 'quebrou', 'sica']
Russian     "Он разбил музыкальную шкатулку"        ->  []
Japanese    "彼は祖母のオルゴールを壊した"              ->  []
```

That is `_significant_words`, which drives rehearsal and retrieval. `[a-z0-9']+` on a
lowercased string is an ASCII range: `música` came back as `sica`, and Cyrillic and
Japanese came back as nothing at all, so a memory in those languages could never be
rehearsed or retrieved. It is `\w` and `casefold` now. CJK has no spaces to tokenize on,
so a token there gets character bigrams instead — not segmentation, and a real tokenizer
would be better, but it is signal rather than none.

Everything else moved out of Python and into `world.yaml`:

```yaml
language: pt-PT
phrasing:
  degraded:
    utterance: "vozes abafadas vindas d{location}, palavras indistintas"
  duration: {minute: "um minuto", minutes: "{n} minutos", ...}
  time_skip:
    awake: "({duration} a passar, e sente-se cada um deles)"
  client:
    now_in: "Estás n{room}."
  stopwords: [ainda, aqui, como, depois, ...]
```

The line that separates what an author writes from what the engine does is **enumerable
things versus unbounded meanings**. A fact has a name, so `keywords` work in any script —
`mentions_fact` was already language-neutral and needed no change. A *meaning* — "someone
arrived", "I told him last week" — has no list, in any language, so asking an author for
one is the wrong shape. Anything of that kind belongs to a model, not a keyword file.

Two more lines held on purpose. **Commands stay the same everywhere** (`/go`, `/wait`) —
a verb that changes name per world is a verb nobody can document. And the English worlds
are byte-identical: `write_in()` returns nothing for English, so the prompts every
measurement in this project was taken against are untouched.

Defaults merge per key, so an English world needs none of this and a world that restates
one line keeps the rest. A half-translated world is an authoring error, caught by the
world guards rather than at load — it should fail review, not refuse to start.

## Authoring

Everything authored is plain YAML on disk — diffable, shareable, legible to a coding
agent asked to change it.

```
worlds/<name>/
  world.yaml          title, blurb, rooms, adjacency, facts the story can turn on
  characters/*.yaml   persona, traits, goals, intentions, relationships
  pressures.yaml      authored complications
  scenes/*.yaml       title, premise, cast, starting positions, mode, turn budget
```

`title` and `blurb` on a world, and `title` and `premise` on a scene, are what the shelf
is made of — a directory name and a scene id are not a reason to click anything:

```yaml
# world.yaml
title: Ashgrove
blurb: >-
  A house, a family, and something one of them has not said out loud.

# scenes/the_dinner.yaml
title: The dinner
premise: >-
  Your brother has been strange all evening, and your sister is in the next room.
```

A scene's `opening` is the first thing the player reads — every other app on this shelf
has one, and without it the first thing asked of somebody is "what do you say" to a room
they know nothing about. It is appended as an event addressed to the player, at
`audibility: private`, so the choke point every leak test covers returns `none` for
everyone else in the room. That is what makes the second person safe here:

```yaml
opening: >-
  Sunday at the house you grew up in. Tomás has been at the kitchen table since six
  and has said about twenty words, none of them about anything. Maria is in the study
  with the door not quite shut. Nobody has eaten yet.
```

Write it from inside your player character's head and no further: Elena does not know
what her brother is sitting on, so her opening does not either. And it must not name a
world fact — the player perceives it in full, so a keyword there satisfies `fact_spoken`
and an arc can end on its own opening paragraph. Both are tested for every shipped world.

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

A `goal` is a standing want, and it is mechanical too. `reactivity` is a reflex — how
hard you jump when something comes up. A goal outlives the moment, and it can stop being
open:

```yaml
goals:
  - id: protect_secret
    description: Keep the broken music box a secret from Maria
    priority: 0.9
    about: music_box     # optional: the fact this goal turns on
```

Naming a fact is what makes it more than prose. It reaches the character's own prompt
(never anyone else's, and never the narrator's — what somebody wants is not something the
room can see), it raises their bid when the subject is raised, and it **closes when they
hear the subject come up**, because a secret you are keeping stops being one you are
keeping the moment it is out.

Closing is judged from that character's own projection, not from the log. A secret that
came out in a room Tomás was not in has not stopped being a secret *to him*, and he goes
on guarding it — the same rule that decides whether a fact was really spoken. A goal
written as prose alone never closes and never moves a bid, which is honest: nothing here
can read "sort out grandmother's belongings fairly" and judge it done.

An `intention` can also change what state a character is in, which today means sleep:

```yaml
intentions:
  - id: turns_in
    description: pulls the curtain across the berth and turns in
    location_id: bunkroom
    ready_after_minutes: 90
    state: asleep
  - id: wakes_for_the_obs
    ready_after_minutes: 210
    state: awake
```

**Asleep in a room is not the same as being in it.** A sleeper perceives nothing — not
into their context, not into their beliefs — and does not bid, because asleep is absent
rather than quiet. Their own state is still theirs to know, so going under and coming back
up are both felt. Sleeping through something said in front of you is the sharpest
asymmetry here, and it costs nothing to allow: the filter only ever *removes* perception.

A jump in time still reaches them, which is the point — this is what spec §8 means by
elapsed time being perceived non-uniformly. One skip, two experiences of it:

```
Maria (asleep) : (a gap — you surface to find 2 hours gone, unfelt)
Elena (awake)  : (2 hours pass, and you feel every one of them)
```

Turning in is an authored moment, never something the engine decides for somebody — an
intention or a `state_change` pressure. The world guards catch a character put under with
nothing to wake them, which would otherwise leave them perceiving nothing for the rest of
the scene, in silence.

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

### A line that reports something rather than only adding to it

"She knows. I already told her about the music box." That is not just
dialogue — it is a claim that a perception happened off-screen, and if the engine
ignores it, Maria walks in later ignorant of something the scene established.

Keyword lists cannot catch this. A *thing* has a name, so `mentions_fact` works in any
language; a *meaning* does not, and asking an author to enumerate the ways of saying "I
told him" is asking them to enumerate a language. So it is a model's judgement — and the
**director's**, not the narrator's, because the director's output is narrow and checkable
where prose is not.

**The model proposes, the engine disposes.** The classifier returns a structure, never
text:

```json
{"reports_telling": true, "to": "maria", "fact": "music_box"}
```

and nothing is trusted until every field survives:

* `from` is whoever actually spoke the line — it is never read from the answer at all
* `fact` must be an id the author wrote; a fact cannot be invented here
* `to` must be a character in the scene or waiting to enter it
* **the speaker must already have known it**, from their projection *before* this line —
  otherwise anybody bootstraps knowledge by asserting it, and a player says "everyone
  already knows my secret" and the scene dissolves

A hallucinated classification fails one of those and becomes a no-op. What it produces is
a private beat placed where the recipient is and addressed to them, so the existing
perception rules do the work: they get it in full, nobody else gets it at all, and from
there it is an ordinary perceived event — it becomes a belief, it shows in the reveal, and
the turn it landed in can be taken again.

The cheap gate is the mechanism that already works: only a line naming one of the world's
facts is classified at all, so almost every line skips the model call. And a reported
telling does not count as the fact being *said out loud* — one pair of ears in the past
tense is not the room, and an arc waiting for somebody to say it is still waiting.

Stated plainly: a misclassification grants somebody knowledge nobody on screen conveyed.
The backdated beat makes that formally a perception, so invariant 1 holds on paper, but it
was created on a model's say-so. Three things bound it — you can only pass on what you
know, nothing is hidden, and the turn can be taken again.

### Things that carry a fact

The only kind of object worth modelling here. A mug is scenery and belongs in a room
description; a key that opens a door is an adventure game and a different product. A
letter, a logbook, a sealed file is a **second channel for the asymmetry the engine
already turns on**:

```yaml
items:
  the_sealed_file:
    name: the grey folder on the desk
    location_id: office
    reveals: the_seventh_file
    text: >
      Marlow Academy — incident report, 4 June. Two paragraphs are struck
      through in a different hand... Your name is in the margin, twice, circled.
```

`/read grey folder`, and it becomes **two** events, because reading is two things: the
room sees you open it, and what it says is private and addressed to you.

```
rook   saw the act: True   read the contents: True
vance  saw the act: True   read the contents: False
```

Both were standing in the same room. Nothing new was needed for that — a private event
addressed to one person is what the perception rules already do.

One consequence falls out rather than being built: **reading is not saying.** A fact
counts as spoken when somebody *other than the actor* hears it in full, and the private
half of a read has no perceiver but its own reader. So an arc waiting for somebody to say
it out loud is still waiting, and you can know something you have not admitted.

Items are authored, necessarily — what they carry is a fact, and a fact is exactly the
thing an author names. Two guards hold across every world: an item pointing at a room or
a fact that does not exist, and an item whose **name** contains a fact keyword. That
second one matters because the name goes into the public beat, so a badly named object
would count as saying the thing out loud just by being picked up.

### Rooms the author did not write

A school has corridors. Nobody wants to write them all, and a story that answers *"there
is no library to go to"* is answering with its own scaffolding. So a world can let the map
grow:

```yaml
discover_rooms: true
```

Then `/go the library` makes one. What keeps that safe is that almost nothing about a room
carries weight — **except its adjacency, and the engine decides that**. A found place hangs
off exactly the room it was reached from, by one symmetric edge, so it behaves like any
other doorway: you hear the hall you stepped out of, and it hears you. A model choosing
edges could join the library to the headmaster's office, which would be a leak rather than
a bad sentence.

So the model writes a name and two lines of description — prose, checked the way every
other generated line is. A description naming one of the world's facts is thrown away and
the room stands bare, because a description reaches the log as narration and would let the
narrator raise the subject just by describing the place.

Found rooms are stored per world, so a corridor found in the first scene is still there in
the fourth, joined to the same place. Two things follow honestly: the map is a **tree**,
not a map — you can always go back the way you came, and two found wings never join up —
and matching is by name, so "library", "the library" and "The Library" are one place.

### Somebody who was not in the room when it opened

A scene's `cast` is who is there at the start. `may_arrive` is who might turn up:

```yaml
cast: [ilse, yusuf, nadia, ana]
may_arrive: [petra]          # written for this world, not in the room yet
```

An authored `arrival` pressure brings them on. Nothing else can — and the narrator least
of all, because it writes prose, while a person who arrives has to arrive as an *event*
with an actor id, and only the director assigns those. A character the narrator invents
stays a ghost in a sentence: it cannot speak, act, be perceived as an actor, or hold a
belief.

The part worth knowing is what makes their memory right. **Nothing backfills it.** They
wait *off-stage*, which is deliberately not a room — `distance` breadth-first-searches
from the listener's room, so an id in no `rooms` map is unreachable in both directions:

```
offstage -> kitchen: -1        kitchen -> offstage: -1
```

Every moment before their arrival therefore resolves to "none" through the ordinary
perception path, with no special case in the code the leak tests depend on. They walk in
knowing only what they walk in on, and there is no decision to get wrong about what they
might have overheard. From the arrival event onward they bid, perceive, remember and are
seen exactly like anybody else.

```
the secret is said in the mess, while she is still out on the line
    what she perceives from off-stage: []
she arrives
    [full] The porch door bangs.
    [full] You picked a night for it.
    her context names the flight: False
```

Both halves are guarded across every world on disk: an `arrival` pressure whose actor is
neither cast nor awaited fails review, as does a scene that awaits somebody it has
already cast.

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
| `POST /invent` | make a world from a sentence and start a story in it |
| `GET /catalogue` | worlds and their scenes, with the ones a story starts from marked |
| `GET`/`PUT /settings` | which model answers — never a credential, in either direction |
| `GET /stories` | your saved stories, most recently played first |
| `POST /stories` | start one (optionally `character: {name, look}`) → the same state as `POST /sessions` |
| `POST /stories/{id}/resume` | pick one back up, on the scene it was left on |
| `DELETE /stories/{id}` | delete one |
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

* The classifier only reconstructs a telling to somebody the author wrote. "I told my
  brother" — where no brother exists in the world — is refused rather than conjuring one.
  Introducing a character nobody wrote is the same director-level shape and is not built.
* The line the classifier is most likely to get wrong is wanting-to-tell versus
  having-told, and it is the one with a consequence. There is a test pinning it, but a
  test with a scripted answer proves the plumbing, not the judgement — that needs
  measuring on a real model.
* Nothing wakes a sleeper on noise. Waking is an authored moment — an intention that
  comes due, or a pressure — so a shout next to somebody who has turned in does not rouse
  them. That would want an audibility threshold on the perception path, and it is a
  feature rather than an oversight until somebody asks for it.
* CJK retrieval is a heuristic, not segmentation. A Japanese or Chinese token gets
  character bigrams, which finds real overlap between two lines about the same thing but
  is nobody's idea of a tokenizer. A language with more than two plural forms (Russian
  has three) can also only pick the least wrong `duration` template — that is a limit of
  the template shape, not something an author can work around.
* `Room.adjacent` values are not read. The field maps a neighbour to how well sound
  crosses that doorway, which would let an author seal one; perception uses only the
  presence of an edge and the *event's* own audibility. Until it is wired up, a one-way
  edge is how you make a room you cannot hear out of.
* **A character the narrator invents is still a ghost, and stays one.** It is told to
  introduce no people who have not appeared, and a small model ignores that — the
  `winterlight` run produced an interrogator and an elderly man who are not on the
  station. Nothing promotes them: an invented person cannot speak, act, be perceived as
  an actor, or hold a belief, because all of that needs an `actor_id` the director
  assigns. It is a quality bug rather than a leak. Bringing somebody genuinely new in is
  `may_arrive` plus an authored pressure — a person the author wrote, at a moment the
  director chose. Deciding to introduce somebody the author never wrote would be a
  director-level classification, which does not exist yet.
* Prompt adherence is the soft spot. Structural rules hold regardless of model (the
  narrator cannot narrate the player, because it does not bid), but the ones that live in
  prompts — invent no props, never speak for a character, don't raise what you guard —
  are only as good as the model reading them. Measured on a 1.5B local model, the
  "what you guard" reminder made no difference at all: 4/16 openings blurted the secret
  with it, 3/16 without. Playing `winterlight` on the same model made it vivid: Yusuf
  recited his own persona aloud three times ("I fix the transfer valve. It's been a
  quarter turn open"), and the narrator invented two people who are not on the station.
  Nothing structural gave way. Treat every prompt-level rule here as unproven until
  measured on the model you actually ship.
* Relationship **affinity** still does not move during play; only trust does, and only on
  a witnessed refusal to answer. Moving either from the *content* of what is said needs a
  model deciding whether someone was being sincere, and a wrong call there quietly
  rewrites a character's inner life — which is the correction invariant 2 forbids.
* Sessions live in memory, so restarting the service drops them — but a story does not:
  it is on disk from the moment it starts, and `POST /stories/{id}/resume` opens a new
  session on it. What is lost across a restart is the open turn (the one `/again` could
  have taken back), not the story.
* A story open in the service is one turn ahead of its own file, because the turn in
  progress is deliberately uncommitted — that is what makes it discardable. `GET /stories`
  overlays what the live session knows, so a card never contradicts the screen; a second
  process reading the same directory would see the older state.

## License

None yet.
