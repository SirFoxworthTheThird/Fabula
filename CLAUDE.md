# What Fabula is

An app for **roleplaying stories** — the same shelf as DreamJourney AI, Fictionlab,
Xoul AI and the rest of the AI-roleplay category. People come here to play through a
story, of whatever kind they feel like: a heist, a romance, a haunting, a school, an
argument in a kitchen.

Two things make it different from the rest of that shelf:

1. **Every character is a separate agent.** Not one model puppeting a cast, but several,
   each with its own private knowledge, memory and reasons to speak. That is what makes
   a character able to be genuinely wrong, genuinely surprised, and genuinely unable to
   use something they never heard.
2. **The application runs locally.** Your machine, your stories, your file on disk. No
   account and no service in the middle. Which model answers is your choice — hosted,
   an aggregator, or something you run yourself — and the key is yours.

Everything else is in service of those two.

## Reading this before making a judgement call

Two rules that keep getting broken, both by me:

**Information asymmetry is the foundation, not the product.** The spec calls it "the
point of difference" from character-chat apps, and it is — it is what makes the fiction
trustworthy. But nobody plays a foundation. If a design question can be answered with
"does this preserve invariant 1" *and* "does this make a story better to play", the
second question is the one that decides it. Twice in this project the first question was
allowed to decide alone, and both times the answer was wrong.

**Breadth beats depth on any single mechanic.** Perception grading, degraded hearing,
trust movement, withholding, memory tiers — these are good, and none of them is the
product. A roleplay app that models earshot beautifully and cannot let somebody start a
story they thought of is not an app. Before deepening any one system, ask whether the
shallow version of a system that does not exist yet would be worth more.

## What "all kinds of stories" costs

The shipped worlds are quiet literary drama, because that is what the first one was and
the rest followed it. That is a sample of one genre pretending to be a sample of the
form. A heist, a romance, a horror scene and a comedy make different demands — of
pacing, of who speaks when, of what a "secret" even is — and none of them has been tried.

## What "local" means, and what it costs

**The application is local, not the models.** It runs on your machine — a package you
install and a service you start — and it keeps your stories in a SQLite file you own.
No account, no SaaS in the middle, nothing leaving the box that you did not point it at.
Which model answers is entirely your choice: a hosted endpoint, an aggregator, your own
llama server. Chosen in the app itself — a panel on the shelf, saved to
`~/.fabula/settings.json`, applying to stories already open — or with `--model` and
`--api-base` for one run. The key comes from the environment and only from there: the
panel reports which provider variables are set and where to put one that is not, and has
nowhere to store a credential even if it wanted to.

So model cost and latency are the *user's* bill and the *user's* wait, not a hardware
ceiling — which makes them an engineering problem rather than a disqualifier. They are
still real. A turn costs one model call per agent that bids, plus the narrator, plus one
per remembered moment per character:

```
ashgrove    (2 agents, 3 player lines) -> 33 model calls
winterlight (4 agents, 3 player lines) -> 51 model calls
```

The bids and the readings now go out together rather than one at a time, which at 300 ms
a round trip takes `winterlight` from 15.3s to 9.4s for those three lines, and the win
grows with the number of people in the room with you. What is left is the part that
cannot be parallel: a character has to hear the last line before deciding to answer it,
so the replies are a queue by nature.

The wait that was worst was not a turn at all: it was the click. A scene's opening costs up
to eight calls and all of them used to sit between **Begin** and the first thing on screen
— 6.1s at a two-second model, measured in the browser. It is now split: opening a story
asks the model nothing and lands on the scene's authored first words, and the room opens
itself afterwards on the stream. The same click, 0.14s. The rule that generalises: a wait
the player spends *inside* the story, watching it happen, is not the same wait as one they
spend in front of a button, even when the clock says it is. Look for the others.

The next thing available here is deferring the readings to the end of a turn instead of
issuing them after each event — in-scene beliefs are only read back by the reveal and by
the guard on a reading itself, so a turn's readings could be one batch rather than eight.
That one changes when a belief becomes durable within a turn, so it is not free.

The other half of the same bill is *which* model answers, and the split that pays is by
job rather than by who: measured on `ashgrove`, 24 of a turn's 35 calls are the memory
readings, the summaries, the classifications and the director's beat — none of which
anybody reads — against 11 of character lines and narration, which are the product.
`--fast-model` sends the first group somewhere cheaper. The trap is that the readings feed
the context the *writing* model gets, so a bad second model shows up as a worse scene a
turn later rather than as bad filing; `/reveal` is where to look. Note what the measurement
ruled out: splitting "the director" from "the characters" would have moved one call in
thirty-five, because a character makes both the expensive call and most of the cheap ones.

The narration beats cost one call each, and the director choosing between them costs one
more per narration — measured at 1 to 2 extra calls across four player lines, because the
choice is only made once the narrator has won the turn and only when the moment could be
more than one thing.

The rule to carry: a feature that adds a per-character or per-moment model call is
spending somebody's money and somebody's patience. Decide how it will be paid for while
designing it, not afterwards. `--no-interpret` is a symptom of not having done that.

What being a local *application* does demand, and what is missing:

* ~~**Install without a toolchain.**~~ Done: `uv tool install git+…` and then `fabula`.
  uv fetches a Python if the machine has none, so "install it and type its name" is true
  on a box with no toolchain. What made that possible was not the install line but
  finding that the app did not work when installed at all: `worlds_root` defaulted to the
  relative path `worlds`, so a wheel installed anywhere but a clone opened on an empty
  shelf, and `--invent` wrote a generated world into whatever directory you were standing
  in. The worlds now ship inside the package and are copied to `~/.fabula/worlds` on
  first run. The rule that generalises: **a default that is a relative path is a bet that
  the user is standing in your source tree.** Grep for the others before adding one.

  What is still missing is a single file somebody can double-click — a frozen binary,
  which means a release pipeline and freezing litellm, and neither is free.
* ~~**A library of your stories.**~~ Done: one SQLite file per story in
  `~/.fabula/stories`, listed, resumed and deleted from the CLI, the API and the web
  start screen. Sessions still live in memory, but a story no longer depends on one.

## What is authored, and the problem with that

Everything: worlds, rooms, facts, characters, pressures, scenes, items — all hand-written
YAML. That is right for the parts a story turns on and wrong as a way to *start* one. The
audience for this app does not write YAML. Any plan that ends with "and then the author
writes the file" has not finished.

The second dent, and the one that finishes the sentence: **a world can be written from a
sentence.** `fabula --invent "a heist that goes wrong in a hotel kitchen"`, or the box at
the top of the shelf, writes the same YAML an author would have written and the engine
plays it without knowing where it came from. What keeps it honest is that the rules a
world has to satisfy are now code (`fabula/inspect.py`) rather than tests: nothing reaches
disk that has not been read, and what cannot be made playable is deleted rather than
offered. Measured: a 1.5B cannot design one and refuses; a 3B can.

It writes the complications too — pressures and intentions, the parts that make a scene
escalate rather than converse. Asked for in plain words; the trigger vocabulary never
leaves `invent.py`, because the evaluator raises on a key it does not recognise, so a
made-up one is a world that cannot be played rather than a pressure that misfires. A
generated pressure only ever narrates, a generated intention is never the player's, and a
complications call that fails loses the complications rather than the world.

And it goes somewhere: the first scene is over when the secret is said out loud, and what
follows is one of two mornings, chosen by whether the one person it is news to was standing
there for it. That branch is built in `invent.py` and never asked for — the model is told
the two situations in words and writes the prose — which is the one place the asymmetry
pays a *story* back rather than only a projection.

What a generated world still does not have against a hand-written one: no goals, no items,
and only ever the one shape of story — a secret, and the morning after it came out. Every
premise gets that shape, which is the same mistake as the shipped worlds all being quiet
literary drama, arrived at from the other direction.

## What the rest of the shelf has

Measured against the category rather than against the spec, three things were missing and
one is now built.

**A way to fix what the model just wrote.** Swipe, edit, delete-back: it is the first
control anybody reaches for, and this app had only `/again`. Now any line can be taken
back, in the browser or with `/back`. The design constraint worth remembering: the event
log is append-only and defended by SQLite triggers, so a rewind *withdraws* events rather
than deleting them, and `get_events` filters at the single funnel every other subsystem
reads through. What was derived (beliefs, readings) is deleted; what was *moved* (trust,
interaction counts, closed goals) is reset and replayed from the authored values, because
an increment is not something a delete can undo.

**Token streaming is closed off, and it is worth knowing why.** Every competitor streams
the reply as it is written. This engine checks everything a character says *after* it is
written and before it becomes an event — the secret-keeping guard, `invents_a_fact`,
`_plays_the_player`, the repeat guard — so streaming would put words on screen ahead of the
checks, and retracting them afterwards tells the player anyway. A guard you can read around
is not a guard. What ships instead is `Watched`: the wire carries a key and a job, never
content, and the player sees the room take its turn one agent at a time. Do not "improve"
this into token streaming without first moving those checks somewhere they can run before
the words are shown.

~~**Nobody can share anything.**~~ Done: character cards read and written, `--card` in and
`--cards` out, PNG and JSON, no new dependency. Two things to keep hold of. **The
instructions are dropped on the floor** — `system_prompt`, `post_history_instructions` and
`character_book` are absent from `cards.KEPT`, because a card is a file off the internet
and those fields exist to reach a model as instructions; the absence *is* the security
property, so do not add them back for fidelity. And an imported card costs **no model
call**: it becomes one room with the two of you in it, which means no facts and therefore
no secrets. That is shallow on purpose. Generating a world around somebody else's character
would put words in their mouth before the player had met them.

**The player can steer.** A standing note — `/steer`, or the button under the composer —
read by the situation writer and the narrator and by nothing else. The line that matters:
it never reaches a character's prompt, because a note that steered what Maria *said* would
be the player operating an agent that is supposed to be somebody else, and that is the
whole thing this engine protects. So it shapes what happens and how it is described, not
what people say; do not "improve" it by threading it into `generate_utterance`. It needs no
guard of its own, because the narration it influences is still checked by `invents_a_fact`
afterwards.

~~**And nothing ever happens anywhere else.**~~ The thing this design makes possible that
one model puppeting a cast cannot do — somebody acting in another room while you are not
there — was built to the last piece and never ran once. Measured across fourteen player
lines in two worlds: zero time skips, zero off-screen actions, eleven authored intentions
waiting. `advance_time` was reachable only from a button the player had no reason to
press, and nothing anywhere sent a character out of the room, so the cast converged on the
player in the first turn and every intention became unreachable — an intention happens
*somewhere*, and they were all here.

Two rules came out of it that are worth more than the feature:

**A mechanism with no trigger is not a feature, it is a fixture.** Every piece worked in
isolation and had tests. What nothing tested was whether the ordinary path through the
engine ever reached them. Before building the next deep thing, play fourteen lines and
count how many times it fired.

**The cast converging on the player is the engine's default and it is wrong.** Pressures
fetch people *to* you; nothing sends anybody away. That shows up as rooms that empty of
meaning, maps nobody uses, and a world that is really one room with a backdrop. Anything
that gives characters somewhere else to be is worth more than another mechanic in the room
you are already in.

Measured on a 3B once it ran, and the payoff is real: Maria leaves, you perceive it only as
"a door somewhere near the study", she finishes the letters where you are not, and when you
walk into the study the room shows you the stack before she mentions it — and then she
mentions it unprompted. The perception grading that was built for secrets turns out to be
what makes *this* good: you are not told she left, you hear a door.

## Not a game

Scenes, end conditions and an onward button are game furniture. Some stories are better
for them and plenty are not: meeting a childhood friend after twenty years is not
something you complete, and a school year is somewhere you stay. So a story can be played
**open-ended** — chosen by the player where they pick it, because the same house can be
one you finish or one you live in, and that is not the author's call.

The half that matters is not hiding the endings, it is that **the engine writes what
happens next**. Pressures have `max_fires`; measured on `ashgrove/the_dinner`, all three
are spent by the seventh player line and every turn after that is identically two people
talking with a narration between them. Nothing reports it, and no amount of authoring
fixes it, because the twentieth complication is the one nobody wrote.

Two things to keep hold of when extending this. An invented situation is an *ordinary
`Pressure`* — not a new event kind and not a second path through perception — which is why
the feature is one function rather than a fork in the director. And the writer is told
less than the director is allowed to know: never the facts, because the deterministic
guard downstream can only catch a secret's literal keywords and a writer that had been
told could paraphrase around them.

The instinct to reach for next, and to be suspicious of: giving this more *narrative*
authority — arcs it plans, characters it introduces, secrets it invents. Every one of
those is a claim the engine cannot check, and the reason this one is safe is that its
whole output is one sentence of narration in a room that already exists.

**The client is no longer only prose.** A world has a cover and everybody in it has a
picture — an `image:` pointing at a file when somebody drew one, and a plate drawn from the
id when nobody did, which is every generated world. That ordering is the point: art is the
one thing an author cannot write in YAML, so the generated version is the default and the
file is the override, and no screen is ever an empty frame waiting for something that does
not exist. Rooms deliberately have none: worlds and characters are what a person browsing
a shelf and sitting in a room actually needs to see, and a picture per room is a third
asset type, a POV question, and a reason to redraw the view every time somebody walks
through a door. Asked and settled — do not add it back as a gap.

The first dent: **the character you play is yours.** A name and a line about how
you come across, chosen where you pick the story, with the name substituted through every
piece of authored prose that used the old one and the line becoming something the room
actually perceives. Everyone *else* is still somebody the author wrote — a cast you can
name yourself is still ahead.
