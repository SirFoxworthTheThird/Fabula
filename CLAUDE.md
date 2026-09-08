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

The next thing available here is deferring the readings to the end of a turn instead of
issuing them after each event — in-scene beliefs are only read back by the reveal and by
the guard on a reading itself, so a turn's readings could be one batch rather than eight.
That one changes when a belief becomes durable within a turn, so it is not free.

The rule to carry: a feature that adds a per-character or per-moment model call is
spending somebody's money and somebody's patience. Decide how it will be paid for while
designing it, not afterwards. `--no-interpret` is a symptom of not having done that.

What being a local *application* does demand, and what is missing:

* **Install without a toolchain.** *Running* is now one word — `fabula` starts the
  service and opens the browser client, which is one HTML file with no build step. But
  installing is still `pip install -e .` from a clone, which is a developer's front door.
  What is missing is a way to get the app onto a machine that has no Python on it.
* ~~**A library of your stories.**~~ Done: one SQLite file per story in
  `~/.fabula/stories`, listed, resumed and deleted from the CLI, the API and the web
  start screen. Sessions still live in memory, but a story no longer depends on one.

## What is authored, and the problem with that

Everything: worlds, rooms, facts, characters, pressures, scenes, items — all hand-written
YAML. That is right for the parts a story turns on and wrong as a way to *start* one. The
audience for this app does not write YAML. Any plan that ends with "and then the author
writes the file" has not finished.
