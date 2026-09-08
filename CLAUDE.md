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
2. **It runs locally.** Your machine, your model, your data. No account, no per-message
   billing, nothing leaving the box unless you point it at a hosted endpoint yourself.

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

## What "local" costs

This is a hard engineering constraint, not a deployment note. A turn costs one model call
per agent that bids, plus the narrator, plus one per remembered moment per character:

```
ashgrove   (2 agents, 3 player lines) -> 33 model calls
winterlight (4 agents, 3 player lines) -> 51 model calls
```

On a hosted frontier model that is a rounding error. On a 7B running on somebody's own
GPU at a few seconds a call, fifty-one calls is minutes of silence between one line and
the next — which is not a roleplay app, whatever the transcript looks like afterwards.

Every feature that adds a per-character or per-moment model call is spending the same
budget. `--no-interpret` exists for this reason and is a symptom, not a fix: the cost
model needs to be part of the design of each feature rather than a flag bolted on after.

## What is authored, and the problem with that

Everything: worlds, rooms, facts, characters, pressures, scenes, items — all hand-written
YAML. That is right for the parts a story turns on and wrong as a way to *start* one. The
audience for this app does not write YAML. Any plan that ends with "and then the author
writes the file" has not finished.
