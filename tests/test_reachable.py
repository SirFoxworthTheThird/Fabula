"""Mechanisms nothing in the ordinary path ever reaches.

Three times in two days the same failure turned up: a piece of machinery
that works, has tests, and that no route through the app arrives at.
Off-screen life, which had no trigger. Items, which had no button outside
the terminal. `departure`, an event kind with a beat, a narrator
instruction and a degraded template that nothing anywhere ever built.

Correctness tests do not catch this, because each piece is correct. What
catches it is asking, of every capability, *how does a player get here* —
and these are the two forms of that question cheap enough to ask on
every run.
"""
import ast
import re
from pathlib import Path

import pytest

from fabula.models import EventKind

ENGINE = Path(__file__).parent.parent / "fabula"
CLIENT = ENGINE / "web" / "index.html"


def source(*names):
    return "\n".join((ENGINE / name).read_text(encoding="utf-8") for name in names)


# --- every kind of event has something that makes one ---------------------

def test_every_event_kind_is_one_the_engine_can_produce():
    """`departure` sat in this list for the whole project with four
    places handling it and nothing building one, so the room you walked
    out of perceived nothing."""
    kinds = set(EventKind.__args__)
    engine = source("director.py", "session.py", "persistence.py")
    built = set(re.findall(r'build_event\(\s*"([a-z_]+)"', engine))

    assert kinds - built == set(), f"nothing ever builds: {sorted(kinds - built)}"


# --- every verb the terminal has, the browser has too ---------------------

TERMINAL_ONLY = {
    # Not story actions: the browser closes the tab and picks a story
    # from the shelf.
    "quit", "exit",
}


def terminal_verbs() -> set[str]:
    return {
        verb.lstrip("/")
        for verb in re.findall(r'"(/[a-z]+)"', (ENGINE / "commands.py").read_text())
    }


def test_every_verb_the_terminal_understands_has_a_way_in_from_the_browser():
    """`/read` was the terminal's and nowhere else for the life of the
    project, so every item in every world was unreachable from the app
    most people use. The check is deliberately crude — a name in the
    client, an endpoint in the service — because the failure it catches
    is not subtle."""
    verbs = terminal_verbs() - TERMINAL_ONLY
    assert verbs, "the parser stopped looking like a parser"
    api = (ENGINE / "api.py").read_text(encoding="utf-8")
    page = CLIENT.read_text(encoding="utf-8")

    # The names the two sides use for the same thing.
    ALSO = {"back": "rewind", "again": "regenerate", "retry": "regenerate",
            "on": "next", "go": "move", "look": "look"}
    missing = []
    for verb in sorted(verbs):
        named = ALSO.get(verb, verb)
        served = f'/{named}"' in api or f'/{named}",' in api
        offered = f'"{named}"' in page or f"{named}(" in page
        if not (served and offered):
            missing.append(f"{verb} (service: {served}, client: {offered})")

    assert not missing, "reachable from the terminal and nowhere else: " + "; ".join(missing)


# --- and nothing imported that nothing calls ------------------------------

def test_the_engine_imports_nothing_it_does_not_use():
    """Not tidiness. `choose_beat` was imported into the director and
    never called, which is what an abandoned path looks like from the
    outside — and the one that replaced it took a different route."""
    unused = []
    for path in sorted(ENGINE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        bound = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name != "*" and alias.name != "annotations":
                        bound.add(alias.asname or alias.name.split(".")[0])
        for name in sorted(bound):
            # Once in the file is the import line and nothing else. Counted
            # over the text rather than the tree so a name used only in a
            # docstring or a string annotation still counts as meant.
            if len(re.findall(rf"\b{re.escape(name)}\b", text)) <= 1:
                unused.append(f"{path.name}: {name}")

    assert not unused, "imported and never used: " + "; ".join(unused)
