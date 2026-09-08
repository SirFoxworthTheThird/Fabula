"""Read a local `.env` into the process environment.

Credentials reach this project through the environment and nowhere else:
there is deliberately no `--api-key` flag anywhere, nothing writes a key
to disk, and no key is ever passed as an argument where it would show up
in shell history or a process listing. A `.env` file is the one
concession to convenience, and it keeps that property — the file is read
into the environment and the value is never handled by anything else.

Two rules matter more than the parsing:

* An exported variable always wins. A file that silently shadowed a key
  you set yourself would make it impossible to say which credential a
  run actually used.
* Nothing here ever prints, logs, or returns a value. Names only.

No dependency for this. `python-dotenv` is a fine library, but reading
twenty lines of `KEY=value` is not worth handing a third party the file
your API keys live in.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

DEFAULT_ENV_FILE = ".env"


def parse_env(text: str) -> dict[str, str]:
    """`KEY=value` lines, with `export ` prefixes, `#` comments, blank
    lines and surrounding quotes tolerated. Anything else is skipped
    rather than raised on: a malformed line in a dotfile should not stop
    someone playing a scene."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[name] = value
    return values


def _warn_if_readable_by_others(path: Path) -> None:
    """A credentials file the rest of the machine can read is worth one
    line of noise. POSIX only — Windows permissions are not expressible
    in this mode bit, and guessing at them would be worse than silence.
    """
    if os.name != "posix":
        return
    try:
        mode = path.stat().st_mode
    except OSError:
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(
            f"fabula: {path} is readable by other users on this machine; "
            f"consider `chmod 600 {path}`.",
            file=sys.stderr,
        )


def load_env(path: str | Path = DEFAULT_ENV_FILE) -> list[str]:
    """Set anything in the file that is not already in the environment.

    Returns the names it set, sorted — names only, never values, so this
    is safe to print. Missing file is not an error; most runs will not
    have one.
    """
    file = Path(path)
    if not file.is_file():
        return []

    _warn_if_readable_by_others(file)

    applied = []
    for name, value in parse_env(file.read_text(encoding="utf-8")).items():
        # Already exported? That one wins, always.
        if os.environ.get(name):
            continue
        os.environ[name] = value
        applied.append(name)
    return sorted(applied)
