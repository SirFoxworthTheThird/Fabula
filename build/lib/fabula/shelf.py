"""Where the worlds are, when nobody said.

`worlds_root` used to default to the relative path `worlds`, which is a
directory that exists in a clone of this repository and nowhere else. So
the app worked when you ran it from the checkout and had an empty shelf
everywhere else — and `--invent` wrote a generated world into whatever
directory you happened to be standing in. That is a developer's front
door, and this is an app people are meant to install.

Two places, and the split is the point:

* **Shipped** — the worlds this build carries, inside the package, next
  to the client's one HTML file. Installed by pip, read-only in spirit.
* **Yours** — `~/.fabula/worlds`, beside `~/.fabula/stories`. Writable,
  and where a generated world lands.

The shipped ones are *copied* into yours the first time rather than read
from both places, because a world is a directory of YAML somebody should
be able to open and change: the whole pitch is that this runs on your
machine and the files are yours, and a shelf half of which is inside
site-packages does not mean that. What that costs is upgrades — a fixed
shipped world will not overwrite the copy you have, because by then it is
yours and not ours.

What is recorded is what has been *delivered*, not what is present, so
deleting a world means it stays deleted rather than reappearing on the
next launch, and a world added in a later version still arrives.
"""
from __future__ import annotations

import shutil
from pathlib import Path

SHIPPED = Path(__file__).parent / "worlds"
YOURS = Path.home() / ".fabula" / "worlds"
# One world id per line. A dotfile, so `catalogue` — which walks
# directories holding a world.yaml — never sees it.
DELIVERED = ".delivered"


def shipped_worlds(shipped: Path | None = None) -> list[Path]:
    shipped = SHIPPED if shipped is None else Path(shipped)
    if not shipped.is_dir():
        return []
    return sorted(p for p in shipped.iterdir() if (p / "world.yaml").is_file())


def stocked(yours: Path | None = None, shipped: Path | None = None) -> Path:
    """Your worlds directory, with anything this build ships that has
    never been handed over before.

    Returns the shipped directory instead if yours cannot be written —
    a read-only shelf is a working app, and refusing to start because a
    home directory is odd is not.
    """
    # Read off the module rather than bound as defaults, so both are one
    # name a test or an embedder can move.
    yours = YOURS if yours is None else Path(yours)
    shipped = SHIPPED if shipped is None else Path(shipped)
    try:
        yours.mkdir(parents=True, exist_ok=True)
        record = yours / DELIVERED
        already = set(record.read_text(encoding="utf-8").split()) if record.is_file() else set()
        for world in shipped_worlds(shipped):
            if world.name in already or (yours / world.name).exists():
                continue
            # dirs_exist_ok is deliberately off: this only ever creates a
            # world that is not there, and never writes over one.
            shutil.copytree(world, yours / world.name)
        delivered = already | {world.name for world in shipped_worlds(shipped)}
        record.write_text("\n".join(sorted(delivered)) + "\n", encoding="utf-8")
    except OSError:
        return shipped
    return yours
