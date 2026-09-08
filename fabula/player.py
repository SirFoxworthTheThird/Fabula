"""The character the player brings.

Everything in a world is authored, and that is right for the parts a
story turns on. It is wrong for the one person the player is: being
handed Rook, or Elena, or Ana Reyes is being handed somebody else's
character to wear, in an app whose whole point is playing a story of
your own.

So a story can be started as somebody you named. Two fields, and each
has to actually reach the fiction or it is decoration:

* **A name.** It replaces the authored one everywhere the author wrote
  it — the other characters' personas, their notes about you, the
  pressures, the room descriptions — because a sister who calls you
  Elena while the screen says Wren is worse than no renaming at all. It
  is a substitution over authored text, so it can rename and nothing
  else: it adds no information and removes none.

* **How you come across.** One line, and it becomes the first thing the
  room perceives about you — an ordinary event in your own room, filtered
  like any other, so the people standing there can react to it and the
  people elsewhere never see it. Which is also why it is asked for as
  what *anyone can see*: a private truth put here would be handed to
  everybody in earshot, which is the one thing this engine exists not to
  do.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A name is substituted into authored YAML before it is parsed, so it
# must not be able to change the *shape* of that YAML. Rejected rather
# than escaped: the set below is punctuation nobody's name needs, and a
# name is not the place to be clever. Everything else — any script, any
# accent — is allowed, because the engine speaks no language of its own.
FORBIDDEN = set("\"\\:{}[]#&*!|>%@`\n\r\t")
NAME_LIMIT = 40
LOOK_LIMIT = 240


@dataclass(frozen=True)
class Player:
    """Who the player is playing. Empty means the authored character."""

    name: str = ""
    look: str = ""

    @property
    def given(self) -> bool:
        return bool(self.called or self.look.strip())

    @property
    def called(self) -> str:
        """The name as it is actually used, everywhere.

        An apostrophe is part of plenty of names and none of anybody's
        business to refuse — but a straight one can close a single-quoted
        YAML scalar, and the name is substituted into authored YAML. The
        typographic one cannot, reads identically, and is what a
        typesetter would have used anyway. Done here, once, so the prose,
        the screen and the saved story all agree on the spelling.
        """
        return self.name.strip().replace("'", "\u2019")

    def complaint(self) -> str | None:
        """Why this cannot be used, or None."""
        name, look = self.called, self.look.strip()
        if len(name) > NAME_LIMIT:
            return f"that name is longer than {NAME_LIMIT} characters"
        if any(character in FORBIDDEN for character in name):
            return "a name cannot contain quotes, colons or brackets"
        if len(look) > LOOK_LIMIT:
            return f"keep it to {LOOK_LIMIT} characters — one line is enough"
        if any(character in "\r\n" for character in look):
            return "one line, not several"
        return None


def rename(text: str, was: str, now: str) -> str:
    """Call the authored character by the player's name instead.

    The full name first, then the first name on its own, so "Ana Reyes"
    and a later bare "Ana" both land — and land on the right halves of
    the new one. Word boundaries, so an id like `ana:` in a relationship
    map is untouched while the prose above it is not.
    """
    was, now = was.strip(), now.strip()
    if not was or not now or was == now:
        return text
    pairs = [(was, now)]
    first_was, first_now = was.split()[0], now.split()[0]
    if first_was != was:
        pairs.append((first_was, first_now))
    for old, new in pairs:
        text = re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", new.replace("\\", ""), text)
    return text
