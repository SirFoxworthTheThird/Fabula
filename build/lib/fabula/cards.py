"""Character cards, in and out.

The category trades in cards: a PNG with a character's description
hidden in a text chunk, read by SillyTavern, Chub, Risu and most of the
rest. A world here is YAML that only this app understands, which means
nothing anybody makes with it can be given to anybody else, and nothing
made anywhere else can be played here. That is the whole sharing story,
and it was missing.

Two directions, and they are not symmetrical.

**Out** is easy and safe. Everybody in a world becomes a card, with the
plate this app already draws for them as the picture, so a world made
here can be handed to somebody who has never heard of it.

**In** is where the care goes, because a card is a file off the
internet. Three rules, and the third is the one that matters:

* **Size and shape.** Capped before it is read, and every field taken by
  name with a type check. A card is a dictionary of strings, not a
  document format.
* **It cannot reshape the world.** Card prose becomes authored YAML, so
  the same characters `fabula.player` refuses in a name are stripped
  here — a description is not the place to be clever.
* **The instructions are dropped on the floor.** A v2 card can carry
  `system_prompt` and `post_history_instructions`: text whose entire
  purpose is to be given to the model as *instructions*. Honouring them
  would mean anybody who can get you to open a file can rewrite how this
  engine behaves — what the narrator may do, what a character may say,
  the rules that keep a secret a secret. So the import takes the
  character and never their instructions. A `character_book` is dropped
  for the same reason: it is a mechanism for injecting text into a
  prompt on a keyword, which is a description of the attack.

What is kept is what a character *is*: their name, what they are like,
the situation, and the first thing they say.
"""
from __future__ import annotations

import base64
import json
import re
import struct
import zlib

from fabula.player import FORBIDDEN

# A card is a few kilobytes of prose. These are generous by a wide margin
# and still small enough that nothing here can be handed a disk image.
MOST_BYTES = 8 * 1024 * 1024
MOST_JSON = 512 * 1024
MOST_FIELD = 8 * 1024

# The chunk keywords the ecosystem uses, in the order they are preferred.
KEYWORDS = ("ccv3", "chara")

# What a card is allowed to tell us. Deliberately not the whole v2 spec:
# `system_prompt`, `post_history_instructions` and `character_book` are
# absent on purpose and their absence is the security property.
KEPT = ("name", "description", "personality", "scenario", "first_mes", "creator")


class NotACard(ValueError):
    """This file is not a character card, or not one that can be read."""


def read(data: bytes) -> dict:
    """The character in a card file, whatever shape it arrived in."""
    if len(data) > MOST_BYTES:
        raise NotACard("that file is far too large to be a character card")
    text = _embedded(data) if data[:8] == b"\x89PNG\r\n\x1a\n" else data
    try:
        found = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise NotACard("there is no character card in that file")
    if not isinstance(found, dict):
        raise NotACard("there is no character card in that file")
    # v2 nests the character under `data`; v1 is the flat object itself.
    inner = found.get("data") if isinstance(found.get("data"), dict) else found
    card = {field: _prose(inner.get(field)) for field in KEPT}
    if not card["name"]:
        raise NotACard("that card does not say who it is")
    return card


def _embedded(data: bytes) -> bytes:
    """The card hidden in a PNG's text chunks.

    Walked by hand rather than with an image library, because reading a
    few chunk headers is a dozen lines and a dependency is forever — and
    because nothing here decodes any pixels, which is the part of an
    untrusted PNG worth being nervous about.
    """
    found: dict[str, bytes] = {}
    at = 8
    while at + 8 <= len(data):
        (length,) = struct.unpack(">I", data[at : at + 4])
        kind = data[at + 4 : at + 8]
        if length > MOST_BYTES:
            raise NotACard("that PNG has a chunk too large to be a character card")
        body = data[at + 8 : at + 8 + length]
        at += 12 + length  # length, type, body, crc
        if kind not in (b"tEXt", b"zTXt"):
            if kind == b"IEND":
                break
            continue
        keyword, _, rest = body.partition(b"\x00")
        name = keyword.decode("latin-1", "replace").lower()
        if name not in KEYWORDS:
            continue
        if kind == b"zTXt":
            # One leading byte says which compression; only deflate is
            # defined, and a bomb is capped by the same limit as the rest.
            try:
                rest = zlib.decompressobj().decompress(rest[1:], MOST_JSON)
            except zlib.error:
                continue
        found[name] = rest
    for name in KEYWORDS:
        if name in found:
            try:
                return base64.b64decode(found[name], validate=False)
            except Exception:
                raise NotACard("the card in that PNG could not be decoded")
    raise NotACard("that PNG has no character card in it")


def _prose(value) -> str:
    """One field, as text this app can safely author with.

    Cards come from strangers and their prose ends up inside YAML that is
    parsed and inside prompts that are read. Anything that is not a
    string becomes nothing; the punctuation that could reshape a document
    is removed rather than escaped, exactly as it is for a name the
    player types.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return ""
    # Placeholders the ecosystem substitutes at runtime, removed *before*
    # the punctuation is — the braces are themselves forbidden, so
    # stripping first would leave the bare words "user" and "char" sitting
    # in the prose. This engine has its own idea of who the player is and
    # it is not something a downloaded file gets to name.
    cleaned = re.sub(r"\{\{\s*[^}]{0,40}\}\}", " ", value[:MOST_FIELD])
    cleaned = "".join(" " if c in FORBIDDEN else c for c in cleaned)
    return " ".join(cleaned.split())


def as_card(world, character, scene=None, cast=None) -> dict:
    """One of this world's people, as a card the rest of the shelf reads.

    What travels is a person: who they are, what they are like, and the
    situation. What does not is the half that makes this engine what it
    is — what they have perceived, who they no longer believe, and the
    fact that everybody else in the room is a separate agent who knows
    something different.

    One thing travels that is worth a warning rather than a silence. A
    persona here may name the secret its *own* character is keeping —
    `inspect` allows exactly that and nothing else — so Tomás's card says
    he broke the music box, because his persona does. That is right for a
    card, which is read by an application that needs to play him, and
    wrong for a person who was hoping to find out. The note on the card
    says so.
    """
    others = [
        person.name
        for person in (cast or {}).values()
        if person.id != character.id and not person.is_user
    ]
    scenario = " ".join(
        part for part in (world.blurb, scene.premise if scene else "") if part
    )
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": character.name,
            "description": character.persona,
            "personality": "",
            "scenario": scenario,
            "first_mes": (scene.opening if scene else "") or "",
            "mes_example": "",
            "creator_notes": (
                f"From {world.name}, a Fabula world. "
                + (f"Also in it: {', '.join(others)}. " if others else "")
                + "In Fabula each character is a separate agent who knows only what "
                  "they perceived; a card cannot carry that, so this is the person "
                  "rather than the story. Spoiler warning: a description here may "
                  "name what this character is keeping to themselves, because that "
                  "is what they know."
            ),
            "system_prompt": "",
            "post_history_instructions": "",
            "alternate_greetings": [],
            "tags": ["fabula"],
            "creator": "",
            "character_version": "",
        },
    }


def png(world, character, scene=None, cast=None, size: int = 256) -> bytes:
    """That card, in the picture, which is how the category trades them."""
    from fabula.art import carded

    order = list(cast or {})
    among = (order.index(character.id), len(order)) if character.id in order else None
    payload = json.dumps(as_card(world, character, scene, cast)).encode("utf-8")
    return carded(world.id, character.id, payload, among=among, size=size)


def as_world(card: dict, worlds_root, player_name: str = "") -> tuple:
    """A card, as somewhere it can actually be played.

    Deliberately without a model. Every other route into a world here
    generates one, and generating around a card would put words in a
    stranger's character's mouth before the player had met them. What a
    card describes is a person and a situation, and the smallest honest
    world for that is one room with the two of you in it — which is also
    exactly the shape the applications this card came from offer.

    No facts, therefore no secrets: this app's whole mechanism is a thing
    one character knows and another does not, and a card does not say
    what that would be. The world is playable and shallow, and the
    honest place to deepen it is the YAML it just became.
    """
    from pathlib import Path

    import yaml

    from fabula.discovery import slug

    name = card["name"]
    world_id = slug(name)[:40] or "someone"
    world_dir = Path(worlds_root) / world_id
    for suffix in range(2, 40):
        if not world_dir.exists():
            break
        world_dir = Path(worlds_root) / f"{world_id}_{suffix}"
    (world_dir / "characters").mkdir(parents=True, exist_ok=True)
    (world_dir / "scenes").mkdir(parents=True, exist_ok=True)

    them = slug(name)[:40] or "them"
    you = "you" if them != "you" else "yourself"
    persona = " ".join(
        part for part in (card["description"], card["personality"]) if part
    ) or f"{name} is here."

    def save(path, data, note=""):
        header = f"# {note}\n" if note else ""
        path.write_text(
            header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    save(
        world_dir / "world.yaml",
        {
            "id": world_dir.name,
            "title": name,
            "blurb": card["scenario"] or f"A character card: {name}.",
            "rooms": {"here": {"name": "here", "description": "", "adjacent": {}}},
            "facts": {},
        },
        f"Imported from a character card{' by ' + card['creator'] if card['creator'] else ''}. "
        "Edit it like anything else.",
    )
    save(
        world_dir / "characters" / f"{them}.yaml",
        {
            "id": them, "name": name, "persona": persona,
            "traits": {"talkativeness": 0.7, "reticence": 0.2},
            "protects": [], "intentions": [],
            "relationships": {you: {"affinity": 0.4, "trust": 0.5}},
            "location_id": "here", "is_user": False,
        },
    )
    save(
        world_dir / "characters" / f"{you}.yaml",
        {
            "id": you, "name": player_name.strip() or "You", "persona": "",
            "traits": {"talkativeness": 0.6, "reticence": 0.3},
            "protects": [], "intentions": [],
            "relationships": {them: {"affinity": 0.4, "trust": 0.5}},
            "location_id": "here", "is_user": True,
        },
    )
    save(
        world_dir / "scenes" / "meeting.yaml",
        {
            "id": "meeting", "world": world_dir.name, "title": name,
            "premise": card["scenario"] or f"You and {name}.",
            "opening": card["first_mes"],
            "mode": "sandbox", "cast": [you, them],
            "starting_positions": {you: "here", them: "here"},
            "turn_budget": 6, "max_consecutive_agent_turns": 3,
        },
    )
    return world_dir, "meeting"
