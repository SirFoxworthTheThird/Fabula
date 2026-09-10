"""A face for everybody, and a cover for every story.

The client was text on a page, which is what a story is, and also what a
terminal is. The category this app is in is one people browse with their
eyes: a shelf of covers, a room with faces in it. Being the one that is
all prose is not principled, it is bare.

The obvious version — an `image:` field pointing at a file — solves it
only for a world somebody has drawn art for, which is none of the four
here and none of the ones the generator writes. So the authored image is
the *override*, and what everything falls back to is drawn here: a plate
derived from an id, deterministic, in SVG, with no files, no fetches and
no service in the middle.

Two things make it look deliberate rather than like a broken avatar:

* **A world's cast is tonally related.** A character's hue is the world's
  hue turned a little, so Ashgrove's people look like Ashgrove's people
  and Winterlight's look like Winterlight's. Nothing is authored for this
  — the relation falls out of seeding the character from both ids.

* **No initials.** A letter on a coloured square reads as a placeholder
  for a picture. A figure reads as a plate, and the figure's proportions
  come from the hash too, so a room of four is four different people at
  a glance and not four of the same shape in four colours.
"""
from __future__ import annotations

import base64
import hashlib
import struct
import zlib
from pathlib import Path

# Deliberately not `hash()`: that is salted per process, so the same
# character would be a different colour every time the service restarted.
def _seed(*parts: str) -> list[int]:
    digest = hashlib.sha256("␟".join(parts).encode("utf-8")).digest()
    return list(digest)


def _spread(byte: int, low: float, high: float) -> float:
    return low + (high - low) * (byte / 255)


def _hue(*parts: str) -> float:
    return _seed(*parts)[0] / 255 * 360


def _tone(hue: float, saturation: float, lightness: float) -> str:
    return f"hsl({hue:.0f} {saturation:.0f}% {lightness:.0f}%)"


def cover(world_id: str, title: str = "") -> str:
    """A world's plate: a horizon in its own colour.

    Bands rather than a picture, because a picture of a house that is not
    the house in the story is worse than no picture — it tells you
    something false before the first line. An abstract plate says "this
    is a story, and it is this one" and stops there.

    Drawn wide, at roughly the proportions of the banner it goes in.
    Squarer art in a letterbox is cropped to its middle by `object-fit`,
    which is exactly where the horizon is — measured, on a first pass
    that came out as four rectangles of flat colour.
    """
    grain = _seed("world", world_id)
    hue = _hue("world", world_id)
    warm = (hue + 24) % 360
    # Light enough to read as art under the scrim the title sits in.
    # The first pass was tasteful and invisible: a dark band beside dark
    # type, which is the thing this was meant to fix.
    ground = _tone(hue, _spread(grain[1], 18, 34), _spread(grain[2], 17, 26))
    far = _tone(hue, _spread(grain[3], 22, 40), _spread(grain[4], 34, 45))
    near = _tone(warm, _spread(grain[5], 26, 44), _spread(grain[6], 52, 66))
    glow = _tone(warm, 70, 78)

    # Where the light is, how high the horizon sits and how much the
    # ridges rise, from the same bytes — so two worlds are not the same
    # picture in two colours.
    light_x = _spread(grain[7], 0.15, 0.85)
    horizon = _spread(grain[8], 54, 68)
    ridge = _spread(grain[9], 9, 20)
    swell = _spread(grain[10], 6, 16)

    return _svg(
        'viewBox="0 0 400 100" preserveAspectRatio="xMidYMid slice" role="img"'
        + (f' aria-label="{_plain(title or world_id)}"' if (title or world_id) else ""),
        f'<defs><radialGradient id="sky" cx="{light_x:.2f}" cy="{horizon / 100:.2f}" r="0.75">'
        f'<stop offset="0" stop-color="{glow}" stop-opacity="0.62"/>'
        f'<stop offset="1" stop-color="{ground}" stop-opacity="0"/>'
        f"</radialGradient></defs>"
        f'<rect width="400" height="100" fill="{ground}"/>'
        f'<rect width="400" height="100" fill="url(#sky)"/>'
        # Two ridges, each a curve rather than a line, so it reads as a
        # place seen from somewhere.
        f'<path d="M0 {horizon:.0f} C 100 {horizon - ridge:.0f} 180 {horizon + 4:.0f}'
        f' 260 {horizon - ridge / 2:.0f} S 360 {horizon - ridge:.0f} 400 {horizon - 2:.0f}'
        f' V100 H0 Z" fill="{far}" opacity="0.9"/>'
        f'<path d="M0 {horizon + swell + 6:.0f} C 120 {horizon + swell:.0f}'
        f' 240 {horizon + swell + 12:.0f} 400 {horizon + swell + 2:.0f}'
        f' V100 H0 Z" fill="{near}" opacity="0.75"/>'
    )


def _figure(world_id: str, character_id: str, among: tuple[int, int] | None) -> dict:
    """The plate's numbers, before anybody decides how to draw them.

    Shared by the SVG the app shows and the raster a character card
    carries, so somebody's picture is the same picture wherever it turns
    up.
    """
    grain = _seed("face", world_id, character_id)
    place, cast = among or (0, 1)
    # A band of the wheel starting well clear of the world's own hue, so
    # nobody is the colour of the sky behind them, walked evenly and then
    # nudged a little so a cast does not look like a paint chart.
    step = 300 / max(cast, 1)
    hue = (
        _hue("world", world_id) + 40 + step * (place % max(cast, 1))
        + _spread(grain[0], -step / 5, step / 5)
    ) % 360
    return {
        "back": (hue, _spread(grain[1], 14, 30), _spread(grain[2], 22, 34)),
        "figure": (hue, _spread(grain[3], 20, 40), _spread(grain[4], 58, 74)),
        "rim": ((hue + 30) % 360, 45.0, 80.0),
        "head": _spread(grain[5], 15, 19),
        "neck": _spread(grain[6], 44, 50),
        "shoulders": _spread(grain[7], 30, 40),
        "lean": _spread(grain[8], -5, 5),
        "top": _spread(grain[9], 30, 36),
    }


def portrait(
    world_id: str,
    character_id: str,
    name: str = "",
    among: tuple[int, int] | None = None,
) -> str:
    """One person's plate: a figure, in a colour out of the world's.

    Not a face. A face nobody wrote is a claim about somebody's
    appearance, and this app already asks the player for their own line
    about how they come across rather than inventing one for them. A
    figure is the honest amount to say.

    `among` is this character's place in the world's cast, and it is the
    difference between a room of people and a room of one person in five
    shades of the same colour. Hashing each hue independently clumps —
    measured, on `winterlight`, which came out as five pinks. Spacing
    them around a band instead guarantees the separation the whole point
    of the plate depends on: telling at a glance who just spoke.
    """
    plate = _figure(world_id, character_id, among)
    back, figure, rim = (_tone(*plate[part]) for part in ("back", "figure", "rim"))
    head, neck = plate["head"], plate["neck"]
    shoulders, lean, top = plate["shoulders"], plate["lean"], plate["top"]

    return _svg(
        'viewBox="0 0 100 100" role="img"'
        + (f' aria-label="{_plain(name or character_id)}"' if (name or character_id) else ""),
        f'<defs><radialGradient id="lit" cx="0.4" cy="0.3" r="0.8">'
        f'<stop offset="0" stop-color="{rim}" stop-opacity="0.35"/>'
        f'<stop offset="1" stop-color="{back}" stop-opacity="0"/>'
        f"</radialGradient></defs>"
        f'<rect width="100" height="100" fill="{back}"/>'
        f'<rect width="100" height="100" fill="url(#lit)"/>'
        f'<g transform="translate({lean:.1f} 0)">'
        f'<circle cx="50" cy="{top:.0f}" r="{head:.1f}" fill="{figure}"/>'
        f'<path d="M{50 - shoulders:.0f} 100 Q 50 {neck:.0f} {50 + shoulders:.0f} 100 Z"'
        f' fill="{figure}"/>'
        f"</g>"
    )


def _rgb(hue: float, saturation: float, lightness: float) -> tuple[int, int, int]:
    import colorsys

    red, green, blue = colorsys.hls_to_rgb(
        (hue % 360) / 360, lightness / 100, saturation / 100
    )
    return round(red * 255), round(green * 255), round(blue * 255)


def raster(
    world_id: str,
    character_id: str,
    among: tuple[int, int] | None = None,
    size: int = 256,
) -> bytes:
    """The same plate, as pixels, for a card that has to be a PNG.

    Drawn by evaluating the two shapes per pixel rather than by pulling
    in an imaging library: the figure is a circle and a filled quadratic,
    both of which answer "is this point inside you" in one line, and a
    dependency for one picture would be a poor trade in an app whose
    install story is already the weak part.
    """
    plate = _figure(world_id, character_id, among)
    back, figure, rim = (_rgb(*plate[part]) for part in ("back", "figure", "rim"))
    head, neck = plate["head"], plate["neck"]
    shoulders, lean, top = plate["shoulders"], plate["lean"], plate["top"]

    rows = bytearray()
    for row in range(size):
        rows.append(0)  # no filter on this scanline
        y = row * 100 / size
        for column in range(size):
            x = column * 100 / size - lean
            # The light, falling where the SVG's gradient puts it.
            glow = max(
                0.0, 1.0 - (((x - 40) ** 2 + (y - 30) ** 2) ** 0.5) / 80
            ) * 0.35
            pixel = tuple(
                round(base + (lit - base) * glow) for base, lit in zip(back, rim)
            )
            if (x - 50) ** 2 + (y - top) ** 2 <= head * head:
                pixel = figure
            elif abs(x - 50) <= shoulders:
                # The shoulder, which is the SVG's quadratic solved
                # rather than guessed at. For a Bézier from (50-w, 100)
                # through the control point (50, neck) to (50+w, 100),
                # x is linear in t — x = 50 + w(2t - 1) — so the curve's
                # height at a column is one substitution, and the filled
                # path is everything below it.
                t = (x - 50 + shoulders) / (2 * shoulders)
                edge = 100 - 2 * t * (1 - t) * (100 - neck)
                if y >= edge:
                    pixel = figure
            rows += bytes(pixel)
    return _png(size, size, bytes(rows))


def _chunk(kind: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body)) + kind + body
        + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    )


def _png(width: int, height: int, rows: bytes, text: dict[str, bytes] | None = None) -> bytes:
    """A PNG, by hand. Colour type 2 is plain RGB with no palette and no
    alpha, which is all a portrait plate needs."""
    out = [b"\x89PNG\r\n\x1a\n"]
    out.append(_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    for keyword, value in (text or {}).items():
        out.append(_chunk(b"tEXt", keyword.encode("latin-1") + b"\x00" + value))
    out.append(_chunk(b"IDAT", zlib.compress(rows, 9)))
    out.append(_chunk(b"IEND", b""))
    return b"".join(out)


def carded(
    world_id: str,
    character_id: str,
    payload: bytes,
    among: tuple[int, int] | None = None,
    size: int = 256,
) -> bytes:
    """This character's plate, with a character card inside it.

    Which is what the rest of the shelf trades in: the picture is the
    file, and the description rides along in a text chunk that every
    reader in the category knows to look for.
    """
    plate = raster(world_id, character_id, among, size)
    body = plate[8:]
    # Slot the card in straight after the header, where every reader
    # expects to find it and before any pixel data.
    header_end = 8 + 25  # signature + IHDR length/type/body/crc
    encoded = base64.b64encode(payload)
    return (
        plate[:header_end]
        + _chunk(b"tEXt", b"chara\x00" + encoded)
        + _chunk(b"tEXt", b"ccv3\x00" + encoded)
        + plate[header_end:]
    )


def _svg(attributes: str, body: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" {attributes}>{body}</svg>'


def _plain(text: str) -> str:
    """A label safe to put in an attribute.

    A character's name can be one the *player* typed, so this is not
    decoration: it is the same rule as everywhere else that model or user
    text reaches a document.
    """
    return (
        str(text)
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")[:80]
    )


# What an authored `image:` is allowed to be. Not a whitelist for its own
# sake: the browser has to be told a type, and a world directory is a
# thing people copy between machines, so "whatever is on disk, served
# with whatever type is guessed" is a worse deal than it looks.
KINDS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".avif": "image/avif",
    ".svg": "image/svg+xml",
}


def authored(world_dir: Path, named: str) -> tuple[Path, str] | None:
    """The file an `image:` names, if it is one this may serve.

    A world directory is content, not code, and it arrives from
    elsewhere: generated here, copied off another machine, downloaded
    from somebody. So the path in it is treated the way any other path
    from outside would be — resolved, and required to still be *inside*
    the world afterwards. `image: ../../../etc/passwd` is the reason this
    function exists rather than a `world_dir / named`.
    """
    named = (named or "").strip()
    if not named:
        return None
    root = Path(world_dir).resolve()
    try:
        found = (root / named).resolve()
        # `is_relative_to` rather than a string prefix: `/worlds/ash` is
        # a string prefix of `/worlds/ashgrove_other`.
        if not found.is_relative_to(root) or not found.is_file():
            return None
    except OSError:
        return None
    kind = KINDS.get(found.suffix.lower())
    return (found, kind) if kind else None
