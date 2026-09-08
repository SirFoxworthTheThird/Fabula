"""Loads worlds, characters, and scenes from the plain-YAML authoring layout
described in spec §11:

    worlds/<name>/world.yaml
    worlds/<name>/characters/*.yaml
    worlds/<name>/scenes/*.yaml
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from fabula.models import Character, Pressure
from fabula.world import (
    CLIENT_TEMPLATES,
    DEGRADED_TEMPLATES,
    DURATION_TEMPLATES,
    STOPWORDS,
    TIME_SKIP_TEMPLATES,
    TRANSMISSION_TEMPLATE,
    Fact,
    Item,
    Phrasing,
    Room,
    World,
)


class Scene(BaseModel):
    id: str
    world: str
    # What this scene is called and the one line that says why you would
    # play it. Both optional: a world written before this existed still
    # loads, and falls back to its id.
    title: str = ""
    premise: str = ""
    mode: Literal["arc", "sandbox"] = "sandbox"
    cast: list[str]
    # Characters written for this world who are not in the room when it
    # opens, and who an authored pressure may bring on. Declared rather
    # than inferred from the pressures, so a typo in a pressure's actor
    # cannot quietly conjure somebody.
    may_arrive: list[str] = Field(default_factory=list)
    starting_positions: dict[str, str] = Field(default_factory=dict)
    start_time: datetime = datetime(2024, 1, 1, 19, 0, 0)
    turn_budget: int = 8
    max_consecutive_agent_turns: int = 4
    # What has to become true for the scene to be over. Same condition
    # vocabulary as a pressure trigger — one language for everything an
    # author declares about scene state. Empty means the scene just runs.
    end_condition: dict = Field(default_factory=dict)
    # Where the story goes from here. Each entry is `{scene: <id>, when:
    # <condition>}`, tried in order, first match winning; an entry with no
    # `when` is the fallback. Empty means the story ends here.
    #
    # This is what turns a menu of scenes into a story. Information
    # asymmetry is what makes the fiction trustworthy; it is not what
    # anybody plays for, and until this existed the engine could play a
    # scene and not a story.
    next: list[dict] = Field(default_factory=list)

    @property
    def name(self) -> str:
        return self.title or self.id.replace("_", " ")


def load_world(world_dir: Path) -> World:
    data = yaml.safe_load((world_dir / "world.yaml").read_text(encoding="utf-8"))
    rooms = {
        room_id: Room(
            id=room_id,
            name=room_data["name"],
            description=room_data.get("description", ""),
            adjacent=room_data.get("adjacent", {}),
        )
        for room_id, room_data in data["rooms"].items()
    }
    facts = {
        fact_id: Fact(id=fact_id, keywords=fact_data.get("keywords", []))
        for fact_id, fact_data in (data.get("facts") or {}).items()
    }
    return World(
        id=data["id"],
        title=data.get("title", ""),
        blurb=data.get("blurb", ""),
        rooms=rooms,
        facts=facts,
        language=data.get("language", "en"),
        discover_rooms=bool(data.get("discover_rooms", False)),
        items={
            item_id: Item(id=item_id, **item_data)
            for item_id, item_data in (data.get("items") or {}).items()
        },
        phrasing=_phrasing(data.get("phrasing") or {}),
    )


def _phrasing(authored: dict) -> Phrasing:
    """Author's wording over the built-in English.

    Merged per key rather than replaced wholesale, so a world can restate
    one line without silently losing the rest to a partial block. A world
    in another language that leaves gaps is an authoring error, caught by
    the world guards in the test suite rather than at load — a half
    translated scene should fail review, not refuse to start.
    """
    return Phrasing(
        degraded={**DEGRADED_TEMPLATES, **(authored.get("degraded") or {})},
        duration={**DURATION_TEMPLATES, **(authored.get("duration") or {})},
        time_skip={**TIME_SKIP_TEMPLATES, **(authored.get("time_skip") or {})},
        transmission=authored.get("transmission") or TRANSMISSION_TEMPLATE,
        client={**CLIENT_TEMPLATES, **(authored.get("client") or {})},
        stopwords=frozenset(authored["stopwords"]) if authored.get("stopwords") else STOPWORDS,
    )


def load_pressures(world_dir: Path) -> list[Pressure]:
    path = world_dir / "pressures.yaml"
    if not path.exists():
        return []
    return [Pressure(**entry) for entry in (yaml.safe_load(path.read_text(encoding="utf-8")) or [])]


def load_characters(world_dir: Path) -> dict[str, Character]:
    characters: dict[str, Character] = {}
    char_dir = world_dir / "characters"
    for path in sorted(char_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        character = Character(**data)
        characters[character.id] = character
    return characters


def load_scene(world_dir: Path, scene_name: str) -> Scene:
    path = world_dir / "scenes" / f"{scene_name}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Scene(**data)


def load_scenario(world_dir: Path, scene_name: str) -> tuple[World, dict[str, Character], Scene]:
    """Load a world, its characters, and one named scene, applying the
    scene's starting-position overrides on top of each character's
    default `location_id`."""
    world = load_world(world_dir)
    characters = load_characters(world_dir)
    scene = load_scene(world_dir, scene_name)

    for character_id, location_id in scene.starting_positions.items():
        if character_id in characters:
            characters[character_id] = characters[character_id].model_copy(
                update={"location_id": location_id}
            )

    return world, characters, scene


def catalogue(worlds_root: Path) -> list[dict]:
    """The shelf: every world, and the scenes a story can start from.

    A scene named as somebody else's `next` is a chapter, not a story —
    starting cold in chapter three is how a menu of scenes reads, and
    people came here to play a story. Derived from what the author wrote
    rather than from a flag, so it cannot fall out of step with it.

    Reading YAML, not opening a scene: no store, no model, nothing
    durable. A world that fails to load is left out rather than taking
    the shelf down with it.
    """
    shelf = []
    for world_dir in sorted(p for p in worlds_root.iterdir() if (p / "world.yaml").is_file()):
        try:
            world = load_world(world_dir)
            characters = load_characters(world_dir)
            scenes = [
                load_scene(world_dir, path.stem)
                for path in sorted((world_dir / "scenes").glob("*.yaml"))
            ]
        except Exception:
            continue
        continues = {
            entry["scene"] for scene in scenes for entry in scene.next if entry.get("scene")
        }
        shelf.append(
            {
                "id": world.id,
                "title": world.name,
                "blurb": world.blurb,
                "language": world.language,
                "scenes": [
                    {
                        "id": scene.id,
                        "title": scene.name,
                        "premise": scene.premise,
                        "opens": scene.id not in continues,
                        "you": next(
                            (
                                characters[cid].name
                                for cid in scene.cast
                                if cid in characters and characters[cid].is_user
                            ),
                            "",
                        ),
                        "with": [
                            characters[cid].name
                            for cid in scene.cast
                            if cid in characters and not characters[cid].is_user
                        ],
                    }
                    for scene in scenes
                ],
            }
        )
    return shelf
