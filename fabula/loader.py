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
from fabula.world import Fact, Room, World


class Scene(BaseModel):
    id: str
    world: str
    mode: Literal["arc", "sandbox"] = "sandbox"
    cast: list[str]
    starting_positions: dict[str, str] = Field(default_factory=dict)
    start_time: datetime = datetime(2024, 1, 1, 19, 0, 0)
    turn_budget: int = 8
    max_consecutive_agent_turns: int = 4
    end_condition: str | None = None


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
    return World(id=data["id"], rooms=rooms, facts=facts)


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
