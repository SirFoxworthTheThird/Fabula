from pathlib import Path

import pytest

from fabula.db import EventStore
from fabula.llm import FakeLLM
from fabula.loader import load_scenario
from fabula.memory import ContextBuilder

ASHGROVE = Path(__file__).parent.parent / "worlds" / "ashgrove"


@pytest.fixture
def scenario():
    """(world, characters, scene) for the shipped ashgrove/the_dinner scene:
    Tomás and Elena (the user) in the kitchen, Maria in the study, one
    secret (the broken music box) known only to Tomás."""
    return load_scenario(ASHGROVE, "the_dinner")


@pytest.fixture
def store():
    return EventStore()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def contexts(scenario, store, fake_llm):
    world, characters, scene = scenario
    return ContextBuilder(world, characters, scene.id, store, fake_llm)
