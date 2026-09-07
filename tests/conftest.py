from pathlib import Path

import pytest

from fabula.loader import load_scenario

ASHGROVE = Path(__file__).parent.parent / "worlds" / "ashgrove"


@pytest.fixture
def scenario():
    """(world, characters, scene) for the shipped ashgrove/the_dinner scene:
    Tomás and Elena (the user) in the kitchen, Maria in the study, one
    secret (the broken music box) known only to Tomás."""
    return load_scenario(ASHGROVE, "the_dinner")
