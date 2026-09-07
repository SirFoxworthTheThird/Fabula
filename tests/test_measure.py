"""The measurement tool for prompt-level rules."""
import fabula.agents as agents
from fabula.measure import VARIANTS, measure_guard

from tests.conftest import ASHGROVE


def test_it_reports_a_count_per_variant(fake_llm):
    results = measure_guard(ASHGROVE, "the_dinner", "tomas", samples=2, llm=fake_llm)

    assert set(results) == set(VARIANTS)
    # FakeLLM invents nothing, so it can never name the guarded subject.
    assert all(count == 0 for count in results.values())


def test_it_puts_the_shipped_guard_line_back(fake_llm):
    """It swaps the builder to compare variants; leaving one installed
    would silently change the engine for everything after."""
    shipped = agents._guarded_subjects

    measure_guard(ASHGROVE, "the_dinner", "tomas", samples=1, llm=fake_llm)

    assert agents._guarded_subjects is shipped


def test_it_restores_the_guard_line_even_when_a_run_fails(fake_llm, monkeypatch):
    shipped = agents._guarded_subjects

    def explode(*_args, **_kwargs):
        raise RuntimeError("provider is down")

    monkeypatch.setattr(agents, "generate_utterance", explode)

    try:
        measure_guard(ASHGROVE, "the_dinner", "tomas", samples=1, llm=fake_llm)
    except RuntimeError:
        pass

    assert agents._guarded_subjects is shipped


def test_the_variants_differ_in_what_they_tell_the_character(scenario, store, fake_llm):
    from fabula.director import Director
    from fabula.narrator import Narrator

    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(fake_llm), fake_llm)
    tomas = characters["tomas"]

    none_ = VARIANTS["no guard line"](tomas, director.contexts, [])
    unnamed = VARIANTS["guarded, unnamed"](tomas, director.contexts, [])
    shipped = agents._guarded_subjects(tomas, director.contexts, [])

    assert none_ == ""
    assert "music box" not in unnamed.lower()  # the point of that variant
    assert "music box" in shipped.lower()
