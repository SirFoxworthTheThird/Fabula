"""Choosing a model and an endpoint from the command line.

Every entry point has to accept the same two flags, or "test it on your
own model" means editing source.
"""
import pytest

from fabula.api import main as serve_main
from fabula.cli import main as cli_main
from fabula.llm import LiteLLMClient
from fabula.measure import main as measure_main
from fabula.playtest import main as playtest_main


def test_the_client_carries_model_and_endpoint():
    client = LiteLLMClient(model="gpt-4.1-nano", api_base="https://example.test/v1")

    assert client.model == "gpt-4.1-nano"
    assert client.api_base == "https://example.test/v1"


def test_an_endpoint_is_only_sent_when_one_was_given(monkeypatch):
    """litellm picks the provider's own base when none is passed; sending
    an empty one would override that with nothing."""
    seen = {}

    class FakeLiteLLM:
        def completion(self, **kwargs):
            seen.update(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setitem(__import__("sys").modules, "litellm", FakeLiteLLM())

    LiteLLMClient(model="m").complete("s", "p")
    assert "api_base" not in seen

    LiteLLMClient(model="m", api_base="https://example.test/v1").complete("s", "p")
    assert seen["api_base"] == "https://example.test/v1"


@pytest.mark.parametrize(
    "entry, argv",
    [
        (cli_main, ["worlds/ashgrove", "the_dinner"]),
        (serve_main, []),
        (playtest_main, ["worlds/ashgrove", "the_dinner"]),
        (measure_main, ["worlds/ashgrove", "the_dinner"]),
    ],
)
def test_every_entry_point_accepts_model_and_api_base(entry, argv, capsys):
    with pytest.raises(SystemExit):
        entry(argv + ["--help"])

    printed = capsys.readouterr().out
    assert "--model" in printed
    assert "--api-base" in printed
