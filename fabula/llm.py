"""Model-call boundary. Every component that needs a model (director's
pressure/time logic later, narrator, character agents) goes through an
`LLMClient`, so tests can substitute `FakeLLM` and never touch the network.
"""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        """Generate a completion. `key` is an optional caller-supplied
        identifier (e.g. a character id) that a fake client may use to
        script canned responses; a real client ignores it."""
        ...


class LiteLLMClient:
    """Provider-agnostic model calls via litellm."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        import litellm

        response = litellm.completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response["choices"][0]["message"]["content"]


class FakeLLM:
    """Deterministic, offline stand-in for tests and for running the CLI
    without an API key.

    Output is a pure function of the inputs (or of `key`, when the caller
    scripts a canned response) and never invents content beyond a fixed
    template. This matters for the leak tests: FakeLLM structurally cannot
    reproduce a secret that never appeared in its prompt, so a passing leak
    test proves the projection excluded it rather than the fake happening
    to omit it.
    """

    def __init__(self, canned: dict[str, str] | None = None):
        self.canned: dict[str, str] = canned or {}
        self.calls: list[tuple[str, str, str | None]] = []

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        self.calls.append((system, prompt, key))
        if key is not None and key in self.canned:
            return self.canned[key]
        digest = hashlib.sha256(f"{system}\n{prompt}".encode()).hexdigest()[:8]
        return f"(a considered pause) [gen:{digest}]"


_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
)


def get_default_llm(model: str = "gpt-4o-mini") -> LLMClient:
    """Real litellm client if a provider key is configured, else FakeLLM
    with a warning — lets the CLI run end-to-end in an offline sandbox."""
    if any(os.environ.get(var) for var in _KEY_ENV_VARS):
        return LiteLLMClient(model=model)
    print(
        "fabula: no model API key found in the environment; "
        "running with FakeLLM (deterministic placeholder text).",
        file=sys.stderr,
    )
    return FakeLLM()
