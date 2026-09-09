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


class ModelUnavailable(RuntimeError):
    """The model did not answer.

    Everything past this boundary is somebody else's machine — a key that
    was never set, a rate limit, a laptop that went to sleep — and none of
    it is a bug in the story. It is caught here and named here so a turn
    can be discarded and offered again instead of taking the session down
    and losing the scene with it.
    """


class LiteLLMClient:
    """Provider-agnostic model calls via litellm.

    `api_base` points at any OpenAI-compatible endpoint — a hosted proxy,
    a local server, an aggregator — in which case the model id usually
    needs an `openai/` prefix so litellm speaks that dialect to it.
    Credentials come from the environment rather than an argument, so
    they stay out of shell history and process listings.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_base: str | None = None):
        self.model = model
        self.api_base = api_base

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        import litellm

        extra = {"api_base": self.api_base} if self.api_base else {}
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                **extra,
            )
        except Exception as failure:
            # Deliberately everything: litellm raises its own hierarchy on
            # top of each provider's, and a player does not need to know
            # which of the two lost. What they need is the scene back.
            raise ModelUnavailable(_one_line(failure)) from failure
        return response["choices"][0]["message"]["content"]


def _one_line(failure: Exception) -> str:
    """The first line of a provider error, which is the part that says
    what went wrong; the rest is a stack from three libraries down."""
    text = str(failure).strip().splitlines()
    return text[0][:300] if text else failure.__class__.__name__


# What a call is *for*, taken from the key every call site already
# passes. Two thirds of a turn is the per-memory reading — one sentence
# of "what does she now think is going on", stored as a belief and read
# back only by the reveal and by the guard on the next reading. Nobody
# sees it. The same goes for the compression that feeds context, the
# classification whose every referent is checked afterwards, and the
# director's beat, which is one id out of a list the engine handed it and
# refuses to accept anything else from.
#
# So this is the seam where a cheaper model belongs, and it is a prefix
# match rather than a new argument threaded through nine call sites,
# because the key already says which job it is.
#
# `invent:` is deliberately not here. A world is a handful of calls once,
# so there is no bill to cut, and it is the one job that was *measured*
# to need the better model: a 1.5B hands back the example it was shown
# and a 3B only just manages. Cheap where it is invisible, not cheap
# where it was hard.
FILING = ("interpret:", "summary:", "classify:", "beat")


def is_filing(key: str | None) -> bool:
    """Is this call bookkeeping, or is it the story?"""
    return bool(key) and key.startswith(FILING)


class Routed:
    """Two models: one that writes, one that files.

    A character's line and the narration are what somebody is paying for.
    The readings, the summaries and the classifications are what the
    engine needs in order to produce them, and on the measured shape of a
    turn they are most of the bill — 23 calls out of 35 on `ashgrove`,
    none of which anybody reads.

    Splitting them is worth doing and worth being careful about, because
    the readings feed the context the *writing* model gets: a cheap model
    filing badly does not show up as bad filing, it shows up as a worse
    scene one turn later. `/reveal` is where to look.

    Nothing here touches perception. The projection is deterministic and
    happens before any of these calls, so a second model cannot be told
    anything the first could not — the invariant does not depend on which
    model, or how many.
    """

    def __init__(self, writes: LLMClient, files: LLMClient | None = None):
        self.writes = writes
        self.files = files or writes

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        return (self.files if is_filing(key) else self.writes).complete(
            system=system, prompt=prompt, key=key
        )


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

    # A fixed sentence for every character was one line of a fake and one
    # thing wrong with it: a real cast does not say the same words in the
    # same order, and pretending otherwise made every generated line look
    # like a paraphrase of every other. The repeat guard, correctly,
    # silenced the entire room. Varied by the digest, so it is still a
    # pure function of the inputs and still invents nothing that was not
    # already a constant here — which is the property the leak tests rest
    # on.
    PLACEHOLDERS = (
        "(a pause)", "(silence)", "(a beat)", "(nothing)", "(quiet)", "(a shrug)",
        "(a breath)", "(stillness)", "(a glance)", "(hesitation)", "(a murmur)",
        "(waiting)",
    )

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        self.calls.append((system, prompt, key))
        if key is not None and key in self.canned:
            return self.canned[key]
        digest = hashlib.sha256(f"{system}\n{prompt}".encode()).hexdigest()[:8]
        placeholder = self.PLACEHOLDERS[int(digest, 16) % len(self.PLACEHOLDERS)]
        return f"{placeholder} [gen:{digest}]"


_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
)


def missing_credentials(model: str, api_base: str | None = None) -> str | None:
    """Why this model cannot be reached from this environment, or None.

    Checked before a story opens rather than at the first model call: the
    old behaviour was a traceback several turns in, with the scene lost
    and a stack from three libraries in place of "no key".
    """
    try:
        import litellm
    except ImportError:  # pragma: no cover - depends on the install
        return "litellm is not installed: pip install -e ."
    verdict = litellm.validate_environment(model=model, api_base=api_base)
    if verdict.get("keys_in_environment"):
        return None
    missing = ", ".join(verdict.get("missing_keys") or []) or "a provider key"
    where = " (a .env file beside you is read too)"
    if api_base:
        # A local server usually ignores the key but the client still
        # insists on one, which is a confusing way to be told to type
        # any string at all.
        return (
            f"{model} at {api_base} needs {missing} in the environment{where}. "
            "A local server generally ignores its value — set it to anything."
        )
    return f"no {missing} in the environment{where}, and {model} needs one."


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
