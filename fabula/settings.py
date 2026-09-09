"""Which model answers, chosen from inside the app.

The pitch is that the application is local and the model is your choice.
That choice lived in command-line flags, which means it was a choice for
whoever started the process — fine for a developer, useless for somebody
who opened the app and wants to point it at a different model than the
one they started with.

So the choice is a small file next to your stories, and the app can
write it.

**No credential is ever in it.** Not a field, not an accident: keys reach
this project through the environment and nowhere else, and a settings
panel that accepted one would be a browser form posting an API key into
a JSON file on disk. What the panel does instead is *look* — it reports
which provider variables the environment has (names only, never values)
and where a `.env` would be read from, so somebody who is missing one is
told exactly what to set and where, rather than being handed a text box
and a false sense of where their key went.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from fabula.concurrency import DEFAULT_WORKERS
from fabula.env import DEFAULT_ENV_FILE
from fabula.library import DEFAULT_ROOT
from fabula.llm import (
    _KEY_ENV_VARS,
    FakeLLM,
    LiteLLMClient,
    LLMClient,
    Routed,
    missing_credentials,
)

DEFAULT_SETTINGS = DEFAULT_ROOT.parent / "settings.json"


@dataclass
class Settings:
    """What a story opened from now on will run on."""

    # Any id litellm understands. Empty means "whatever the environment
    # implies" — a provider key gives you a sensible default, and no key
    # gives you placeholder text you can still play through.
    model: str = ""
    # An OpenAI-compatible endpoint: a proxy, an aggregator, llama.cpp on
    # this machine.
    api_base: str = ""
    # A second, cheaper model for the calls nobody reads — the per-memory
    # readings, the summaries, the classifications, the director's beat.
    # Measured on `ashgrove`, that is 24 calls out of 35. Empty means
    # "the same model as everything else", which is what it was before
    # this existed, so leaving it alone changes nothing.
    fast_model: str = ""
    # Where that one lives, when it is not where the first one lives —
    # the whole point of the split is that these can be different
    # providers, a hosted model writing and a local one filing.
    fast_api_base: str = ""
    # The per-memory reading. It is most of a scene's model calls, so it
    # is the one knob that visibly changes what a turn costs.
    interpret: bool = True
    # Whether the director chooses which beat the narrator writes, rather
    # than taking them in the order the engine offers. One call per
    # narration, and only when the moment could be more than one thing.
    direct: bool = True
    # How many of a turn's independent calls may be in flight at once.
    workers: int = DEFAULT_WORKERS

    @classmethod
    def load(cls, path: Path | str = DEFAULT_SETTINGS) -> Settings:
        """What is on disk, or the defaults. A settings file that has
        become unreadable is not a reason to refuse to start."""
        try:
            stored = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return cls()
        known = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in stored.items() if k in known})

    def save(self, path: Path | str = DEFAULT_SETTINGS) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def client(self) -> LLMClient | None:
        """The model these settings name, or None for "work it out from
        the environment" — which is what the engine already does.

        A second model named here wraps the first rather than replacing
        it: everything still goes through one `LLMClient`, and which of
        the two answers is decided by what the call is for.
        """
        if not self.model:
            return None
        writes = LiteLLMClient(model=self.model, api_base=self.api_base or None)
        if not self.fast_model:
            return writes
        return Routed(
            writes,
            LiteLLMClient(
                model=self.fast_model,
                api_base=(self.fast_api_base or self.api_base) or None,
            ),
        )

    def unreachable(self) -> str | None:
        """Why this choice will not work from here, or None.

        Both models, because a second one is a second way for a story to
        die several turns in — and the whole reason this is checked
        before a scene opens rather than at the first call is that the
        old behaviour was a traceback with the scene already lost.
        """
        if not self.model:
            # A second model and no first one is a setting that does
            # nothing: the engine falls back to the environment, which is
            # one model for everything, and the box somebody just filled
            # in has no effect they will ever see.
            return (
                "name the model that writes before the one that files — "
                "the second one is for the calls the first would otherwise make"
            ) if self.fast_model else None
        first = missing_credentials(self.model, self.api_base or None)
        if first:
            return first
        if not self.fast_model:
            return None
        second = missing_credentials(
            self.fast_model, (self.fast_api_base or self.api_base) or None
        )
        return f"the second model: {second}" if second else None


def keys_present() -> list[str]:
    """Which provider variables this process can see. Names only — the
    values are never read here, and never leave the process at all."""
    return [name for name in _KEY_ENV_VARS if os.environ.get(name)]


def describe(settings: Settings) -> dict:
    """Everything a settings panel needs, and nothing it must not have."""
    return {
        "model": settings.model,
        "api_base": settings.api_base,
        "fast_model": settings.fast_model,
        "fast_api_base": settings.fast_api_base,
        "interpret": settings.interpret,
        "direct": settings.direct,
        "workers": settings.workers,
        # What a story started right now would actually run on.
        "using": settings.model or ("a model from the environment" if keys_present() else ""),
        # And what the calls nobody reads would run on, when that is
        # something else.
        "filing_with": settings.fast_model if settings.model else "",
        "unreachable": settings.unreachable(),
        "keys_present": keys_present(),
        "env_file": str(Path(DEFAULT_ENV_FILE).resolve()),
        "placeholder": not settings.model and not keys_present(),
    }
