"""Two models: one that writes, one that files.

The question this answers is not "can we configure a second model" but
"which second model is worth configuring". Measured on `ashgrove`, three
player lines cost 35 model calls, and 24 of them are the per-memory
readings, the summaries, the classifications and the director's beat —
none of which anybody ever reads. The character lines and the narration
are the eleven somebody is actually paying for.

So the seam is what a call is *for*, not who makes it. Splitting by
"director versus characters" would have moved one call in thirty-five,
because a character makes both the expensive call and most of the cheap
ones.

The keys already say which job it is, so this is a client that dispatches
on them rather than an argument threaded through nine call sites.
"""
import tempfile
from pathlib import Path

import pytest

from fabula.llm import FakeLLM, LiteLLMClient, Routed, is_filing
from fabula.session import Session
from fabula.settings import Settings

from tests.conftest import ASHGROVE


class Named(FakeLLM):
    """A fake that says which of the two it is, and remembers what it was
    asked for."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.keys = []

    def complete(self, system, prompt, key=None):
        self.keys.append(key)
        return f"[{self.name}] " + super().complete(system, prompt, key)


@pytest.fixture
def pair():
    return Named("writes"), Named("files")


# --- which job is which -------------------------------------------------

def test_the_key_already_says_what_the_call_is_for():
    assert is_filing("interpret:maria")
    assert is_filing("summary:tomas")
    assert is_filing("classify:tomas")
    assert is_filing("beat")

    assert not is_filing("maria"), "a character's line is the product"
    assert not is_filing("__narrator__")
    assert not is_filing("pressure:the_mantel_draws_the_eye")
    assert not is_filing("place:kitchen")
    assert not is_filing(None)


def test_making_a_world_stays_on_the_good_model():
    """A handful of calls once, so there is no bill to cut — and it is the
    one job measured to need the better model: a 1.5B hands back the
    example it was shown, and a 3B only just manages."""
    for key in ("invent:world", "invent:scene", "invent:complications",
                "invent:aftermath", "invent:repair"):
        assert not is_filing(key), key


def test_a_second_model_is_optional_and_absent_by_default(pair):
    writes, _ = pair
    only = Routed(writes)

    assert only.complete(system="s", prompt="p", key="interpret:maria").startswith("[writes]")


# --- what a real turn actually spends -----------------------------------

def test_most_of_a_turn_goes_to_the_model_nobody_reads(pair):
    """The measurement the split exists for. If this ever inverts, the
    feature has stopped paying for itself and should be reconsidered
    rather than kept out of habit.

    *Most*, not any particular multiple. A reading is per character who
    perceived the line, so the ratio is really a measure of how many
    people are in earshot: 1.7 filed per written on `ashgrove` over eight
    lines, 1.2 on `winterlight` with four people in one room, and 3.4 on
    `ashgrove` back when the cast converged on the player and never left
    the room again. Asserting one of those numbers was asserting that
    nobody ever walks out.
    """
    writes, files = pair
    session = Session.open(
        ASHGROVE, "the_dinner", llm=Routed(writes, files),
        db_path=str(Path(tempfile.mkdtemp()) / "story.sqlite"),
    )
    for _ in range(3):
        session.say("Say something into the room.")

    assert len(files.keys) > len(writes.keys), (
        f"{len(files.keys)} filed against {len(writes.keys)} written"
    )
    assert all(is_filing(k) for k in files.keys)
    assert not any(is_filing(k) for k in writes.keys)
    session.close()


def test_the_prose_the_player_reads_comes_from_the_writing_model(pair):
    """What somebody is paying for. Every line that reaches them is the
    good model's, whatever the filing costs."""
    writes, files = pair
    session = Session.open(
        ASHGROVE, "the_dinner", llm=Routed(writes, files),
        db_path=str(Path(tempfile.mkdtemp()) / "story.sqlite"),
    )
    perceived = session.say("Tomás, you have been strange all evening.")

    generated = [
        p for p in perceived
        if p.event.actor_id != session.user_character.id and "[gen:" in p.perceived_content
    ]
    assert generated, "somebody answered"
    assert all("[writes]" in p.perceived_content for p in generated)
    session.close()


# --- and it is still one client ----------------------------------------

def test_nothing_downstream_knows_there_are_two(pair):
    """`LLMClient` is one method, and this satisfies it. The engine, the
    session and every agent go on holding a single client."""
    writes, files = pair
    routed = Routed(writes, files)
    session = Session.open(
        ASHGROVE, "the_dinner", llm=routed,
        db_path=str(Path(tempfile.mkdtemp()) / "story.sqlite"),
    )

    assert session.llm is routed
    assert session.director.llm is routed
    session.close()


def test_a_second_model_cannot_be_told_anything_the_first_could_not(pair):
    """The question that would normally block a change like this. It does
    not, and the reason is structural: the projection is deterministic and
    happens before any model call, so the number of models downstream of
    it cannot matter."""
    writes, files = pair
    session = Session.open(
        ASHGROVE, "the_dinner", llm=Routed(writes, files),
        db_path=str(Path(tempfile.mkdtemp()) / "story.sqlite"),
    )
    session.say("What have you been sitting on all evening?")

    # Maria is in the study with the door not quite shut; the music box is
    # Tomás's. Nothing sent to either model on her behalf may name it.
    for asked in files.calls + writes.calls:
        system, prompt, key = asked
        if key and key.endswith("maria"):
            assert "music box" not in (system + prompt).lower(), key
    session.close()


# --- the settings that name them ---------------------------------------

def test_settings_build_one_client_or_two():
    assert Settings(model="gpt-4o-mini").client().__class__ is LiteLLMClient

    both = Settings(model="gpt-4o-mini", fast_model="openai/qwen2.5-3b").client()

    assert isinstance(both, Routed)
    assert both.writes.model == "gpt-4o-mini"
    assert both.files.model == "openai/qwen2.5-3b"


def test_the_filing_model_inherits_the_endpoint_unless_given_its_own():
    shared = Settings(model="a", api_base="http://localhost:8080/v1", fast_model="b").client()
    assert shared.files.api_base == "http://localhost:8080/v1"

    split = Settings(
        model="a", api_base="http://localhost:8080/v1",
        fast_model="b", fast_api_base="http://localhost:9090/v1",
    ).client()
    assert split.writes.api_base == "http://localhost:8080/v1"
    assert split.files.api_base == "http://localhost:9090/v1"


def test_naming_no_second_model_changes_nothing():
    """The default has to be invisible: somebody who does not care about
    any of this must get exactly what they got before."""
    assert Settings().client() is None
    assert Settings(fast_model="openai/qwen2.5-3b").client() is None, "no first model, no story"


def test_neither_model_can_hold_a_credential():
    """The same rule as the first one, and worth restating because a
    second model is a second text box on a web page."""
    for field in Settings.__dataclass_fields__:
        assert not any(word in field.lower() for word in ("key", "token", "secret", "password"))


def test_a_filing_model_with_nothing_to_file_for_says_so():
    """It would otherwise save, do nothing, and never explain itself: with
    no first model the engine falls back to the environment, which is one
    model for everything."""
    complaint = Settings(fast_model="openai/qwen2.5-3b").unreachable()

    assert complaint and "before the one that files" in complaint
    assert Settings().unreachable() is None, "and an empty panel is not a complaint"
