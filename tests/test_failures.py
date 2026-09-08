"""When the model does not answer.

Everything past the model boundary is somebody else's machine: a key
that was never set, a rate limit, a laptop that went to sleep. None of
it is a bug in the story, and none of it should cost the player their
story — which is what it did. A missing key surfaced as a provider
traceback several turns in, the process died, and the scene died with it.
"""
import threading

import pytest

from fabula.llm import FakeLLM, ModelUnavailable, missing_credentials
from fabula.session import Session

from tests.conftest import ASHGROVE


class FailsAfter(FakeLLM):
    """Answers `n` times, then behaves like a provider that has stopped."""

    def __init__(self, n: int):
        super().__init__()
        self.left = n
        self._lock = threading.Lock()

    def complete(self, system: str, prompt: str, key: str | None = None) -> str:
        with self._lock:
            self.left -= 1
            spent = self.left < 0
        if spent:
            raise ModelUnavailable("Missing credentials")
        return super().complete(system, prompt, key)


def state_of(session: Session) -> tuple:
    return (
        [(e.seq, e.kind, e.actor_id, e.content) for e in session.store.get_events(session.scene.id)],
        {cid: [b.content for b in session.store.get_beliefs(cid)] for cid in session.characters},
        session.turns_played,
    )


@pytest.mark.parametrize("workers", [1, 8])
def test_a_failed_turn_leaves_the_scene_exactly_where_it_was(workers):
    """Half a turn is the worst outcome available: a model that failed on
    the third of five characters would otherwise leave two of them having
    heard something the others never will, permanently, in the file."""
    llm = FailsAfter(60)
    session = Session.open(ASHGROVE, "the_reckoning", llm=llm, workers=workers)
    session.say("Pass the bread.")
    before = state_of(session)

    llm.left = 0  # the provider stops answering
    with pytest.raises(ModelUnavailable):
        session.say("Where is the music box?")

    assert state_of(session) == before


def test_the_story_goes_on_after_the_model_comes_back():
    """A failure is a moment that did not happen, not the end of the
    session — the player retypes and carries on."""
    llm = FailsAfter(60)
    session = Session.open(ASHGROVE, "the_reckoning", llm=llm)
    session.say("Pass the bread.")

    llm.left = 0
    with pytest.raises(ModelUnavailable):
        session.say("Where is the music box?")

    llm.left = 60
    perceived = session.say("Where is the music box?")

    assert any("music box" in p.perceived_content for p in perceived)
    assert session.turns_played == 2, "the take that never happened is not counted"


def test_a_retake_after_a_failure_replays_the_last_real_moment():
    """`/again` means the last moment that actually happened. Offering
    back a take that never landed would replay nothing."""
    llm = FailsAfter(60)
    session = Session.open(ASHGROVE, "the_reckoning", llm=llm)
    session.say("Tomás?")

    llm.left = 0
    with pytest.raises(ModelUnavailable):
        session.say("Where is the music box?")

    llm.left = 60
    again = session.regenerate()

    assert any("Tomás?" in p.perceived_content for p in again)
    assert not any("music box" in p.perceived_content for p in again)


def test_a_failure_in_the_first_turn_of_all_leaves_a_playable_scene():
    """The opening line is already in the log by then, so a turn that
    fails must roll back to it rather than to nothing."""
    # Exactly one call: the scene's own opening line, and nothing after.
    session = Session.open(ASHGROVE, "the_reckoning", llm=FailsAfter(1))
    opening = session.store.get_events(session.scene.id)
    assert [e.kind for e in opening] == ["narration"]

    with pytest.raises(ModelUnavailable):
        session.say("Tomás?")

    assert session.store.get_events(session.scene.id) == opening


def test_a_missing_key_is_reported_before_the_story_opens(monkeypatch):
    """It used to be a provider traceback several turns in, with the
    scene lost and a stack from three libraries in place of "no key"."""
    for var in ("OPENAI_API_KEY", "AZURE_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    complaint = missing_credentials("gpt-4o-mini")

    assert complaint and "OPENAI_API_KEY" in complaint

    monkeypatch.setenv("OPENAI_API_KEY", "anything")
    assert missing_credentials("gpt-4o-mini") is None


def test_pointing_at_a_local_server_says_what_that_needs(monkeypatch):
    """A local server generally ignores the key's value but the client
    still insists on one, which is a confusing way to be told to type any
    string at all."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    complaint = missing_credentials("openai/qwen", "http://127.0.0.1:8080/v1")

    assert complaint and "set it to anything" in complaint


def test_the_cli_refuses_to_start_rather_than_failing_mid_scene(monkeypatch, tmp_path):
    from fabula.cli import main as cli_main

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        cli_main([
            "ashgrove", "the_dinner",
            "--model", "gpt-4o-mini",
            "--worlds", str(ASHGROVE.parent),
            "--library", str(tmp_path / "stories"),
        ])

    # And it refused early enough that no story was saved for a session
    # that could never have played.
    assert not (tmp_path / "stories").exists()


def test_the_service_answers_502_rather_than_a_traceback(tmp_path):
    from fastapi.testclient import TestClient

    from fabula.api import create_app

    llm = FailsAfter(60)
    app = create_app(worlds_root=ASHGROVE.parent, llm=llm, library_root=tmp_path / "stories")
    with TestClient(app, raise_server_exceptions=False) as client:
        opened = client.post("/sessions", json={"world": "ashgrove", "scene": "the_reckoning"})
        session_id = opened.json()["session_id"]
        llm.left = 0
        answer = client.post(f"/sessions/{session_id}/say", json={"text": "Tomás?"})

        assert answer.status_code == 502
        assert "did not answer" in answer.json()["detail"]

        # And the session is still there, on the scene it was on.
        llm.left = 60
        assert client.get(f"/sessions/{session_id}").status_code == 200
        assert client.post(f"/sessions/{session_id}/say", json={"text": "Tomás?"}).status_code == 200


def test_the_terminal_keeps_playing_after_a_failure(monkeypatch, capsys):
    """The player should get their prompt back and a plain sentence, not
    a stack trace and a lost scene."""
    from fabula import cli

    llm = FailsAfter(60)
    session = Session.open(ASHGROVE, "the_reckoning", llm=llm)

    typed = iter(["Tomás?", "__fail__", "Are you still there?", "/quit"])

    def next_line(_prompt: str) -> str:
        line = next(typed)
        llm.left = 0 if line == "__fail__" else 60
        return line

    monkeypatch.setattr("builtins.input", next_line)
    cli.run(session)
    printed = capsys.readouterr().out

    assert "did not answer" in printed
    assert "Traceback" not in printed
    assert "Are you still there?" in [
        p.perceived_content for p in session.perceived_so_far()
    ], "the line after the failure was played"


def test_starting_a_story_when_the_model_is_unreachable_says_so(tmp_path):
    """A scene opens with a line of its own, which is a model call — so
    a wrong key now fails on the first click rather than the first line,
    and that is the click that must not produce a traceback."""
    from fastapi.testclient import TestClient

    from fabula.api import create_app

    app = create_app(
        worlds_root=ASHGROVE.parent, llm=FailsAfter(0), library_root=tmp_path / "stories"
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        refused = client.post("/stories", json={"world": "ashgrove", "scene": "the_dinner"})

        assert refused.status_code == 502
        assert "did not answer" in refused.json()["detail"]
        # And no half-made story was left on the shelf.
        assert client.get("/stories").json() == []
