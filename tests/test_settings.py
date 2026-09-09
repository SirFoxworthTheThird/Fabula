"""Choosing the model from inside the app.

"Which model answers is your choice" was true of whoever started the
process and nobody else: it lived in `--model`. A person who opened the
app could not change it, which for the one thing this project claims as
a differentiator is most of the way to not having it.

The rule the panel is built around: **it never takes a key.** Credentials
reach the engine through the environment and nowhere else, so the panel
reports what is there — names, never values — and says where to put one
that is not. A browser form posting an API key into a JSON file would be
a worse place for it than the environment, dressed up as a better one.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fabula.api import create_app
from fabula.llm import FakeLLM, LiteLLMClient
from fabula.settings import Settings, describe, keys_present

from tests.conftest import ASHGROVE

WORLDS = ASHGROVE.parent


@pytest.fixture
def settings_file(tmp_path):
    return tmp_path / "settings.json"


@pytest.fixture
def offline(monkeypatch):
    """Nothing in this file may reach a provider. The point is which
    client the engine ends up holding, not what it says."""
    monkeypatch.setattr(
        LiteLLMClient, "complete", lambda self, system, prompt, key=None: "(placeholder)"
    )


@pytest.fixture
def client(tmp_path, settings_file, monkeypatch, offline):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    app = create_app(
        worlds_root=WORLDS,
        library_root=tmp_path / "stories",
        settings_path=settings_file,
    )
    with TestClient(app) as test_client:
        yield test_client


# --- The file ---------------------------------------------------------


def test_settings_survive_a_restart(settings_file):
    Settings(model="gpt-4o-mini", api_base="http://localhost:8080/v1", workers=3).save(
        settings_file
    )

    assert Settings.load(settings_file).model == "gpt-4o-mini"
    assert Settings.load(settings_file).workers == 3


def test_an_unreadable_settings_file_is_not_a_reason_to_refuse_to_start(settings_file):
    settings_file.write_text("{ this is not json")

    assert Settings.load(settings_file) == Settings()


def test_a_field_this_version_does_not_know_is_ignored(settings_file):
    settings_file.write_text(json.dumps({"model": "gpt-4o-mini", "from_the_future": 1}))

    assert Settings.load(settings_file).model == "gpt-4o-mini"


def test_the_settings_have_nowhere_to_put_a_key():
    """Not an omission — the shape of the thing. Anything that looks like
    a credential field here would be a credential on disk."""
    import dataclasses

    for field in dataclasses.fields(Settings):
        assert "key" not in field.name
        assert "token" not in field.name
        assert "secret" not in field.name


def test_what_the_panel_is_told_about_keys_is_names_only(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-the-actual-secret")

    reported = json.dumps(describe(Settings(model="gpt-4o-mini")))

    assert "OPENAI_API_KEY" in reported
    assert "sk-the-actual-secret" not in reported
    assert keys_present() == ["OPENAI_API_KEY"]


# --- Over the wire ----------------------------------------------------


def test_the_panel_says_what_a_story_would_run_on(client):
    current = client.get("/settings").json()

    assert current["keys_present"] == ["OPENAI_API_KEY"]
    assert current["placeholder"] is False
    assert current["pinned"] is False


def test_choosing_a_model_the_environment_cannot_reach_is_refused(client, monkeypatch, settings_file):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    refused = client.put("/settings", json={"model": "gpt-4o-mini"})

    assert refused.status_code == 400
    assert "OPENAI_API_KEY" in refused.json()["detail"]
    assert not settings_file.exists(), "and nothing was saved"


def test_a_saved_choice_is_on_disk_and_has_no_credential_in_it(client, settings_file):
    saved = client.put(
        "/settings",
        json={"model": "gpt-4o-mini", "api_base": "", "interpret": False, "workers": 2},
    )

    assert saved.status_code == 200
    stored = json.loads(settings_file.read_text())
    assert stored == {
        "model": "gpt-4o-mini",
        "api_base": "",
        "fast_model": "",
        "fast_api_base": "",
        "interpret": False,
        "direct": True,
        "workers": 2,
    }
    assert "not-a-real-key" not in settings_file.read_text()


def test_the_panel_refuses_to_be_handed_a_key(client):
    """Closed on purpose: an unknown field is rejected rather than
    stored, so no client can talk this into keeping one."""
    for field in ("api_key", "openai_api_key", "token"):
        refused = client.put("/settings", json={"model": "gpt-4o-mini", field: "sk-x"})
        assert refused.status_code == 422, field


def test_a_new_story_runs_on_what_was_chosen(client):
    client.put("/settings", json={"model": "gpt-4o-mini", "workers": 2, "interpret": False})

    started = client.post("/stories", json={"world": "ashgrove", "scene": "the_dinner"})
    session = client.app.state.sessions[started.json()["session_id"]].session

    assert isinstance(session.llm, LiteLLMClient)
    assert session.llm.model == "gpt-4o-mini"
    assert session.director.workers == 2
    assert session.director.interpret_beliefs is False


def test_a_story_already_open_changes_model_too(client):
    """Four objects hold the client, and a change that reached three of
    them would leave a character still talking to the old endpoint."""
    started = client.post("/stories", json={"world": "ashgrove", "scene": "the_dinner"})
    session = client.app.state.sessions[started.json()["session_id"]].session

    client.put("/settings", json={"model": "gpt-4o-mini"})

    for holder in (session.llm, session.director.llm, session.director.contexts.llm,
                   session.director.narrator.llm):
        assert isinstance(holder, LiteLLMClient), "every holder of the client moved"
        assert holder.model == "gpt-4o-mini"


def test_nothing_about_the_story_moves_when_the_model_does(client):
    started = client.post("/stories", json={"world": "ashgrove", "scene": "the_dinner"})
    session_id = started.json()["session_id"]
    session = client.app.state.sessions[session_id].session
    before = [e.content for e in session.store.get_events(session.scene.id)]

    client.put("/settings", json={"model": "gpt-4o-mini"})

    assert [e.content for e in session.store.get_events(session.scene.id)] == before
    assert client.get(f"/sessions/{session_id}").status_code == 200


def test_a_model_set_on_the_command_line_wins_and_says_so(tmp_path, settings_file):
    """A flag is for this run. The panel should not quietly disagree with
    what the process was started with."""
    Settings(model="gpt-4o-mini").save(settings_file)
    app = create_app(
        worlds_root=WORLDS,
        llm=FakeLLM(),
        library_root=tmp_path / "stories",
        settings_path=settings_file,
    )
    with TestClient(app) as client:
        assert client.get("/settings").json()["pinned"] is True
        refused = client.put("/settings", json={"model": "gpt-4o-mini"})
        assert refused.status_code == 409
        assert "command line" in refused.json()["detail"]

        started = client.post("/stories", json={"world": "ashgrove", "scene": "the_dinner"})
        session = app.state.sessions[started.json()["session_id"]].session
        assert isinstance(session.llm, FakeLLM)


# --- The other door ---------------------------------------------------


def test_the_terminal_uses_the_same_choice(settings_file, tmp_path, monkeypatch, offline):
    """Two front doors that disagreed about which model answers would be
    the same bug as a flag nobody can change from the app."""
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    Settings(model="gpt-4o-mini", workers=2, interpret=False).save(settings_file)

    from fabula import cli

    played = {}
    monkeypatch.setattr(cli, "run", lambda session, **kwargs: played.update(session=session))
    cli.main([
        "ashgrove", "the_dinner",
        "--worlds", str(WORLDS),
        "--library", str(tmp_path / "stories"),
        "--settings", str(settings_file),
    ])

    session = played["session"]
    assert isinstance(session.llm, LiteLLMClient)
    assert session.llm.model == "gpt-4o-mini"
    assert session.director.workers == 2
    assert session.director.interpret_beliefs is False
    session.close()


def test_a_flag_still_wins_in_the_terminal(settings_file, tmp_path, monkeypatch, offline):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    Settings(model="gpt-4o-mini").save(settings_file)

    from fabula import cli

    played = {}
    monkeypatch.setattr(cli, "run", lambda session, **kwargs: played.update(session=session))
    cli.main([
        "ashgrove", "the_dinner",
        "--model", "gpt-4o",
        "--worlds", str(WORLDS),
        "--library", str(tmp_path / "stories"),
        "--settings", str(settings_file),
    ])

    assert played["session"].llm.model == "gpt-4o"
    played["session"].close()
