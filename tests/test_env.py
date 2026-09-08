"""Reading a local `.env`.

Credentials reach this project through the environment and nowhere else:
no `--api-key` flag exists, nothing writes a key to disk, and no key is
passed as an argument where it would land in shell history or a process
listing. A dotfile is the one concession to convenience, so it has to
keep those properties rather than quietly undo them.
"""
import os

import pytest

from fabula.env import load_env, parse_env


def test_it_parses_the_shapes_people_actually_write():
    parsed = parse_env(
        "\n".join(
            [
                "# a comment",
                "",
                "OPENAI_API_KEY=sk-plain",
                'QUOTED="sk-quoted"',
                "SINGLE='sk-single'",
                "export EXPORTED=sk-exported",
                "SPACED = sk-spaced ",
                "URL=https://example.invalid/v1?a=b",
            ]
        )
    )

    assert parsed == {
        "OPENAI_API_KEY": "sk-plain",
        "QUOTED": "sk-quoted",
        "SINGLE": "sk-single",
        "EXPORTED": "sk-exported",
        "SPACED": "sk-spaced",
        "URL": "https://example.invalid/v1?a=b",  # only the first = splits
    }


def test_a_malformed_line_is_skipped_rather_than_raised_on():
    """A typo in a dotfile should not stop someone playing a scene."""
    parsed = parse_env("this is not an assignment\nGOOD=yes\n=novalue\n")

    assert parsed == {"GOOD": "yes"}


def test_an_exported_variable_always_wins(tmp_path, monkeypatch):
    """A file that silently shadowed a key you set yourself would make it
    impossible to say which credential a run actually used."""
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-shell")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-the-file\nOTHER=from-the-file\n")

    applied = load_env(env_file)

    assert os.environ["OPENAI_API_KEY"] == "from-the-shell"
    assert os.environ["OTHER"] == "from-the-file"
    assert applied == ["OTHER"]


def test_it_reports_names_and_never_values(tmp_path, monkeypatch):
    monkeypatch.delenv("SECRET_TOKEN", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET_TOKEN=sk-do-not-print-me\n")

    applied = load_env(env_file)

    assert applied == ["SECRET_TOKEN"]
    assert "sk-do-not-print-me" not in str(applied)


def test_a_missing_file_is_not_an_error(tmp_path):
    """Most runs will not have one."""
    assert load_env(tmp_path / "nothing-here") == []


def test_a_world_readable_file_is_flagged(tmp_path, monkeypatch, capsys):
    if os.name != "posix":
        pytest.skip("mode bits are POSIX-only")
    monkeypatch.delenv("SOME_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_KEY=value\n")
    env_file.chmod(0o644)

    load_env(env_file)

    warning = capsys.readouterr().err
    assert "readable by other users" in warning
    assert "value" not in warning


def test_a_private_file_says_nothing(tmp_path, monkeypatch, capsys):
    if os.name != "posix":
        pytest.skip("mode bits are POSIX-only")
    monkeypatch.delenv("SOME_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_KEY=value\n")
    env_file.chmod(0o600)

    load_env(env_file)

    assert capsys.readouterr().err == ""


def test_the_repo_refuses_to_track_a_real_dotenv():
    """The whole point. `.env.example` is the only one that belongs
    here."""
    from pathlib import Path

    ignore = (Path(__file__).parent.parent / ".gitignore").read_text(encoding="utf-8")
    rules = ignore.splitlines()

    assert ".env" in rules
    assert ".env.*" in rules
    assert "!.env.example" in rules


def test_nothing_accepts_a_key_as_an_argument():
    """A flag would put the key in shell history and in every `ps` on the
    machine; a parameter would let a caller pass one through a signature
    instead of the environment. There is deliberately neither, and this
    is the test that keeps it that way.

    Checked against the syntax tree rather than the text, so the prose
    explaining the rule does not trip the rule.
    """
    import ast
    from pathlib import Path

    for module in sorted((Path(__file__).parent.parent / "fabula").glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.arg):
                assert "api_key" not in node.arg, f"{module.name}: {node.arg}"
            if isinstance(node, ast.keyword):
                assert node.arg is None or "api_key" not in node.arg, module.name
            if isinstance(node, ast.Call):
                literals = [
                    a.value for a in node.args if isinstance(a, ast.Constant)
                    and isinstance(a.value, str)
                ]
                assert "--api-key" not in literals, module.name
