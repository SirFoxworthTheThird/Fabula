"""The playtest harness — an author's tool for judging whether a scene
reads well, which is the one thing the test suite cannot decide."""
from fabula.commands import run_command
from fabula.playtest import DEFAULT_SCRIPT, playtest, read_script
from fabula.session import Session

from tests.conftest import ASHGROVE


def test_a_script_file_ignores_blanks_and_comments(tmp_path):
    script = tmp_path / "scene.txt"
    script.write_text("# a comment\n\nsay this\n  /look  \n")

    assert read_script(script) == ["say this", "/look"]


def test_no_script_falls_back_to_the_default(tmp_path):
    assert read_script(None) == DEFAULT_SCRIPT


def test_the_default_script_presses_on_the_secret():
    """The point of the default run is to exercise the parts most likely
    to read badly: being asked about the secret, and leaving the room."""
    joined = " ".join(DEFAULT_SCRIPT).lower()

    assert "music box" in joined
    assert "/go" in joined and "/wait" in joined and "/look" in joined


def test_a_playtest_runs_end_to_end_and_prints_a_transcript(capsys, fake_llm):
    # The client is injected rather than left to the environment: a
    # contributor with an API key set should not have the unit suite
    # quietly start calling a provider.
    playtest(ASHGROVE, "the_dinner", ["Tomás, hello.", "/look"], show_beliefs=True, llm=fake_llm)

    printed = capsys.readouterr().out
    assert "the_dinner" in printed
    assert "Tomás, hello." in printed
    assert "what each character came away believing" in printed
    assert "no client may ever show this" in printed


def test_transcript_only_mode_omits_the_belief_dump(capsys, fake_llm):
    playtest(ASHGROVE, "the_dinner", ["Tomás, hello."], show_beliefs=False, llm=fake_llm)

    assert "came away believing" not in capsys.readouterr().out


def test_an_empty_room_answers_with_the_room(fake_llm):
    """A player alone used to get "No one answers." and nothing else —
    they would speak, or act, and the world would not move. Silence is a
    real answer when everyone has left, but a room that never responds is
    not a story. So when nobody is there to reply, the narrator does, and
    what it describes is the place — never the person the player is
    playing.
    """
    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)
    run_command(session, "/go study")  # Maria starts there; Tomás does not follow
    session.store.append_event(
        session.director.build_event("arrival", "maria", "kitchen", "Maria steps out.")
    )

    outcome = run_command(session, "Maria? Are you still here?")

    answered = [p for p in outcome.perceived if p.event.actor_id != "elena"]
    assert answered, "the room said something back"
    assert all(p.event.kind == "narration" for p in answered)
    assert outcome.message != "No one answers."


def test_commands_are_shared_with_the_cli(fake_llm):
    """Both terminal clients dispatch through one place, so /go cannot
    come to mean different things in each."""
    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)

    assert run_command(session, "/quit").quit
    assert run_command(session, "").perceived == []
    assert "no cellar to go to" in run_command(session, "/go cellar").message

    moved = run_command(session, "/go study")
    assert "the study" in moved.message
    assert session.here() == "study"
