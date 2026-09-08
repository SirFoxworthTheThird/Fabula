"""The engine speaks no language of its own.

Everything the engine can put inside a character's perception used to be
an English constant in Python: what you hear through a wall, how a jump
in time feels, the words ignored when judging whether two lines are
about the same thing. A scene in Portuguese got English injected into it,
and in Cyrillic or Japanese the memory system silently stopped working.

Asking an author to enumerate a language is the wrong shape — you cannot
list the ways of saying "someone arrived". Naming the things a story
turns on is not: those have names, and the author knows them. So facts
stay authored keywords, and everything else moves out of Python or is
made script-agnostic.
"""
from pathlib import Path

import pytest

from fabula.chronology import describe_duration, render_time_skip
from fabula.llm import FakeLLM
from fabula.loader import load_world
from fabula.memory import _significant_words, find_rehearsed, project, retrieve, TieredEvent
from fabula.models import Event, ProjectedEvent
from fabula.session import Session
from fabula.world import (
    DEGRADED_TEMPLATES,
    DURATION_TEMPLATES,
    TIME_SKIP_TEMPLATES,
    degrade_content,
    write_in,
)

WORLDS = Path(__file__).parent.parent / "worlds"
VILAMAR = WORLDS / "vilamar"
ALL_WORLDS = sorted(p.parent for p in WORLDS.glob("*/world.yaml"))


def a_line(text: str, seq: int = 1) -> ProjectedEvent:
    from datetime import datetime

    event = Event(
        id=seq, scene_id="s", seq=seq, story_time=datetime(2024, 1, 1),
        kind="utterance", actor_id="a", location_id="r", content=text,
    )
    return ProjectedEvent(event=event, perceived_content=text, perception="full")


# --- words, in every script -------------------------------------------

@pytest.mark.parametrize(
    "language, text",
    [
        ("English", "He broke his grandmother's music box"),
        ("Portuguese", "Ele quebrou a caixa de música da avó"),
        ("Russian", "Он разбил музыкальную шкатулку своей бабушки"),
        ("Greek", "Έσπασε το μουσικό κουτί της γιαγιάς του"),
        ("Japanese", "彼は祖母のオルゴールを壊した"),
    ],
)
def test_every_script_yields_something_to_compare(language, text):
    """`[a-z0-9']+` on a lowercased string is an ASCII range. Cyrillic and
    Japanese produced the empty set, so rehearsal and retrieval could
    never fire in those languages at all."""
    assert _significant_words(text), language


def test_an_accent_no_longer_cuts_a_word_in_half():
    words = _significant_words("Ele quebrou a caixa de música da avó")

    assert "música" in words
    assert "sica" not in words  # what the ASCII range left behind


def test_a_line_that_echoes_another_is_recognised_in_portuguese():
    """Rehearsal (§6 rule 3) and retrieval (rule 4) both run on this."""
    first = a_line("Ele quebrou a caixa de música da avó", 1)
    echo = a_line("A caixa de música da avó já estava partida", 2)

    assert find_rehearsed(echo, [first])


def test_the_same_holds_in_a_non_latin_script():
    first = a_line("Он разбил музыкальную шкатулку", 1)
    echo = a_line("Музыкальную шкатулку уже разбил кто-то", 2)

    assert find_rehearsed(echo, [first])


def test_japanese_gets_signal_from_bigrams_rather_than_nothing():
    """Not segmentation — a real tokenizer would be better, and this is
    documented as a limit rather than a solution. But two lines about the
    same thing now overlap, and two about different things do not."""
    box = a_line("彼はオルゴールを壊した", 1)
    same = a_line("オルゴールはもう壊れていた", 2)
    other = a_line("今日は天気がとてもいいですね", 3)

    assert find_rehearsed(same, [box])
    assert not find_rehearsed(other, [box])


def test_stopwords_are_the_worlds_own():
    """An English list filters nothing in Portuguese, so common words
    count as evidence that two lines are about the same thing."""
    world = load_world(VILAMAR)

    assert "para" in world.phrasing.stopwords
    assert "para" not in _significant_words("para onde foi", world.phrasing.stopwords)
    assert "para" in _significant_words("para onde foi")  # with the English default


def test_retrieval_uses_them(scenario):
    world, _, _ = scenario
    tiered = [TieredEvent(projected=a_line("Ele quebrou a caixa de música"), tier="gist", salience=0.5)]

    assert retrieve("A caixa de música partida", tiered)


# --- perceived text is the author's ------------------------------------

def test_what_you_half_hear_is_in_the_worlds_language():
    from datetime import datetime

    world = load_world(VILAMAR)
    event = Event(
        scene_id="s", seq=1, story_time=datetime(2024, 5, 18), kind="utterance",
        actor_id="joaquim", location_id="cozinha", content="Vendi o barco.",
    )

    assert degrade_content(event, world) == "vozes abafadas vindas da cozinha, palavras indistintas"


def test_time_is_felt_in_the_worlds_language():
    from datetime import datetime

    world = load_world(VILAMAR)
    skip = Event(
        scene_id="s", seq=1, story_time=datetime(2024, 5, 18), kind="time_skip",
        location_id="cozinha", content="", metadata={"minutes": 25},
    )

    assert describe_duration(25, world.phrasing) == "25 minutos"
    assert describe_duration(60, world.phrasing) == "uma hora"
    assert render_time_skip(skip, "ines", [skip], world.phrasing) == (
        "(25 minutos a passar, e sente-se cada um deles)"
    )


def test_nothing_english_reaches_a_portuguese_scene(fake_llm):
    """The whole point, end to end: play a scene and read back every
    string the engine put into somebody's perception."""
    session = Session.open(VILAMAR, "o_almoco", llm=fake_llm)
    build = session.director.build_event
    session.store.append_event(
        build("utterance", "joaquim", "cozinha", "Vendi o barco.", audibility="adjacent")
    )
    session.store.append_event(
        build("time_skip", None, "cozinha", "25 minutos.", audibility="building",
              metadata={"minutes": 25})
    )
    log = session.store.get_events(session.scene.id)

    english = {"muffled", "words unclear", "a door somewhere", "footsteps fading",
               "time passes", "minutes pass", "an hour", "you feel every one"}
    for character in session.characters.values():
        for perceived in project(character, log, session.world,
                                 initial_location=character.location_id):
            lowered = perceived.perceived_content.lower()
            assert not [phrase for phrase in english if phrase in lowered], lowered


@pytest.mark.parametrize("world_dir", ALL_WORLDS, ids=lambda p: p.name)
def test_a_world_in_another_language_leaves_no_english_behind(world_dir):
    """A partial `phrasing` block falls back to the built-in English per
    key, which keeps an English world working and would quietly put
    English inside a Portuguese one. Half-translated is an authoring
    error, and this is where it is caught."""
    world = load_world(world_dir)
    if world.language.split("-")[0] == "en":
        return

    for kind, default in DEGRADED_TEMPLATES.items():
        assert world.phrasing.degraded.get(kind) != default, f"degraded.{kind} is still English"
    for key, default in DURATION_TEMPLATES.items():
        assert world.phrasing.duration.get(key) != default, f"duration.{key} is still English"
    for key, default in TIME_SKIP_TEMPLATES.items():
        assert world.phrasing.time_skip.get(key) != default, f"time_skip.{key} is still English"


def test_a_partial_block_keeps_the_rest_rather_than_dropping_it():
    """Merged per key, so restating one line does not silently lose the
    others to a partial block."""
    world = load_world(VILAMAR)

    assert set(world.phrasing.degraded) == set(DEGRADED_TEMPLATES)


# --- what the model is told -------------------------------------------

def test_english_worlds_get_no_extra_prompt_line():
    """Every prompt-level rule in this project was measured on the two
    English worlds. Quietly adding a sentence to their prompts would
    invalidate that."""
    assert write_in("en") == ""
    assert write_in("en-GB") == ""
    assert write_in("") == ""


def test_a_non_english_world_tells_the_model_which_language():
    line = write_in("pt-PT")

    assert "pt-PT" in line


def test_the_language_reaches_an_actual_prompt():
    from fabula.agents import generate_utterance

    llm = FakeLLM()
    session = Session.open(VILAMAR, "o_almoco", llm=llm)
    events = session.store.get_events(session.scene.id)

    generate_utterance(session.characters["joaquim"], events, session.director.contexts, llm)

    system, _, _ = llm.calls[-1]
    assert "pt-PT" in system


# --- what the client says in its own voice ------------------------------

def test_a_client_line_is_the_worlds_too():
    """Not story text, but read in the same breath. "You are in o
    quintal" is the same bug as an English descriptor, one layer out."""
    world = load_world(VILAMAR)

    assert world.phrasing.say("now_in", room="o quintal") == "Estás no quintal."
    assert world.phrasing.say("no_answer") == "Ninguém responde."


def test_the_command_layer_speaks_it(fake_llm):
    from fabula.commands import run_command

    session = Session.open(VILAMAR, "o_almoco", llm=fake_llm)

    assert "Não há nenhum sítio" in run_command(session, "/go lisboa").message
    assert run_command(session, "/again").message == "Ainda não se jogou nada."


def test_commands_themselves_stay_the_same_everywhere(fake_llm):
    """A command that changes name per world is a command nobody can
    document. The prose is the author's; the verbs are not."""
    from fabula.commands import run_command

    session = Session.open(VILAMAR, "o_almoco", llm=fake_llm)

    assert run_command(session, "/look").message or run_command(session, "/look").perceived
    assert run_command(session, "/quit").quit is True


@pytest.mark.parametrize("world_dir", ALL_WORLDS, ids=lambda p: p.name)
def test_no_english_client_line_survives_in_another_language(world_dir):
    from fabula.world import CLIENT_TEMPLATES

    world = load_world(world_dir)
    if world.language.split("-")[0] == "en":
        return

    for key, default in CLIENT_TEMPLATES.items():
        assert world.phrasing.client.get(key) != default, f"client.{key} is still English"


def test_english_worlds_are_completely_unchanged():
    """Two worlds and every measurement in this project depend on the
    English defaults staying exactly as they were."""
    from fabula.world import CLIENT_TEMPLATES

    world = load_world(WORLDS / "ashgrove")

    assert world.language == "en"
    assert world.phrasing.degraded == DEGRADED_TEMPLATES
    assert world.phrasing.duration == DURATION_TEMPLATES
    assert world.phrasing.time_skip == TIME_SKIP_TEMPLATES
    assert world.phrasing.client == CLIENT_TEMPLATES
