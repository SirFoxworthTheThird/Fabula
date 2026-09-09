"""The post-scene reveal.

Asymmetry is invisible while you play — a character with nothing to say
looks exactly like a character saying nothing. This is where it becomes
visible, and it is the one part of the engine that steps outside a
point of view on purpose.
"""
from fabula.reveal import build_reveal, knows_fact, render, reveal_text
from fabula.session import Session

from tests.conftest import ASHGROVE

SECRET = "music box"


def a_session(fake_llm):
    return Session.open(ASHGROVE, "the_dinner", llm=fake_llm)


def test_it_lists_what_happened_out_of_sight(fake_llm):
    session = a_session(fake_llm)
    session.move("study")  # Elena leaves; Tomás stays in the kitchen
    session.store.append_event(
        session.director.build_event(
            "utterance", "tomas", "kitchen", "I broke Grandma's music box, and I never said."
        )
    )

    built = build_reveal(session)

    # Whatever else went on in the kitchen once she had left, the one
    # thing she needed to miss is in there.
    missed = [e.content for e in built.missed]
    assert "I broke Grandma's music box, and I never said." in missed
    perceived = {p.perceived_content for p in session.perceived_so_far()}
    assert not any(content in perceived for content in missed), "missed means missed"
    assert SECRET in render(built, session).lower()  # the reveal is where it surfaces


def test_a_reveal_never_appears_in_a_character_context(fake_llm):
    """It is a spoiler for the player, not a channel into the fiction."""
    session = a_session(fake_llm)
    session.move("study")
    session.store.append_event(
        session.director.build_event(
            "utterance", "tomas", "kitchen", "I broke Grandma's music box."
        )
    )

    reveal_text(session)

    events = session.store.get_events(session.scene.id)
    # Every character, and judged against what they actually perceived —
    # the scene is free to move somebody into earshot, and the invariant
    # is that the reveal is not what put the secret there.
    for character in session.characters.values():
        context = session.director.contexts.for_character(character, events).lower()
        heard = any(
            SECRET in p.perceived_content.lower()
            for p in session.director.contexts.project(character, events)
        )
        assert "never knew" not in context, character.id
        if not heard:
            assert SECRET not in context, character.id


def test_asking_for_a_reveal_changes_nothing(fake_llm):
    """Read-only, so you can look and keep playing."""
    session = a_session(fake_llm)
    session.say("Tomas?")
    before = [(e.seq, e.content) for e in session.store.get_events(session.scene.id)]
    beliefs_before = len(session.store.get_beliefs("tomas"))

    reveal_text(session)
    reveal_text(session)

    after = [(e.seq, e.content) for e in session.store.get_events(session.scene.id)]
    assert after == before
    assert len(session.store.get_beliefs("tomas")) == beliefs_before


def test_it_pairs_what_was_half_heard_with_what_it_was(fake_llm):
    session = a_session(fake_llm)
    session.move("study")
    session.store.append_event(
        session.director.build_event(
            "utterance",
            "tomas",
            "kitchen",
            "I BROKE THE MUSIC BOX!",
            audibility="adjacent",
        )
    )

    built = build_reveal(session)

    assert built.half_heard
    heard, event = built.half_heard[-1]
    assert SECRET not in heard.lower()          # she only got a muffled version
    assert SECRET in event.content.lower()      # this is what it actually was


def test_a_time_skip_is_not_something_you_misheard(fake_llm):
    """It reads as degraded outside its own room, but its wording is
    written per character on purpose. Nobody mishears an hour passing."""
    session = a_session(fake_llm)
    session.move("study")
    session.director.advance_time(minutes=30, consent=lambda _m: True)

    built = build_reveal(session)

    assert all(event.kind != "time_skip" for _heard, event in built.half_heard)


def test_who_knew_the_secret_at_the_end(fake_llm):
    session = a_session(fake_llm)

    built = build_reveal(session)

    # Tomás holds it because it is authored as his; nobody else has heard it.
    assert built.knowledge["music_box"] == {"Tomás": True, "Maria": False, "Elena": False}
    assert knows_fact(session, session.characters["tomas"], "music_box")


def test_hearing_it_said_counts_as_knowing(fake_llm):
    session = a_session(fake_llm)
    session.store.append_event(
        session.director.build_event(
            "utterance", "tomas", "kitchen", "I broke Grandma's music box."
        )
    )

    built = build_reveal(session)

    assert built.knowledge["music_box"]["Elena"] is True   # she was in the room
    assert built.knowledge["music_box"]["Maria"] is False  # she was not


def test_half_hearing_it_does_not_count_as_knowing(fake_llm):
    """The degraded descriptor never carries the words, so someone who
    only caught raised voices did not learn the thing."""
    session = a_session(fake_llm)
    session.store.append_event(
        session.director.build_event(
            "utterance",
            "tomas",
            "kitchen",
            "I broke Grandma's music box!",
            audibility="adjacent",
        )
    )

    built = build_reveal(session)

    assert built.knowledge["music_box"]["Maria"] is False


def test_a_quiet_scene_says_so_rather_than_printing_nothing(fake_llm):
    session = a_session(fake_llm)

    text = reveal_text(session)

    assert "Nothing happened out of your sight." in text


def test_the_reveal_shows_regard_that_moved_out_of_sight(fake_llm):
    """Elena has no way to know her sister stopped believing her brother
    tonight — which is precisely why the reveal is where it belongs."""
    session = Session.open(ASHGROVE, "the_reckoning", llm=fake_llm)
    session.say("What happened to Grandma's music box?")

    built = build_reveal(session)
    moved = {(r.who, r.toward): r for r in built.regard}

    assert ("Maria", "Tomás") in moved
    entry = moved[("Maria", "Tomás")]
    assert entry.trust < entry.started
    # The player's own feelings are not reported back to them.
    assert not [r for r in built.regard if r.who == session.user_character.name]
    assert "Maria trusts Tomás less" in render(built, session)


def test_a_scene_where_nothing_moved_reports_nothing(fake_llm):
    session = Session.open(ASHGROVE, "the_dinner", llm=fake_llm)

    built = build_reveal(session)

    assert built.regard == []
    assert "trusts" not in render(built, session)
