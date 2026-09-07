"""Invariant 1: a character can never be given information their
character could not perceive. These are the highest-priority tests in
the suite (spec §13) — everything else is negotiable, this is not.
"""
from fabula.agents import generate_utterance
from fabula.llm import FakeLLM
from fabula.memory import assemble_context, project
from fabula.models import Event

SECRET = "music box"


def secret_event(scene, seq: int = 1) -> Event:
    return Event(
        scene_id=scene.id,
        seq=seq,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="tomas",
        location_id="kitchen",
        content="I broke Grandma's music box, Elena. It wasn't the cat.",
        audibility="room",
        addressed_to=["elena"],
    )


def test_maria_projection_contains_zero_tokens_of_the_secret(scenario, store, fake_llm):
    world, characters, scene = scenario
    event = secret_event(scene)
    maria = characters["maria"]

    projected = project(maria, [event], world)

    assert all(SECRET not in p.perceived_content.lower() for p in projected)
    assert all("cat" not in p.perceived_content.lower() for p in projected)
    context = assemble_context(maria, projected, scene.id, store, fake_llm)
    assert SECRET not in context.lower()


def test_maria_reply_does_not_reference_the_secret(scenario, contexts, fake_llm):
    _world, characters, scene = scenario
    event = secret_event(scene)
    maria = characters["maria"]

    reply = generate_utterance(maria, [event], contexts, fake_llm)

    assert SECRET not in reply.lower()
    # FakeLLM's output is a pure function of its prompt and invents nothing,
    # so this also proves the prompt itself never carried the secret.
    system, prompt, _ = fake_llm.calls[-1]
    assert SECRET not in system.lower()
    assert SECRET not in prompt.lower()


def test_tomas_in_the_same_room_does_perceive_it(scenario):
    world, characters, scene = scenario
    event = secret_event(scene)
    tomas = characters["tomas"]

    projected = project(tomas, [event], world)

    assert len(projected) == 1
    assert projected[0].perception == "full"
    assert SECRET in projected[0].perceived_content.lower()


def test_degraded_perception_never_carries_true_content(scenario):
    """A louder event (audibility 'adjacent') reaches the next room, but
    only as a degraded descriptor — never the real content."""
    world, characters, scene = scenario
    event = Event(
        scene_id=scene.id,
        seq=1,
        story_time=scene.start_time,
        kind="utterance",
        actor_id="tomas",
        location_id="kitchen",
        content="I broke Grandma's music box!",
        audibility="adjacent",
    )
    maria = characters["maria"]

    projected = project(maria, [event], world)

    assert len(projected) == 1
    assert projected[0].perception == "degraded"
    assert SECRET not in projected[0].perceived_content.lower()


def test_narration_never_describes_what_an_offscreen_character_cannot_perceive(scenario):
    world, characters, scene = scenario
    trigger = secret_event(scene, seq=1)
    narration = Event(
        scene_id=scene.id,
        seq=2,
        story_time=scene.start_time,
        kind="narration",
        actor_id=None,
        location_id="kitchen",
        content="Tomás glances guiltily at the music box on the shelf.",
        audibility="room",
    )
    maria = characters["maria"]

    projected = project(maria, [trigger, narration], world)

    assert all(SECRET not in p.perceived_content.lower() for p in projected)


def test_bid_rationale_never_leaks_into_another_characters_context(scenario, monkeypatch):
    """The director sees every bid, but a rationale must never travel
    further than the director: it is never written into an Event, so it
    can never reach another character's projection."""
    from fabula.db import EventStore
    from fabula.director import Director
    import fabula.director as director_module
    from fabula.narrator import Narrator

    world, characters, scene = scenario
    store = EventStore()
    director = Director(store, world, characters, scene, Narrator(FakeLLM()), FakeLLM())

    marker = "RATIONALE_MARKER_SHOULD_NEVER_LEAK"
    original_get_bid = director_module.get_bid

    def spy_get_bid(character, event, level, events, contexts, llm):
        bid = original_get_bid(character, event, level, events, contexts, llm)
        return bid.model_copy(update={"one_line_reason": marker})

    monkeypatch.setattr(director_module, "get_bid", spy_get_bid)

    user_event = director.build_event(
        "utterance", "elena", "kitchen", "Tomas, tell me something, anything!"
    )
    turn_events = director.run_turn(user_event)

    for event in turn_events:
        assert marker not in event.content

    for character in characters.values():
        assert marker not in director.contexts.for_character(character, store.get_events(scene.id))


def test_secret_does_not_leak_after_50_turns_and_a_summarization_pass(
    scenario, store, fake_llm
):
    """Spec §13: summaries must not leak what the projection excluded.
    Summaries are built from perceived_content only, so a span Maria
    never perceived cannot be compacted into her memory."""
    world, characters, scene = scenario
    maria = characters["maria"]

    events = [secret_event(scene, seq=1)]
    for seq in range(2, 62):
        # Maria's own room fills up, pushing anything old into the
        # summarized and gist tiers.
        events.append(
            Event(
                id=seq,
                scene_id=scene.id,
                seq=seq,
                story_time=scene.start_time,
                kind="utterance",
                actor_id="maria",
                location_id="study",
                content=f"Maria sorts another letter, number {seq}.",
                audibility="room",
            )
        )

    projected = project(maria, events, world)
    context = assemble_context(maria, projected, scene.id, store, fake_llm)

    assert len(projected) == 60  # she perceived everything in the study, nothing from the kitchen
    assert SECRET not in context.lower()
    assert "[earlier," in context  # summarization really did run

    # The summarizer's own prompts never carried the secret either.
    for system, prompt, _key in fake_llm.calls:
        assert SECRET not in system.lower()
        assert SECRET not in prompt.lower()

    # And nothing stored in the summaries table carries it.
    rows = store.conn.execute("SELECT summary_text FROM summaries").fetchall()
    assert rows
    assert all(SECRET not in row["summary_text"].lower() for row in rows)


def test_the_cli_prints_only_the_user_characters_projection(scenario, store, capsys):
    """A client that prints the raw log hands the player their own
    character's blind spots. The terminal is a POV, not a transcript."""
    from fabula.cli import _show
    from fabula.director import Director
    from fabula.narrator import Narrator
    from fabula.session import Session

    world, characters, scene = scenario
    director = Director(store, world, characters, scene, Narrator(FakeLLM()), FakeLLM())
    session = Session(world, characters, scene, store, director, characters["elena"])

    # Elena steps out to the study; Tomás says the secret in the kitchen.
    store.append_event(
        director.build_event("arrival", "elena", "study", "Elena steps through.")
    )
    said = store.append_event(
        director.build_event(
            "utterance",
            "tomas",
            "kitchen",
            "I broke Grandma's music box, and I never told anyone.",
        )
    )

    _show(session.pov([said]), characters)

    printed = capsys.readouterr().out
    assert SECRET not in printed.lower()
    assert printed == ""  # a room away with room-scoped speech, she gets nothing


def test_secret_does_not_cross_a_scene_boundary_through_persistence(scenario, store):
    """Spec §13: the same leak test, across a scene boundary with
    persistence. Beliefs are encoded from a character's own projection,
    so what Maria never perceived cannot be carried into the next scene
    as something she remembers."""
    from fabula.director import Director
    from fabula.memory import form_belief
    from fabula.narrator import Narrator
    from fabula.persistence import age_beliefs, begin_scene, encode_belief

    world, characters, scene = scenario
    begin_scene(store, characters)
    director = Director(store, world, characters, scene, Narrator(FakeLLM()), FakeLLM())

    # Scene one: Tomás says it in the kitchen, Maria is in the study.
    director.run_turn(
        director.build_event(
            "utterance",
            "tomas",
            "kitchen",
            "I broke Grandma's music box, and I let them blame the cat.",
        )
    )

    # Whatever anyone encoded, hers is free of it — and so is everything
    # she carries forward.
    assert all(SECRET not in b.content.lower() for b in store.get_beliefs("maria"))

    # Scene two, same store: beliefs age across the boundary and come back.
    begin_scene(store, characters)
    later = scene.model_copy(update={"id": "the_morning_after"})
    next_director = Director(store, world, characters, later, Narrator(FakeLLM()), FakeLLM())
    next_director.run_turn(
        next_director.build_event("utterance", "elena", "study", "Maria, did you sleep?")
    )

    assert all(SECRET not in b.content.lower() for b in store.get_beliefs("maria"))
    context = next_director.contexts.for_character(
        characters["maria"], store.get_events(later.id)
    )
    assert SECRET not in context.lower()

    # And Tomás, who did say it, still remembers it a scene later.
    age_beliefs(store, "tomas")
    assert any(SECRET in b.content.lower() for b in store.get_beliefs("tomas"))
