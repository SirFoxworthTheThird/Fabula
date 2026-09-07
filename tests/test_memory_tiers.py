"""M2: memory tiers, decay, rehearsal, retrieval, and the token budget
(spec §6)."""
from fabula.memory import (
    GIST_AGE,
    HIGH_SALIENCE,
    VERBATIM_WINDOW,
    assemble_context,
    assign_tiers,
    estimate_tokens,
    find_rehearsed,
    project,
    retrieve,
    score_salience,
)
from fabula.models import Event


def kitchen_event(scene, seq, content, salience=0.4, kind="utterance", actor="tomas"):
    return Event(
        id=seq,
        scene_id=scene.id,
        seq=seq,
        story_time=scene.start_time,
        kind=kind,
        actor_id=actor,
        location_id="kitchen",
        content=content,
        audibility="room",
        salience_base=salience,
    )


def long_log(scene, count=60, salience=0.4):
    return [kitchen_event(scene, seq, f"Line number {seq} about the soup.", salience)
            for seq in range(1, count + 1)]


def test_recent_events_stay_verbatim_and_old_ones_decay(scenario):
    world, characters, scene = scenario
    tomas = characters["tomas"]
    projected = project(tomas, long_log(scene), world)

    tiered = assign_tiers(tomas, projected, rehearsals={})

    assert tiered[-1].tier == "verbatim"
    assert tiered[-VERBATIM_WINDOW].tier == "verbatim"
    assert tiered[0].tier == "gist"
    assert {t.tier for t in tiered} == {"verbatim", "summarized", "gist"}


def test_high_salience_events_are_exempt_from_demotion(scenario):
    world, characters, scene = scenario
    tomas = characters["tomas"]
    events = long_log(scene)
    # The oldest event is a betrayal: high base salience, ancient history.
    events[0] = kitchen_event(scene, 1, "You lied to me about everything.", salience=0.95)
    projected = project(tomas, events, world)

    tiered = assign_tiers(tomas, projected, rehearsals={})

    assert score_salience(tomas, projected[0]) >= HIGH_SALIENCE
    assert tiered[0].tier == "verbatim"
    assert tiered[0].exempt


def test_rehearsal_promotes_an_old_event_back_up_the_tiers(scenario):
    world, characters, scene = scenario
    tomas = characters["tomas"]
    events = long_log(scene)
    projected = project(tomas, events, world)

    without = assign_tiers(tomas, projected, rehearsals={})
    # Event 1 was re-mentioned by the most recent event.
    with_rehearsal = assign_tiers(tomas, projected, rehearsals={1: events[-1].seq})

    assert without[0].tier == "gist"
    assert with_rehearsal[0].tier == "verbatim"


def test_rehearsal_detection_matches_on_re_mentioned_content(scenario):
    world, characters, scene = scenario
    tomas = characters["tomas"]
    events = [
        kitchen_event(scene, 1, "Grandmother's cabinet was left unlocked overnight."),
        kitchen_event(scene, 2, "The weather turned cold."),
        kitchen_event(scene, 3, "About that unlocked cabinet — who was it?"),
    ]
    projected = project(tomas, events, world)

    rehearsed = find_rehearsed(projected[-1], projected[:-1])

    assert 1 in rehearsed
    assert 2 not in rehearsed


def test_retrieval_pulls_a_cued_detail_back_into_context(scenario):
    world, characters, scene = scenario
    tomas = characters["tomas"]
    events = long_log(scene)
    events[0] = kitchen_event(scene, 1, "The garden spade went missing from the shed.")
    projected = project(tomas, events, world)
    tiered = assign_tiers(tomas, projected, rehearsals={})
    assert tiered[0].tier != "verbatim"

    promoted = retrieve("Has anyone seen that missing garden spade?", tiered)

    assert 0 in promoted


def test_retrieval_only_ever_sees_this_characters_own_history(scenario):
    """Retrieval takes already-projected events, so a detail Maria never
    perceived cannot be retrieved into her context — there is no path
    from here to the global log."""
    world, characters, scene = scenario
    maria = characters["maria"]
    events = long_log(scene) + [
        kitchen_event(scene, 61, "The spade is buried under the roses.", salience=0.9)
    ]

    projected = project(maria, events, world)
    tiered = assign_tiers(maria, projected, rehearsals={})

    assert projected == []
    assert retrieve("where is the spade?", tiered) == set()


def test_context_stays_within_the_token_budget(scenario, store, fake_llm):
    world, characters, scene = scenario
    tomas = characters["tomas"]
    projected = project(tomas, long_log(scene, count=120), world)

    context = assemble_context(tomas, projected, scene.id, store, fake_llm, budget=200)

    assert estimate_tokens(context) <= 400  # rendering adds tags; content is what's budgeted
    unbudgeted = assemble_context(tomas, projected, scene.id, store, fake_llm, budget=100_000)
    assert len(context) < len(unbudgeted)


def test_summaries_are_computed_once_and_reused(scenario, store, fake_llm):
    """Spec §6 rule 5: re-summarizing on every turn would let beliefs
    drift on their own and destroy reproducibility."""
    world, characters, scene = scenario
    tomas = characters["tomas"]
    projected = project(tomas, long_log(scene), world)

    first = assemble_context(tomas, projected, scene.id, store, fake_llm)
    calls_after_first = len(fake_llm.calls)
    second = assemble_context(tomas, projected, scene.id, store, fake_llm)

    assert first == second
    assert calls_after_first > 0  # summarization did happen
    assert len(fake_llm.calls) == calls_after_first  # and was not repeated


def test_gist_tier_only_ever_holds_perceived_content(scenario):
    world, characters, scene = scenario
    maria = characters["maria"]
    events = [
        Event(
            id=seq,
            scene_id=scene.id,
            seq=seq,
            story_time=scene.start_time,
            kind="utterance",
            actor_id="tomas",
            location_id="kitchen",
            content="I broke Grandma's music box!",
            audibility="adjacent",
            salience_base=0.4,
        )
        for seq in range(1, GIST_AGE + 20)
    ]
    projected = project(maria, events, world)
    tiered = assign_tiers(maria, projected, rehearsals={})

    assert tiered[0].tier == "gist"
    assert all("music box" not in t.projected.perceived_content.lower() for t in tiered)
