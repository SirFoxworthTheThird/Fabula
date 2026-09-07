"""Measure whether a prompt-level rule actually changes behaviour.

The engine's mechanical guarantees hold whatever model is behind them:
perception filtering, the withholding bid, the presence line, the
narrator's refusal to bid on the protagonist. The rules that live in
prompts do not — they are worth exactly as much as the model reading
them, which differs by model and cannot be assumed.

This measures one of them: does telling a character what they guard
change how often they raise it unprompted? Three variants of the same
scene opening, N samples each:

    no guard line     — the character is told nothing
    names the subject — what the engine currently ships
    guarded, unnamed  — told there is a subject, but not which

On a 1.5B local model the answer was no: 3/16, 4/16, 3/16, which is
noise. That model ignores most prompt discipline, so it says little
about a capable one. Run this against the model you actually intend to
ship before believing the shipped line does anything.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fabula.agents as agents
from fabula.llm import LiteLLMClient, LLMClient, get_default_llm
from fabula.models import Character, Event
from fabula.session import Session

OPENING = "Tomás, you have been quiet all evening."


def _no_guard(character: Character, contexts, events: list[Event]) -> str:
    return ""


def _unnamed_guard(character: Character, contexts, events: list[Event]) -> str:
    if not character.protects:
        return ""
    return (
        "\nThere is one subject you will not discuss and never raise yourself. "
        "If it comes up, you turn the conversation rather than answer."
    )


VARIANTS = {
    "no guard line": _no_guard,
    "names the subject": None,  # whatever the engine currently ships
    "guarded, unnamed": _unnamed_guard,
}


def guarded_keywords(session: Session, character: Character) -> list[str]:
    return [
        keyword.lower()
        for fact_id in character.protects
        for keyword in session.world.facts[fact_id].keywords
        if fact_id in session.world.facts
    ]


def measure_guard(
    world_dir: Path,
    scene_name: str,
    character_id: str,
    opening: str = OPENING,
    samples: int = 16,
    llm: LLMClient | None = None,
) -> dict[str, int]:
    """How often the character names what they guard, unprompted, per variant.

    Swaps the guard-line builder for each variant. That is a measurement
    tool reaching into the module under test, which is why it lives here
    and not in the engine's own path.
    """
    llm = llm or get_default_llm()
    shipped = agents._guarded_subjects
    results: dict[str, int] = {}

    try:
        for label, variant in VARIANTS.items():
            agents._guarded_subjects = variant or shipped
            blurts = 0
            for _ in range(samples):
                session = Session.open(world_dir, scene_name, llm=llm)
                character = session.characters[character_id]
                session.store.append_event(
                    session.director.build_event(
                        "utterance",
                        session.user_character.id,
                        session.here(),
                        opening,
                    )
                )
                line = agents.generate_utterance(
                    character,
                    session.store.get_events(session.scene.id),
                    session.director.contexts,
                    llm,
                ).lower()
                if any(word in line for word in guarded_keywords(session, character)):
                    blurts += 1
            results[label] = blurts
    finally:
        agents._guarded_subjects = shipped

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fabula-measure",
        description="Measure whether the guarded-subject prompt line changes behaviour.",
    )
    parser.add_argument("world_dir", type=Path, help="Path to worlds/<name>/")
    parser.add_argument("scene", help="Scene name (file stem under scenes/)")
    parser.add_argument("--character", default="tomas", help="Who is guarding something")
    parser.add_argument("--samples", type=int, default=16, help="Openings per variant")
    parser.add_argument("--model", default=None, help="Any model id litellm understands")
    parser.add_argument("--api-base", default=None, help="An OpenAI-compatible endpoint")
    args = parser.parse_args(argv)

    llm = LiteLLMClient(model=args.model, api_base=args.api_base) if args.model else None
    results = measure_guard(
        args.world_dir,
        args.scene,
        args.character,
        samples=args.samples,
        llm=llm,
    )

    print(f"\n{args.character} named what they guard, unprompted:\n")
    for label, blurts in results.items():
        print(f"  {label:18}: {blurts:3}/{args.samples}")
    print(
        "\nDifferences of one or two are noise at this sample size. If the "
        "variants land together, the shipped line is doing nothing on this model."
    )


if __name__ == "__main__":
    main()
