"""The text verbs a terminal client understands.

Shared by the interactive CLI and the playtest harness so the two cannot
drift into meaning different things by the same word. Parsing text into
`Session` calls is a client concern, which is why it lives here rather
than in the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from fabula.models import ProjectedEvent
from fabula.session import Session


@dataclass
class Outcome:
    perceived: list[ProjectedEvent] = field(default_factory=list)
    # A note from the client to the player — never story content.
    message: str | None = None
    quit: bool = False


def run_command(
    session: Session, line: str, consent: Callable[[int], bool] | None = None
) -> Outcome:
    line = line.strip()
    if not line:
        return Outcome()

    if line in ("/quit", "/exit"):
        return Outcome(quit=True)

    if line == "/wait":
        if session.pending_skip() is None:
            return Outcome(message="Nothing is pending; time stays where it is.")
        perceived = session.wait(consent=consent)
        if not perceived:
            return Outcome(message="Time stays where it is.")
        return Outcome(perceived=perceived)

    if line == "/look":
        perceived = session.look()
        if not perceived:
            return Outcome(message="Nothing here has changed since you last looked.")
        return Outcome(perceived=perceived)

    if line.startswith("/go "):
        destination = line[len("/go ") :].strip()
        try:
            perceived = session.move(destination)
        except ValueError:
            return Outcome(message=f"There is no {destination} to go to.")
        return Outcome(
            perceived=perceived,
            message=f"You are in {session.world.room_name(session.here())}.",
        )

    return Outcome(perceived=session.say(line))
