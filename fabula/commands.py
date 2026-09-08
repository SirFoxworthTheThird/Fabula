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
    # The scene reached its declared end on this action.
    ended: bool = False
    # This replaces the previous take rather than following it, so a
    # client showing a transcript has to drop the tail before rendering.
    replaced: bool = False


def run_command(
    session: Session, line: str, consent: Callable[[int], bool] | None = None
) -> Outcome:
    played = session.turns_played
    outcome = _dispatch(session, line, consent)
    # Report the ending on the action that caused it, once — and only for
    # an action that actually played a turn. The comparison is against the
    # state that turn started from, which the session records, because a
    # retake rewinds and so it cannot be measured before dispatch.
    if session.turns_played != played:
        outcome.ended = not session.ended_at_turn_start and session.ended()
    return outcome


def _dispatch(
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

    if line in ("/again", "/retry"):
        # A director calling "again", not an undo of the story: the take
        # is thrown away and played once more from the same point.
        if not session.can_regenerate():
            return Outcome(message="Nothing has been played yet.")
        return Outcome(perceived=session.regenerate(), replaced=True)

    if line == "/reveal":
        # A spoiler, and only ever on request. Read-only, so the scene is
        # untouched and play can continue after looking.
        from fabula.reveal import reveal_text

        return Outcome(message=reveal_text(session))

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

    perceived = session.say(line)
    if not [p for p in perceived if p.event.actor_id != session.user_character.id]:
        # Speaking to an empty room is a legitimate outcome — everyone may
        # have left, and the player only heard a door. But a client that
        # prints nothing is indistinguishable from one that crashed, so
        # say plainly that the silence is the answer.
        return Outcome(perceived=perceived, message="No one answers.")
    return Outcome(perceived=perceived)
