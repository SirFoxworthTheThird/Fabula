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
    # The story moved to another scene. Clients hold a session, so they
    # have to be told to pick up the new one.
    went_on: "Session | None" = None
    perceived: list[ProjectedEvent] = field(default_factory=list)
    # A note from the client to the player — never story content.
    message: str | None = None
    quit: bool = False
    # The scene reached its declared end on this action.
    ended: bool = False
    # This replaces the previous take rather than following it, so a
    # client showing a transcript has to drop the tail before rendering.
    replaced: bool = False
    # The scene was taken back and is shorter than the client's copy of
    # it: everything on screen is now wrong, not just the tail.
    redraw: bool = False


def run_command(
    session: Session, line: str, consent: Callable[[int], bool] | None = None
) -> Outcome:
    played = session.takes_played
    outcome = _dispatch(session, line, consent)
    # Report the ending on the action that caused it, once — and only for
    # an action that actually played a turn. The comparison is against the
    # state that turn started from, which the session records, because a
    # retake rewinds and so it cannot be measured before dispatch.
    if session.takes_played != played:
        outcome.ended = not session.ended_at_turn_start and session.ended()
    return outcome


def _dispatch(
    session: Session, line: str, consent: Callable[[int], bool] | None = None
) -> Outcome:
    say = session.world.phrasing.say
    line = line.strip()
    if not line:
        return Outcome()

    if line in ("/quit", "/exit"):
        return Outcome(quit=True)

    if line == "/wait":
        if session.pending_skip() is None:
            return Outcome(message=say("nothing_pending"))
        perceived = session.wait(consent=consent)
        if not perceived:
            return Outcome(message=say("time_stays"))
        return Outcome(perceived=perceived)

    if line in ("/next", "/on"):
        following = session.go_on()
        if following is None:
            return Outcome(message=say("no_next"))
        return Outcome(
            went_on=following,
            message=say("now_playing", scene=following.scene.id.replace("_", " ")),
        )

    if line.startswith("/back"):
        # How far back, in things the player said: "/back" is the last
        # one, "/back 3" is three ago. Counted in their own lines because
        # that is what a person remembers doing, rather than in log
        # positions, which is what the engine happens to count in.
        rest = line[len("/back"):].strip()
        try:
            how_many = max(1, int(rest)) if rest else 1
        except ValueError:
            return Outcome(message=say("back_how_many"))
        mine = [
            event.seq
            for event in session.store.get_events(session.scene.id)
            if event.actor_id == session.user_character.id and event.kind == "utterance"
        ]
        if len(mine) < how_many:
            return Outcome(message=say("nothing_to_take_back"))
        session.rewind_to(mine[-how_many] - 1)
        return Outcome(message=say("taken_back", lines=how_many), redraw=True)

    if line in ("/again", "/retry"):
        # A director calling "again", not an undo of the story: the take
        # is thrown away and played once more from the same point.
        if not session.can_regenerate():
            return Outcome(message=say("nothing_played"))
        return Outcome(perceived=session.regenerate(), replaced=True)

    if line == "/reveal":
        # A spoiler, and only ever on request. Read-only, so the scene is
        # untouched and play can continue after looking.
        from fabula.reveal import reveal_text

        return Outcome(message=reveal_text(session))

    if line.startswith("/read "):
        wanted = line[len("/read ") :].strip()
        try:
            return Outcome(perceived=session.read(wanted))
        except ValueError:
            return Outcome(message=say("no_such_thing", thing=wanted))

    if line == "/look":
        perceived = session.look()
        here = session.items_here()
        # Looking around says what there is to read: an item nobody can
        # find is an item that may as well not be authored.
        note = (
            say("things_here", things=", ".join(item.name for item in here))
            if here
            else (None if perceived else say("nothing_changed"))
        )
        return Outcome(perceived=perceived, message=note)

    if line.startswith("/go "):
        destination = line[len("/go ") :].strip()
        try:
            perceived = session.move(destination)
        except ValueError:
            return Outcome(message=say("no_such_room", room=destination))
        return Outcome(
            perceived=perceived,
            message=say("now_in", room=session.world.room_name(session.here())),
        )

    perceived = session.say(line)
    if not [p for p in perceived if p.event.actor_id != session.user_character.id]:
        # Speaking to an empty room is a legitimate outcome — everyone may
        # have left, and the player only heard a door. But a client that
        # prints nothing is indistinguishable from one that crashed, so
        # say plainly that the silence is the answer.
        return Outcome(perceived=perceived, message=say("no_answer"))
    return Outcome(perceived=perceived)
