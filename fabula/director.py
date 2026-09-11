"""The director: speaker selection and turn-loop arbitration (spec §7).

Arbitration itself is plain code — argmax over bids — because by the time
bids exist, the model calls that matter (character generation, and any
LLM-assisted ambiguous bid) have already happened. The director's own
model use is reserved for pressure selection and time advancement,
which land in later milestones; M0 has neither.

The director sees every bid, including narrator bids, but only ever
emits a speaker id (or "yield"/"narrate"). Bid rationales stop here —
they are never written into an Event, so they can never reach another
character's projection.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Literal

from fabula.agents import generate_utterance
from fabula.bidding import get_bid, prefilter_candidates
from fabula.classify import classify
from fabula.concurrency import DEFAULT_WORKERS, in_parallel
from fabula.chronology import (
    LARGE_SKIP_MINUTES,
    UNASKED,
    derive_skip_minutes,
    describe_duration,
    due_intentions,
    pending_intentions,
)
from fabula.db import EventStore
from fabula.discovery import invents_a_fact
from fabula.loader import Scene
from fabula.memory import ContextBuilder, form_belief, location_at_seq, record_rehearsals
from fabula.interpret import (
    INTERPRETATION_WINDOW,
    keep_interpretation,
    propose_interpretation,
)
from fabula.persistence import (
    begin_scene,
    close_reached_goals,
    encode_belief,
    witnessed_withholding,
    worth_keeping,
)
from fabula.models import Belief, Bid, Character, Event, Intention, Pressure, ProjectedEvent
from fabula.beats import NOBODY_SPEAKS, Beat
from fabula.beats import available as available_beats
from fabula.narrator import NARRATOR_ID, Narrator, as_bid
from fabula import situations
from fabula.pressures import scene_state, select_pressure
from fabula.world import World, resolve_perception, mentions_fact

DecisionKind = Literal["speak", "narrate", "yield_to_user", "fire_pressure", "withhold"]

YIELD_FLOOR = 0.12

# Long enough that saying it twice in one scene cannot be a coincidence.
REPEAT_WORDS = 6

# A line is an echo of an earlier one if it starts the same way, or if it
# is mostly the same words again. Word-for-word was the whole rule and it
# is not enough: measured against Qwen2.5-3B over fourteen player lines,
# Maria announced the same intention eight times —
#
#   I'll start the letter sorting then. / I'll begin with the letters
#   then. / I'll start with the ones that seem urgent. / I'll start with
#   the oldest letters first. / I'll start by checking the oldest ones,
#   Elena. / I'll start with the oldest letters, Elena.
#
# — and the exact-match guard caught none of them, because no two are
# identical. Paraphrase, not repetition, is how a small model loops.
#
# Both numbers were fitted to that transcript and to a set of genuinely
# different lines that share a topic word, which is the false positive
# worth avoiding: a scene about letters has everybody saying "letters",
# and that is a conversation rather than a loop.
ECHO_OPENING = 3      # words a line may share with its own predecessor's start
ECHO_SHARE = 0.5      # of the shorter line's content words
ECHO_FLOOR = 3        # content words before overlap is worth measuring
ECHO_SHARED = 2       # and at least this many held in common
ECHO_BACK = 6         # how many of their own recent lines to look at

# How long a character holds a secret they are keeping before the engine
# stops holding them back.
#
# Measured against Qwen2.5-3B on `ashgrove/the_dinner`: Tomás, who is the
# only person alive who knows he broke it, answered the first line of the
# scene with "I heard the music box move." His prompt contains the words
# he is protecting — it has to, or he cannot behave as somebody keeping
# them — and a small model cannot leave a salient token alone. On the
# sandbox that costs a scene; on `the_reckoning`, which is over the
# moment those words are said out loud, it ends the story on turn one.
#
# Not a permanent block, because in ashgrove *nobody else knows*: a rule
# that stopped him ever saying it would make that arc unreachable, and
# somebody finally cracking is the thing the scene is built to earn. So
# it is a floor and not a ceiling — in log positions, the unit the
# authored triggers already use, and roughly the point at which
# ashgrove's own pressures start firing.
HOLDS_FOR = 12

# Words that carry no subject. English plus the shipped worlds' other
# language, which is the honest scope: the opening-phrase rule needs no
# vocabulary at all, so a language this list does not cover is judged by
# that alone rather than judged wrongly.
EMPTY_WORDS = frozenset("""
a an the and or but if then so as at by for from in into of on to with not no nor
i im ill ive id you youre your he she it its we they them us our their his her my mine
this that these those there here what who when where how why just about now too very
is am are was were be been being do does did done have has had will would shall should
can could may might must one ones thing things some any all other another same than own
o os as um uma uns umas de do da dos das em no na nos nas por para com sem que se
nao sim eu tu ele ela nos eles elas meu minha teu tua seu sua isso isto aquilo
""".split())


def _fold(text: str) -> str:
    """Strip diacritics and lowercase, so "Tomas" matches "Tomás"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def detect_addressed_to(content: str, characters: dict[str, Character], exclude_id: str | None) -> list[str]:
    """Deterministic, text-only detection of who a line names by first
    name — this feeds turn-taking (spec §7 step 3's "addressed by name"
    criterion), not knowledge filtering, so it carries no invariant-1 risk."""
    folded = _fold(content)
    addressed = []
    for character_id, character in characters.items():
        if character_id == exclude_id:
            continue
        first_name = _fold(character.name.split()[0])
        if re.search(rf"\b{re.escape(first_name)}\b", folded):
            addressed.append(character_id)
    return addressed


@dataclass
class Decision:
    kind: DecisionKind
    character_id: str | None = None
    pressure: Pressure | None = None
    bids: list[Bid] = field(default_factory=list)


def arbitrate(
    bids: list[Bid],
    narrator_bid: Bid | None,
    consecutive_agent_turns: int,
    max_consecutive_agent_turns: int,
    pressure_choice: tuple[Pressure, float] | None = None,
) -> Decision:
    """Guard first: a hard cap on consecutive non-user turns, so
    agent-to-agent exchange always terminates and the user is never
    starved out of their own scene (spec §7 guards). A pressure never
    overrides that guard — the user's turn outranks the drama."""
    if consecutive_agent_turns >= max_consecutive_agent_turns:
        return Decision(kind="yield_to_user", bids=bids)

    contenders = list(bids)
    if narrator_bid is not None:
        contenders.append(narrator_bid)

    best_desire = max((b.desire for b in contenders), default=0.0)

    if pressure_choice is not None and pressure_choice[1] >= max(best_desire, YIELD_FLOOR):
        return Decision(kind="fire_pressure", pressure=pressure_choice[0], bids=bids)

    if not contenders:
        return Decision(kind="yield_to_user", bids=bids)

    winner = max(contenders, key=lambda b: b.desire)
    if winner.desire < YIELD_FLOOR:
        return Decision(kind="yield_to_user", bids=bids)
    if winner.character_id == NARRATOR_ID:
        return Decision(kind="narrate", character_id=NARRATOR_ID, bids=bids)
    if winner.kind == "withhold":
        return Decision(kind="withhold", character_id=winner.character_id, bids=bids)
    return Decision(kind="speak", character_id=winner.character_id, bids=bids)


class Director:
    def __init__(
        self,
        store: EventStore,
        world: World,
        characters: dict[str, Character],
        scene: Scene,
        narrator: Narrator,
        llm=None,
        pressures: list[Pressure] | None = None,
        interpret_beliefs: bool = True,
        waiting: dict[str, Character] | None = None,
        classify_reports: bool = True,
        workers: int = DEFAULT_WORKERS,
        direct_beats: bool = True,
        open_ended: bool = False,
    ):
        self.store = store
        self.world = world
        self.characters = characters
        self.scene = scene
        self.narrator = narrator
        self.llm = llm
        # Reading each remembered moment is one model call per character
        # per moment, and it dominates the bill for a scene. Turning it
        # off costs the annotations and nothing that was provable.
        self.interpret_beliefs = interpret_beliefs
        # Reading a line for what it reports costs a model call, but only
        # for a line that names one of the world's facts — almost none do.
        self.classify_reports = classify_reports
        # Whether the model gets to choose which beat the narrator
        # writes. One call per narration, and only when there is more
        # than one thing the moment could be; off, the first of the
        # offered beats is taken, which is the deterministic order.
        self.direct_beats = direct_beats
        # Played to stay in rather than to finish: no ending is reported,
        # no seam is crossed, and when the authored pressures run out the
        # engine writes the next thing that happens instead of the story
        # going quiet for good.
        self.open_ended = open_ended
        # What the player has asked for, if anything. Reaches the
        # situation writer and the narrator and nobody else — see
        # `fabula.steering`.
        self.steering = ""
        # How many of the independent calls in a turn — the bids, the
        # readings — may be in flight at once. Somebody else's endpoint
        # is on the other end of them; 1 is the old sequential engine.
        self.workers = workers
        self.pressures = pressures or []
        # Written for this world, not in the room when it opened. An
        # authored pressure may bring one on; nothing else can, and the
        # narrator least of all — it writes prose, and a person who
        # arrives has to arrive as an event.
        self.waiting = waiting or {}
        self.contexts = ContextBuilder(world, characters, scene.id, store, llm)

    def build_event(
        self,
        kind: str,
        actor_id: str | None,
        location_id: str,
        content: str,
        audibility: str = "room",
        metadata: dict | None = None,
        story_time: datetime | None = None,
        detail_level: str = "full",
        addressed_to: list[str] | None = None,
    ) -> Event:
        all_events = self.store.get_events(self.scene.id)
        if story_time is None:
            story_time = all_events[-1].story_time if all_events else self.scene.start_time
        if addressed_to is None:
            addressed_to = (
                detect_addressed_to(content, self.characters, exclude_id=actor_id)
                if kind in ("utterance", "action")
                else []
            )
        return Event(
            scene_id=self.scene.id,
            seq=self.store.next_seq(self.scene.id),
            story_time=story_time,
            kind=kind,
            actor_id=actor_id,
            location_id=location_id,
            content=content,
            audibility=audibility,
            addressed_to=addressed_to,
            metadata=metadata or {},
            detail_level=detail_level,
        )

    def run_turn(self, user_event: Event) -> list[Event]:
        """Runs spec §7 steps 2-8 for one user input. `user_event` must
        already be constructed (kind/location/etc set by the caller); this
        appends it and then loops until yield or turn-budget exhaustion."""
        stored_user_event = self.store.append_event(user_event)
        self._absorb()
        events_this_turn = [stored_user_event]
        reported = self.record_transmission(stored_user_event)
        if reported is not None:
            events_this_turn.append(reported)
            self._absorb()
        return events_this_turn + self._loop(stored_user_event, must_answer=True)

    def open_turn(self, opening: Event) -> list[Event]:
        """Let the room have the first word.

        A story that waits for the player to speak first puts the whole
        burden of starting it on them: you arrive somewhere, nobody says
        anything, and the only way to find out you are not alone is to
        talk to the air. Whoever is standing there bids on the scene's
        opening exactly as they would on a line — so somebody who has
        nothing to say still says nothing, and it costs one bid each,
        once, for the people actually in the room with you.

        One beat, and no pressures. A greeting is the room noticing you;
        a pressure is the director escalating, and a story whose first
        move is its own complication has started without you. The budget
        of one says the same thing from the other side: an opening is a
        hello, not a conversation you were not in.
        """
        return self._loop(opening, budget=1, pressures=False)

    def _loop(
        self,
        last_event: Event,
        budget: int | None = None,
        pressures: bool = True,
        must_answer: bool = False,
    ) -> list[Event]:
        """Bid, arbitrate, act, repeat — until somebody yields to the
        player or the scene runs out of budget."""
        events_this_turn: list[Event] = []
        consecutive_agent_turns = 0
        answered = False
        # How little anybody wanted the turn, at its quietest. Read after
        # the loop by `_meanwhile`, which needs to know whether the room
        # was going anywhere before it lets half an hour of the evening
        # go by.
        quietest = 1.0

        # Counted down on beats rather than on iterations. Time moving is
        # not the room taking a beat — it is the reason the room has
        # nothing to say — so spending one on it made a quiet turn
        # quieter, which is backwards.
        beats = budget if budget is not None else self.scene.turn_budget
        while beats > 0:
            all_events = self.store.get_events(self.scene.id)
            candidates = prefilter_candidates(last_event, self.characters, all_events, self.world)

            # Every bid, the narrator's included, goes out at once.
            # Nobody's bid can see anybody else's — that is what makes
            # these separate agents — so making them queue was pure
            # waiting, and the fuller the room the longer it got.
            def bid_for(character: Character) -> Callable[[], Bid]:
                location = location_at_seq(
                    character.id, character.location_id, all_events, last_event.seq + 1
                )
                level = resolve_perception(last_event, character.id, location, self.world)
                return lambda: get_bid(
                    character, last_event, level, all_events, self.contexts, self.llm
                )

            # What the room could use, decided here rather than left to
            # the narrator to infer from the last line. The director is
            # the only omniscient component, so what it hands over is one
            # id from a closed set plus something anybody standing there
            # can already see — never a sentence, and never anything read
            # from beliefs or trust.
            offered = available_beats(
                last_event,
                all_events,
                self.world,
                self.characters,
                protagonist_id=self.protagonist_id(),
                alone=not candidates,
            )
            beat = offered[0] if offered else None
            bids: list[Bid] = in_parallel(
                [bid_for(character) for character in candidates], self.workers
            )
            narrator_bid = as_bid(beat)
            top_bid = max((b.desire for b in bids), default=0.0)
            quietest = min(quietest, top_bid)

            beats -= 1
            pressure_choice = (
                select_pressure(
                    self.pressures,
                    scene_state(all_events, self.characters, self.world),
                    self.world.facts,
                    # An arc ramps its pressures harder the longer it runs,
                    # because it is climbing toward an ending. Played
                    # open-ended there is no ending to climb toward, so the
                    # ramp would only get louder forever: hold equilibrium
                    # instead, which is what sandbox already means.
                    "sandbox" if self.open_ended else self.scene.mode,
                    top_bid,
                )
                if pressures
                else None
            )
            if pressures and pressure_choice is None and self.open_ended:
                pressure_choice = self._something_happens(all_events, last_event, top_bid)
            decision = arbitrate(
                bids,
                narrator_bid,
                consecutive_agent_turns,
                self.scene.max_consecutive_agent_turns,
                pressure_choice,
            )

            if decision.kind == "yield_to_user":
                # A turn the player perceives nothing of is the worst
                # answer the app can give: they say something into a room
                # with somebody standing in it and get "No one answers."
                # Nobody bid and no beat was due — so the room takes the
                # turn rather than nobody having it.
                #
                # *Perceived*, not merely appended: a pressure firing two
                # rooms away is the story moving, and it is still silence
                # where the player is standing.
                if not must_answer or answered:
                    break
                anchor = self._heard_last(all_events)
                if anchor is None:
                    break
                last_event, beat = anchor
                decision = Decision(kind="narrate", character_id=NARRATOR_ID, bids=bids)

            if decision.kind == "fire_pressure":
                new_event = self._fire(decision.pressure)
            elif decision.kind == "withhold":
                character = self.characters[decision.character_id]
                location = location_at_seq(
                    character.id, character.location_id, all_events, last_event.seq + 1
                )
                # Not answering is something the room can see, so it is an
                # ordinary event with an actor — visible, attributable, and
                # perception-filtered like anything else.
                new_event = self.build_event(
                    "action",
                    character.id,
                    location,
                    self.narrator.render_withholding(character, self.world, location),
                    metadata={"withheld": True},
                )
            elif decision.kind == "narrate":
                # Who is standing there, and never the player among
                # them: naming them is inviting the one thing the
                # narrator must not do.
                here = [
                    character.name
                    for character in self.characters.values()
                    if not character.is_user
                    and location_at_seq(
                        character.id, character.location_id, all_events, last_event.seq + 1
                    )
                    == last_event.location_id
                ]
                # Which of the moments this could be is a judgement, not
                # a rule — the room has gone quiet *and* somebody has
                # stopped talking *and* there is a letter nobody has
                # picked up. Asked only now, once the narrator has won
                # the turn, so the call is paid for by a narration that
                # is definitely being written.
                beat = self.pick_beat(offered, last_event, here) or beat
                content = self.narrator.generate(
                    last_event, all_events, self.world, beat=beat, present=here
                )
                named = invents_a_fact(content, self.world) or self._plays_the_player(content)
                if named:
                    # A narration that names a fact raises the subject in
                    # front of everybody in the room — it satisfies
                    # `fact_spoken` and can end an arc that was waiting
                    # for somebody to say it out loud. Nobody said it, so
                    # the beat is dropped and the turn goes on.
                    #
                    # And one that names the player is playing the one
                    # character somebody else is holding. The prompt has
                    # said not to since M0 and a small model does it
                    # anyway — measured on a 1.5B: "Elena's finger
                    # brushed against the dusty glass of a photo album",
                    # which Elena never did.
                    consecutive_agent_turns += 1
                    continue
                new_event = self.build_event("narration", None, last_event.location_id, content)
            else:
                character = self.characters[decision.character_id]
                location = location_at_seq(
                    character.id, character.location_id, all_events, last_event.seq + 1
                )
                content = generate_utterance(character, all_events, self.contexts, self.llm)
                if self._gives_away(character.id, content, all_events):
                    # One more try, in case it was the salient token and
                    # not the character. If they reach for it again this
                    # early, they visibly do not answer instead — which
                    # is a thing the room can see, and truer to somebody
                    # holding on than a line they would not have said.
                    content = generate_utterance(
                        character, all_events, self.contexts, self.llm
                    )
                    if self._gives_away(character.id, content, all_events):
                        new_event = self.build_event(
                            "action",
                            character.id,
                            location,
                            self.narrator.render_withholding(
                                character, self.world, location
                            ),
                            metadata={"withheld": True, "held_back": True},
                        )
                        last_event = self.store.append_event(new_event)
                        events_this_turn.append(last_event)
                        answered = answered or self._reaches_user(last_event)
                        self._absorb()
                        consecutive_agent_turns += 1
                        continue
                if self._already_said(character.id, content, all_events):
                    # Word for word what they last said. Their own lines
                    # are in the context they were given, and a small
                    # model repeats them anyway — measured on a 1.5B,
                    # Maria said one sentence twice inside four lines,
                    # which reads as the app being broken rather than as
                    # a character insisting. One more try, and if it
                    # comes back the same they say nothing this beat.
                    content = generate_utterance(
                        character, all_events, self.contexts, self.llm
                    )
                    if self._already_said(character.id, content, all_events):
                        consecutive_agent_turns += 1
                        continue
                new_event = self.build_event("utterance", character.id, location, content)

            last_event = self.store.append_event(new_event)
            events_this_turn.append(last_event)
            answered = answered or self._reaches_user(last_event)
            self._absorb()
            # Anybody can report having told somebody something, not only
            # the player.
            reported = self.record_transmission(last_event)
            if reported is not None:
                events_this_turn.append(reported)
                self._absorb()
            consecutive_agent_turns += 1

            if self._addresses_user(last_event):
                # Someone spoke to the player. Carrying on past that lets
                # another character answer a question the player was
                # asked, which is the sharpest way to make them a
                # spectator in their own scene — the turn budget alone
                # does not prevent it, because the loop simply had turns
                # left. The floor is theirs.
                break

        # And then the house, once the room has finished. After the loop
        # rather than inside it, which is not a detail: anywhere in the
        # middle it either costs the room the beat it was about to take
        # or buys the director an extra round to fire a pressure in, and
        # both were measured. Out here it changes nothing about the turn
        # that just happened — it is the next thing the player finds when
        # they look up.
        if pressures:
            # The whole log, not this turn's slice: both halves of the
            # quiet measure count backwards from the last time this fired.
            events_this_turn.extend(
                self._meanwhile(self.store.get_events(self.scene.id), quietest)
            )

        return events_this_turn

    def pick_beat(
        self, offered: list[Beat], last_event: Event, here: list[str]
    ) -> Beat | None:
        """Which of the available beats this moment wants.

        The model proposes and the engine disposes, exactly as everywhere
        else it is asked anything: it answers with one id, the id has to
        be one of the ids offered, and anything else falls back to the
        deterministic first choice. It cannot invent a beat, cannot write
        an instruction, and cannot reach past the closed vocabulary.

        What it is shown is what the room can see — the same material the
        narrator gets. The director is omniscient; there is no reason to
        hand any of that to something whose whole job is picking between
        three labels.
        """
        if self.llm is None or not self.direct_beats or len(offered) < 2:
            return None
        options = "\n".join(f"- {beat.id}: {beat.label}" for beat in offered)
        system = (
            "You direct a scene in an interactive story. You are choosing what the "
            "narrator should give a beat to next — not writing it. Answer with one id "
            "from the list and nothing else."
        )
        prompt = (
            f"Location: {self.world.room_name(last_event.location_id)}\n"
            + (f"Who is here: {', '.join(here)}\n" if here else "")
            + f"The last thing that happened ({last_event.kind}): {last_event.content}\n\n"
            f"What could take the beat:\n{options}\n\n"
            "Which one does this moment want? Answer with the id alone."
        )
        answer = self.llm.complete(system=system, prompt=prompt, key="beat")
        wanted = (answer or "").strip().strip(".\"'`").lower()
        for beat in offered:
            if beat.id == wanted:
                return beat
        # A model that answered with a sentence, or with a beat nobody
        # offered, has said nothing. The scene takes the first one.
        return None

    def _plays_the_player(self, content: str) -> str | None:
        """Does this narration have the player doing something?

        Their name, on a word boundary and diacritics folded, is enough:
        the narrator is told never to describe them at all, so any
        mention is the rule being broken rather than a borderline case.
        Deterministic, because the prompt-level version of this rule is
        the one a small model ignores most reliably.
        """
        user = next((c for c in self.characters.values() if c.is_user), None)
        if user is None:
            return None
        folded = _fold(content)
        for name in {user.name, user.name.split()[0]}:
            if re.search(rf"(?<!\w){re.escape(_fold(name))}(?!\w)", folded):
                return f"names {user.name}"
        return None

    def _already_said(self, character_id: str, content: str, all_events: list[Event]) -> bool:
        """Is this something this character has already said?

        Three rules, cheapest first, all deterministic — the prompt-level
        version of "do not repeat yourself" is the one a small model
        ignores most reliably, and their own lines are already in the
        context they were given when they wrote this.

        1. **Word for word**, once punctuation and spacing are set aside:
           their own last line, the one just said in front of them (a
           small model parrots the previous speaker as readily as
           itself), or anything of their own this scene once the line is
           long enough that saying it twice cannot be a coincidence.
        2. **The same opening.** People do not start consecutive
           sentences identically; a model announcing what it is about to
           do does exactly that.
        3. **Mostly the same words.** Set against the shorter line, so a
           long restatement of a short line still counts, with a floor on
           both, because one shared word between two four-word lines is a
           coincidence and not a loop.

        Short lines are left alone by 2 and 3: "No." twice is a person.
        """
        def bare(text: str) -> list[str]:
            return "".join(c for c in text.lower() if c.isalnum() or c.isspace()).split()

        def subject(text: str) -> set[str]:
            """What a line is about, as far as this can tell without
            knowing the language: its words, less the ones that carry no
            subject and less everybody's name, since a vocative is not
            what a line is about."""
            called = {
                part.lower()
                for person in self.characters.values()
                for part in _fold(person.name).split()
            }
            return {
                word for word in bare(text)
                if len(word) > 2 and word not in EMPTY_WORDS and word not in called
            }

        words = bare(content)
        spoken = [e for e in all_events if e.kind == "utterance"]
        mine = [e for e in spoken if e.actor_id == character_id]

        # Word for word: their own last line and the one just said in
        # front of them at any length, and — once the line is long enough
        # that saying it twice cannot be a coincidence — anything of
        # their own this scene and anything recently said in the room.
        # Measured on a 3B: Tomás said "I'll pour the last of this tea"
        # and Maria said it back, identically, one turn later with the
        # player's line in between, which checking only the previous
        # utterance walked straight past. "No." twice is still a person.
        against = [e.content for e in (mine[-1:] + spoken[-1:])]
        if len(words) > REPEAT_WORDS:
            against += [e.content for e in (mine + spoken[-ECHO_BACK:])]
        if any(bare(earlier) == words for earlier in against):
            return True

        now = subject(content)
        for earlier in [e.content for e in mine[-ECHO_BACK:]]:
            if len(words) >= ECHO_OPENING and words[:ECHO_OPENING] == bare(earlier)[:ECHO_OPENING]:
                return True
            was = subject(earlier)
            if len(now) < ECHO_FLOOR or len(was) < ECHO_FLOOR:
                continue
            common = now & was
            if len(common) >= ECHO_SHARED and len(common) / min(len(now), len(was)) >= ECHO_SHARE:
                return True
        return False

    def _gives_away(self, character_id: str, content: str, all_events: list[Event]) -> bool:
        """Is this character handing over the thing they are keeping,
        before the scene has given them any reason to?

        `protects` is authored and the keywords are exact, so this is the
        same deterministic shape as the narrator's `invents_a_fact` — and
        it is needed for the same reason. The prompt tells them what they
        are sitting on and tells them not to raise it; that rule holds on
        a capable model and does not hold on a small one, and prompts
        that only sometimes hold are not what this engine rests on.

        It expires, deliberately. A secret nobody can ever say is not a
        secret, it is a locked door: in ashgrove nobody but Tomás knows,
        so a permanent rule would make `the_reckoning` unwinnable. Once
        the scene has actually run, or once somebody else has raised the
        subject, he is on his own.
        """
        character = self.characters.get(character_id)
        if character is None or character.is_user or not character.protects:
            return False
        turn = all_events[-1].seq if all_events else 0
        if turn > HOLDS_FOR:
            return False
        for fact_id in character.protects:
            fact = self.world.facts.get(fact_id)
            if fact is None or not mentions_fact(fact, content):
                continue
            # Already out, by somebody. Holding them to it now would be
            # the engine keeping a secret the room has heard.
            if any(
                mentions_fact(fact, event.content)
                for event in all_events
                if event.actor_id != character_id
            ):
                continue
            return True
        return False

    def _reaches_user(self, event: Event) -> bool:
        """Did the player perceive any of that, at any fidelity?"""
        user = next((c for c in self.characters.values() if c.is_user), None)
        if user is None:
            return False
        where = self.current_location(user)
        return resolve_perception(event, user.id, where, self.world) != "none"

    def _heard_last(self, all_events: list[Event]) -> tuple[Event, Beat] | None:
        """Something to hang a beat on, in the room the player is in.

        The anchor has to be a line they actually heard: `last_event` may
        be two rooms away, and putting it in the prompt would narrate
        their room out of words they never perceived. With nothing heard
        yet there is still the room itself, which needs no anchor at all.
        """
        user = next((c for c in self.characters.values() if c.is_user), None)
        if user is None:
            return None
        where = self.current_location(user)
        heard = [
            event
            for event in all_events
            if event.location_id == where
            and resolve_perception(event, user.id, where, self.world) == "full"
        ]
        if not heard:
            return None
        return heard[-1], Beat(NOBODY_SPEAKS, 1.0, "nobody had anything to say")

    def protagonist_id(self) -> str | None:
        return next((c.id for c in self.characters.values() if c.is_user), None)

    def _addresses_user(self, event: Event) -> bool:
        user = next((c for c in self.characters.values() if c.is_user), None)
        return bool(user and event.actor_id != user.id and user.id in event.addressed_to)

    def advance_time(
        self,
        minutes: int | None = None,
        consent: Callable[[int], bool] | None = None,
        unasked: bool = False,
    ) -> list[Event]:
        """Skip forward to the next moment something is ready, log the
        skip, and resolve what happened off screen while it elapsed.

        Returns the events appended, empty if nothing was pending or the
        user declined a large skip.

        `unasked` marks a skip the engine decided on rather than one the
        player asked for — see `_meanwhile`. It changes nothing about how
        the skip is resolved or perceived; it is a note in the log, so
        the drift measure can tell that something happened and the same
        stretch of quiet does not keep buying the same jump.
        """
        all_events = self.store.get_events(self.scene.id)
        if minutes is None:
            minutes = derive_skip_minutes(self.characters, all_events, blocked=self.unreachable)
        if minutes is None or minutes <= 0:
            return []

        # The user is a character with agency: time is not something they
        # should ever lose without noticing (spec §8).
        if minutes > LARGE_SKIP_MINUTES and not (consent and consent(minutes)):
            return []

        previous_time = all_events[-1].story_time if all_events else self.scene.start_time
        user_location = next(
            (c.location_id for c in self.characters.values() if c.is_user),
            next(iter(self.world.rooms)),
        )
        skip = self.build_event(
            "time_skip",
            None,
            user_location,
            f"{describe_duration(minutes, self.world.phrasing)} pass.",
            audibility="building",
            metadata={"minutes": minutes, **({UNASKED: True} if unasked else {})},
            story_time=previous_time + timedelta(minutes=minutes),
        )
        appended = [self.store.append_event(skip)]
        appended.extend(self._resolve_offscreen(minutes, unasked=unasked))
        self._absorb()
        return appended

    def establish(self, where: str) -> Event | None:
        """The line that opens a scene.

        A story that begins with a blank prompt is a text box, not a
        story: the player is told the name of the room and left to guess
        what is in it. The narrator has always had a 0.9 "establish the
        scene" bid for exactly this and it could never fire — by the time
        anything is bid on, the player has already spoken and the log is
        no longer empty. So the opening is not bid on at all; it is the
        one narration the scene owes the player before they type.

        Nothing if the scene has already started, so resuming a story
        does not re-describe a room somebody is standing in the middle of.
        """
        # Anything but the scene's own opening words, which come first
        # and are addressed to the player rather than to the room.
        if [e for e in self.store.get_events(self.scene.id) if not e.metadata.get("opening")]:
            return None
        content = self.narrator.describe_place(where, self.world)
        named = invents_a_fact(content, self.world)
        if named:
            # An opening line that names a secret hands it to everyone in
            # the room before anyone has spoken — and can end an arc on
            # the first beat. Fall back to what the author wrote, and to
            # silence if that names one too.
            authored = self.world.rooms[where].description if where in self.world.rooms else ""
            if not authored or invents_a_fact(authored, self.world):
                return None
            content = authored
        # Not absorbed: the curtain going up is not something that
        # happened to anybody. Everyone in the room reads it in their
        # context like any other line, but nobody forms a durable memory
        # of what the room they are standing in looks like — which would
        # otherwise cost a reading per character before the player has
        # typed anything.
        return self.store.append_event(
            self.build_event("narration", None, where, content)
        )

    def brief(self, who: Character, where: str) -> Event | None:
        """The scene's own first words, to the player and nobody else.

        Every other platform on this shelf opens with one: a paragraph
        that says what you have walked into, so the first thing asked of
        somebody is not "what do you say" to a room they know nothing
        about. Authored rather than generated — the same every time, free,
        and good prose instead of whatever a model made of a room name.

        Private, through the ordinary perception path: `audibility:
        private` addressed to the player means the choke point every leak
        test covers already returns "none" for everybody else. That is
        what lets it be written in the second person, and lets it say
        what only this character would know coming in.
        """
        if not self.scene.opening.strip() or self.store.get_events(self.scene.id):
            return None
        return self.store.append_event(
            self.build_event(
                "narration",
                None,
                where,
                self.scene.opening.strip(),
                audibility="private",
                addressed_to=[who.id],
                metadata={"opening": True},
            )
        )

    def introduce(self, where: str, look: str) -> Event | None:
        """What the room can see of the player, in the player's own words.

        An ordinary event, so it is filtered like one: the people in the
        room perceive it and the people elsewhere never do. It is not
        world truth and nothing checks it against any — the same as
        anything anybody says about themselves — but it is *perceivable*,
        which is what makes it something the room can answer.

        A description that names one of the world's own facts is refused:
        it would hand a secret to everybody standing there before a word
        was spoken, and could end an arc on its first beat.
        """
        if not look.strip() or invents_a_fact(look, self.world):
            return None
        return self.store.append_event(self.build_event("narration", None, where, look.strip()))

    def current_location(self, character: Character) -> str:
        """Where a character is now, replayed from their arrivals in the
        log rather than read off the scene's starting snapshot."""
        all_events = self.store.get_events(self.scene.id)
        last_seq = all_events[-1].seq if all_events else 0
        return location_at_seq(character.id, character.location_id, all_events, last_seq + 1)

    def _resolve_offscreen(self, elapsed_minutes: int, unasked: bool = False) -> list[Event]:
        """Off-screen action is resolved coarsely: elapsed time produces a
        few `detail_level: "summary"` events per character. Specifics are
        materialized lazily, when someone is actually there to see them.

        Only genuinely off-screen characters are resolved this way. Someone
        standing in the room with the user is not off screen, and flattening
        what they did into a coarse stub would both under-describe it and
        hand the user a summary-resolution line as if they had watched it
        happen. Their intention simply stays pending.
        """
        appended = []
        for character in self.characters.values():
            if character.is_user:
                continue
            events = self.store.get_events(self.scene.id)
            for intention in due_intentions(character, events, elapsed_minutes):
                # One question, asked in one place, so the derivation and
                # the resolution cannot disagree about what is possible:
                # are they there, is the player there, and is anybody
                # watching something they would only do alone.
                if self.unreachable(character, intention):
                    continue
                summary = self.build_event(
                    "action",
                    character.id,
                    intention.location_id,
                    f"{character.name} {intention.description}",
                    metadata={
                        "intention_id": intention.id,
                        **({UNASKED: True} if unasked else {}),
                    },
                    detail_level="summary",
                )
                appended.append(self.store.append_event(summary))
                if intention.state:
                    # A state is a fact about them, not a coarse stub of
                    # something they did, so it is logged in full and
                    # separately — `was_asleep` reads these, and a summary
                    # that later materialises must not change the answer.
                    appended.append(
                        self.store.append_event(
                            self.build_event(
                                "state_change",
                                character.id,
                                intention.location_id,
                                f"{character.name} is {intention.state}.",
                                metadata={
                                    "character_id": character.id,
                                    "state": intention.state,
                                    "intention_id": intention.id,
                                    **({UNASKED: True} if unasked else {}),
                                },
                            )
                        )
                    )
        return appended

    def unreachable(self, character: Character, intention: Intention) -> bool:
        """Would resolution refuse this intention as things stand?

        The same conditions `_resolve_offscreen` applies, so the skip
        derivation cannot propose a jump that resolution will decline.

        Three of them, and the first was missing. An intention names the
        room it happens in, and somebody standing in the kitchen cannot
        finish the letters in the study — but resolution only ever
        checked where the *character* was against where the *user* was,
        so Maria sorted the letters in a room she was not in. Worse, and
        the way it was found: with the player sitting in the study, it
        put her doing it in front of them.
        """
        here = self.current_location(character)
        if here != intention.location_id:
            # Not there to do it. `_steps_out` is what changes that.
            return True
        user = next((c for c in self.characters.values() if c.is_user), None)
        if user is not None and here == self.current_location(user):
            # Watched is not off screen. Resolving it here would hand the
            # player a coarse summary of something they were standing in
            # the middle of.
            return True
        return bool(intention.private and self._others_present(here, character.id))

    def pending_skip(self) -> int | None:
        """How long the next skip would be, counting only what could
        actually come of it."""
        return derive_skip_minutes(
            self.characters, self.store.get_events(self.scene.id), blocked=self.unreachable
        )

    def _company(self, location_id: str) -> int:
        """How many people who are not the player are in this room."""
        return sum(
            1
            for character in self.characters.values()
            if not character.is_user and self.current_location(character) == location_id
        )

    def _others_present(self, location_id: str, actor_id: str) -> bool:
        """Is anyone but the actor standing in this room right now?"""
        return any(
            character.id != actor_id and self.current_location(character) == location_id
            for character in self.characters.values()
        )

    def materialize(self, summary_event: Event, observer: Character) -> Event | None:
        """Expand a coarsely-resolved off-screen event into specifics — the
        moment the user finds the kitchen ransacked is when "Tomás searched
        the house" becomes a description.

        Only from inside the room: standing elsewhere, there is nothing to
        see, so there is nothing to materialize. Appends rather than
        rewrites, because the log is append-only; the specifics are a new
        perceivable event, filtered like any other.
        """
        all_events = self.store.get_events(self.scene.id)
        observer_location = location_at_seq(
            observer.id, observer.location_id, all_events, all_events[-1].seq + 1
        )
        if summary_event.location_id != observer_location:
            return None
        if summary_event.detail_level != "summary":
            return None
        if any(e.metadata.get("materializes") == summary_event.id for e in all_events):
            return None

        content = self.narrator.materialize(summary_event, self.world)
        event = self.build_event(
            "narration",
            None,
            summary_event.location_id,
            content,
            metadata={"materializes": summary_event.id},
        )
        appended = self.store.append_event(event)
        self._absorb()
        return appended

    def unmaterialized_here(self, observer: Character) -> list[Event]:
        """Coarse off-screen events waiting in the observer's room.

        An empty log is a real case, not an edge one: looking around is
        the first thing many players do, and before that there is nothing
        to look past. This used to index the last event of an empty list
        and crash — in every world, since M4 — because every test spoke
        before it looked.
        """
        all_events = self.store.get_events(self.scene.id)
        if not all_events:
            return []
        location = location_at_seq(
            observer.id, observer.location_id, all_events, all_events[-1].seq + 1
        )
        done = {e.metadata.get("materializes") for e in all_events}
        return [
            e
            for e in all_events
            if e.detail_level == "summary" and e.location_id == location and e.id not in done
        ]

    def _meanwhile(self, all_events: list[Event], top_bid: float) -> list[Event]:
        """What the people you are not with were going to do anyway.

        Every character can have `intentions` — things they mean to do,
        somewhere else, once enough time has passed — and resolving them
        is the one thing this engine can do that a single model puppeting
        a cast cannot: somebody acts while you are not there, and you
        find out afterwards from what you walk into. All of that was
        built, and none of it ever ran, because `advance_time` was
        reachable from exactly one place: the player pressing "let thirty
        minutes pass", a button whose consequence they have no reason to
        expect. Eleven authored intentions across five worlds, waiting
        for somebody to guess.

        So time moves on its own when the room has gone quiet. Its own
        measure rather than the one that decides whether to *invent* a
        complication: that bar is high because inventing is expensive and
        unauthored, and waiting for the story to go slack before anybody
        may leave the room turned out to be the wrong bar for somebody
        going to do what the author already said they would. Measured at
        the stricter one, `winterlight` — four agents, nine intentions,
        the world this is for — never moved once in fourteen lines.

        Two limits. Only a jump the player would not have been asked
        about: past that, spec §8 says time is not something they lose
        without noticing, and an engine that quietly skipped three hours
        would be taking the scene off them. And only when something
        actually comes of it — `pending_skip` counts only intentions
        resolution would accept, so this never buys a silence.
        """
        if not situations.lull(all_events, self.protagonist_id(), top_bid):
            return []
        # Somebody gets up first. Nothing in this engine ever sent a
        # character out of the room, so the cast converged on the player
        # in the first turn and stood there — which made every intention
        # unreachable for the rest of the scene, because an intention
        # happens somewhere and they were all here.
        appended = self._steps_out(all_events)
        minutes = self.pending_skip()
        if minutes is None or minutes > LARGE_SKIP_MINUTES:
            return appended
        return appended + self.advance_time(minutes, unasked=True)

    def leaves(
        self, character: Character, from_room: str, to_room: str, unasked: bool = False
    ) -> Event:
        """Somebody getting up and going, where they are going *from*.

        `departure` has been one of the engine's event kinds since the
        beginning, with a beat, a narrator instruction and a degraded
        template — "footsteps fading from {location}" — and nothing
        anywhere ever built one. So moving was half an event: the room
        you walked into saw you arrive, and the room you walked out of
        perceived nothing at all. Somebody sitting at the table with you
        did not see you stand up and leave, which is not a perception
        rule, it is a hole in one.

        `audibility: adjacent`, so the room you left sees it and the room
        you are going to hears the footsteps. Nothing positional depends
        on it — `location_at_seq` replays arrivals — so this is purely
        what the room perceived.
        """
        event = self.store.append_event(
            self.build_event(
                "departure",
                character.id,
                from_room,
                f"{character.name} goes through to {self.world.room_name(to_room)}.",
                audibility="adjacent",
                metadata={UNASKED: True} if unasked else {},
            )
        )
        self._absorb()
        return event

    def _steps_out(self, all_events: list[Event]) -> list[Event]:
        """Somebody leaves to go and do the thing they meant to do.

        An `Intention` names the room it happens in, and the author wrote
        that room precisely because it is not the one the scene opens in
        — the letters are in the study, the glue seam is in the kitchen
        after everybody has gone to bed. But there was nothing anywhere
        that made a character *walk* there, so the room they meant to go
        to may as well not have been written.

        One person per turn, and never into the room the player is
        standing in: going somewhere to do a thing in front of them is
        not going somewhere. An ordinary `arrival`, built the same way
        the player's own move is, so the rooms that would hear it hear it
        and the map moves the one way it has ever moved.
        """
        user = next((c for c in self.characters.values() if c.is_user), None)
        user_location = self.current_location(user) if user else None
        for character in self.characters.values():
            if character.is_user:
                continue
            here = self.current_location(character)
            if here == user_location and self._company(here) < 2:
                # The last person in the room with the player does not
                # get to wander off. A house that empties around somebody
                # is not a world carrying on without them, it is a scene
                # being taken away — and a two-hander where the other
                # one leaves is the story ending rather than continuing
                # elsewhere.
                continue
            for intention in pending_intentions(character, all_events):
                there = intention.location_id
                if there in (here, user_location) or there not in self.world.rooms:
                    continue
                if intention.private and self._others_present(there, character.id):
                    # No point going to be alone in a room that is not
                    # empty. It waits, exactly as it waits here.
                    continue
                # Marked like everything else the house does, or the
                # departure would quietly count as the scene taking a
                # turn and ramp every pressure in the world.
                going = self.leaves(character, here, there, unasked=True)
                return [
                    going,
                    self.store.append_event(
                        self.build_event(
                            "arrival",
                            character.id,
                            there,
                            f"{character.name} comes in from {self.world.room_name(here)}.",
                            audibility="adjacent",
                            metadata={UNASKED: True},
                        )
                    ),
                ]
        return []

    def _something_happens(
        self, all_events: list[Event], last_event: Event, top_bid: float
    ) -> tuple[Pressure, float] | None:
        """Write the next thing that happens, when nothing else can.

        Only reached in open-ended play, and only once the authored
        pressures have nothing left to offer — `select_pressure` returned
        None, which in a sandbox means either the room is still generating
        its own tension or every pressure is spent. `drifting` separates
        those two: it asks whether the last several events were all people
        talking.

        A `None` here is ordinary. The turn carries on exactly as it did
        before this existed, so a model that failed or answered with
        nothing costs a beat of atmosphere and not the story.
        """
        if self.llm is None:
            return None
        if not situations.drifting(all_events, self.protagonist_id(), top_bid):
            return None
        # Placed where the player is standing by default. A situation
        # two rooms away is the story moving and still silence where they
        # are, which is the failure the `must_answer` path exists to stop.
        player = next((c for c in self.characters.values() if c.is_user), None)
        made = situations.compose(
            self.llm,
            self.world,
            all_events,
            self.characters,
            self.current_location(player) if player else last_event.location_id,
            steering=self.steering,
        )
        # Scored like any other pressure, so it competes rather than
        # interrupts: somebody with something to say still wins the turn.
        return (made, made.weight) if made else None

    def _fire(self, pressure: Pressure) -> Event:
        """Turn an authored pressure into one ordinary event. The firing is
        recorded in the event's metadata, so cooldown and max_fires derive
        from the log rather than from state held on the side — an unlogged
        firing would be the same class of bug as an unlogged utterance."""
        effect = pressure.effect
        location_id = effect["location"]
        actor_id = effect.get("actor")
        if effect["kind"] == "arrival" and actor_id in self.waiting:
            self.admit(actor_id)
        metadata = {"pressure_id": pressure.id}
        if situations.is_invented(pressure):
            # So `/reveal`, the drift check and anybody reading the log can
            # tell what the author wrote from what the engine did when the
            # author ran out.
            metadata["invented"] = True
        if effect["kind"] == "state_change":
            metadata |= {"character_id": actor_id, "state": effect["state"]}
        content = self.narrator.render_pressure(pressure, location_id, self.world)
        return self.build_event(
            effect["kind"],
            actor_id,
            location_id,
            content,
            audibility=effect.get("audibility", "room"),
            metadata=metadata,
        )

    def record_transmission(self, event: Event) -> Event | None:
        """If this line reports having told somebody something, log that
        telling as the perception it describes.

        A private beat, placed where the recipient is and addressed to
        them, so the existing perception rules do all the work: they get
        it in full, nobody else gets it at all, and the speaker knows
        what they themselves said. From there it is an ordinary perceived
        event — it becomes a belief, it shows in the reveal, and the turn
        it landed in can be taken again.

        Its story time is *now*, not backdated. The log records when
        something entered the story, and a timestamp running backwards
        would walk the clock in every client for no gain; the content and
        the metadata say it happened before tonight.
        """
        speaker = self.characters.get(event.actor_id)
        if speaker is None:
            return None
        all_events = self.store.get_events(self.scene.id)
        reachable = {**self.characters, **self.waiting}
        told = classify(
            event,
            speaker,
            reachable,
            self.world,
            # Everything they had perceived *before* this line. Their own
            # utterance is in their projection the moment it is appended,
            # so including it would let anybody bootstrap knowledge by
            # asserting it: "I already told Maria about the music box"
            # would make the music box theirs to pass on.
            [
                perceived
                for perceived in self.contexts.project(speaker, all_events)
                if perceived.event.seq < event.seq
            ],
            self.store,
            self.llm if self.classify_reports else None,
        )
        if told is None:
            return None

        recipient = reachable[told.recipient_id]
        subject = self.world.facts[told.fact_id].keywords[0]
        return self.store.append_event(
            self.build_event(
                "action",
                speaker.id,
                self.current_location(recipient),
                self.world.phrasing.transmission.format(
                    speaker=speaker.name, recipient=recipient.name, subject=subject
                ),
                audibility="private",
                addressed_to=[recipient.id],
                metadata={"transmission": told.fact_id},
            )
        )

    def admit(self, character_id: str) -> Character:
        """Bring somebody who was waiting off-stage into the scene.

        Only ever called for an arrival, which is what makes their memory
        answer itself: `location_at_seq` replays their own arrivals from
        wherever they started, and they started nowhere reachable — so
        everything before the event that brings them in resolves to "none"
        and they walk in knowing only what they walk in on. No backfill,
        no decision about what they might have overheard, and nothing to
        get wrong.
        """
        character = self.waiting.pop(character_id)
        # In place: the session, the context builder and this object all
        # hold the same dict, and somebody who joined only the director's
        # copy would bid without ever being seen to be in the room.
        self.characters[character_id] = character
        begin_scene(self.store, self.characters)
        return character

    def _absorb(self) -> None:
        """After each append, let every character take in what they just
        perceived: rehearse the earlier moments this one re-mentions,
        encode it as a belief if it was salient enough to keep, count the
        interaction, and move their regard for whoever acted.

        All of it runs per character against their own projection, so an
        event they could not perceive is neither rehearsed, remembered,
        counted nor held against anyone — it does not exist for them.
        """
        all_events = self.store.get_events(self.scene.id)
        if not all_events:
            return
        appended = all_events[-1].seq

        # First, what each character took in, worked out before anything
        # is written down: who perceived the new event at all, and what
        # they would remember of it.
        taking_in: list[tuple[Character, list[ProjectedEvent], Belief | None]] = []
        for character in self.characters.values():
            projected = self.contexts.project(character, all_events)
            record_rehearsals(
                character, projected, self.store, self.world.phrasing.stopwords
            )
            if not projected or projected[-1].event.seq != appended:
                # They did not perceive the event that was just appended,
                # so there is nothing new for them to take in. Without
                # this check an older event stays "newest" for them and
                # gets absorbed again on every append they miss — one
                # withheld beat charging them four times over while the
                # others talk in another room.
                continue
            taking_in.append(
                (character, projected, form_belief(character, projected[-1]))
            )

        def reading_for(
            character: Character, projected: list[ProjectedEvent], belief: Belief | None
        ) -> Callable[[], str | None]:
            """What this character privately made of it — or None for a
            moment nobody is going to be asked to read."""
            # Not for the player. Nothing reads their memory back into a
            # prompt — they are holding it — so writing down what they
            # privately think would be the engine deciding their inner
            # life, and paying a model call to do it.
            reader = None if character.is_user or not self.interpret_beliefs else self.llm
            # And never for a moment about to be discarded: paying to
            # read something below the memory floor is money spent on
            # the weather.
            if reader is None or not worth_keeping(belief):
                return lambda: None
            # The run-up is what makes the moment readable: "he said
            # nothing" means one thing after small talk and another after
            # being asked where the music box went. Their perceived lines
            # only, so the reading cannot see further than they did.
            window = projected[-INTERPRETATION_WINDOW:]
            return lambda: propose_interpretation(
                character, window, self.world, self.store, reader
            )

        # One private reading each, and no character's reading can see
        # another's, so they go out together rather than one after the
        # next — this is the loop that costs a call per remembered moment
        # per character, and it was the whole cast waiting in a queue.
        readings = in_parallel(
            [reading_for(*entry) for entry in taking_in], self.workers
        )

        # Then the writing, back on this thread and in cast order: a turn
        # stays one unit of work, and a parallel turn leaves the store in
        # exactly the state a sequential one would.
        for (character, projected, belief), reading in zip(taking_in, readings):
            if reading is not None:
                keep_interpretation(
                    self.store, character, projected[-INTERPRETATION_WINDOW:], reading
                )
            encode_belief(
                self.store, character, belief, interpret=lambda kept=reading: kept or ""
            )

            newest = projected[-1]
            actor_id = newest.event.actor_id
            if actor_id and actor_id != character.id:
                self.store.bump_interaction(character.id, actor_id)
            # Watching someone refuse to answer is the one thing in the
            # engine that deterministically moves how they are regarded —
            # for everyone but the player. Deciding that Elena believes
            # her brother less tonight is telling the person holding her
            # how they feel, which is the same overreach as narrating
            # their actions for them.
            if not character.is_user:
                witnessed_withholding(self.store, character.id, newest)
            # A goal whose subject they have now heard raised is closed.
            # Judged from their own projection: a secret that came out in
            # a room they were not in has not stopped being a secret to
            # them, and they go on guarding it.
            close_reached_goals(self.store, character, projected, self.world)
