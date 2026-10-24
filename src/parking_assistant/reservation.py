import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

FIELD_ORDER = ("first_name", "last_name", "car_number", "start", "end")
TIME_FORMAT = "%Y-%m-%dT%H:%M"
DISPLAY_FORMAT = "%a %d %b %Y, %H:%M"
MAX_STAY = timedelta(days=30)
MAX_ADVANCE = timedelta(days=90)
START_GRACE = timedelta(minutes=5)

NAME_PATTERN = re.compile(r"[^\W\d_]+(?:[ '\-][^\W\d_]+)*")
PLATE_PATTERN = re.compile(r"[A-Z0-9]{3,10}")

QUESTIONS = {
    "first_name": "What is your first name?",
    "last_name": "What is your last name?",
    "car_number": "What is your car registration (licence plate) number?",
    "start": "When will you arrive? Please give the date and time.",
    "end": "When will you leave? Please give the date and time.",
}


class InvalidSlotError(ValueError):
    pass


class SlotUpdates(BaseModel):
    """Values the user provided in their latest message. Use an empty string for anything not provided."""

    first_name: str = Field(description="Driver's first name, or empty string.")
    last_name: str = Field(description="Driver's last name, or empty string.")
    car_number: str = Field(description="Car registration plate, or empty string.")
    start: str = Field(description="Arrival as YYYY-MM-DDTHH:MM in local time, or empty string.")
    end: str = Field(description="Departure as YYYY-MM-DDTHH:MM in local time, or empty string.")
    confirmation: Literal["yes", "no", "none"] = Field(
        description="'yes' or 'no' only if the user answers a request to confirm the summary."
    )


@dataclass(frozen=True)
class ReservationDraft:
    first_name: str = ""
    last_name: str = ""
    car_number: str = ""
    start: str = ""
    end: str = ""
    awaiting_confirmation: bool = False
    active: bool = False


class Step(Enum):
    ASK = "ask"
    CONFIRM = "confirm"
    REVISE = "revise"
    BOOK = "book"


@dataclass(frozen=True)
class Outcome:
    draft: ReservationDraft
    step: Step
    problems: tuple[str, ...]


def missing_fields(draft: ReservationDraft) -> tuple[str, ...]:
    values = asdict(draft)
    return tuple(name for name in FIELD_ORDER if not values[name])


def is_started(draft: ReservationDraft) -> bool:
    return draft.active


def normalize_name(value: str, label: str) -> str:
    name = " ".join(value.split())
    if len(name) > 50 or not NAME_PATTERN.fullmatch(name):
        raise InvalidSlotError(f"The {label} should contain letters only.")
    return name


def normalize_car_number(value: str) -> str:
    plate = re.sub(r"[\s-]", "", value).upper()
    if not PLATE_PATTERN.fullmatch(plate):
        raise InvalidSlotError("A car number should have 3 to 10 letters or digits.")
    return plate


def parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value).replace(second=0, microsecond=0, tzinfo=None)
    except ValueError as error:
        raise InvalidSlotError(f"I could not understand the date and time '{value}'.") from error


def format_time(moment: datetime) -> str:
    return moment.strftime(TIME_FORMAT)


def period_problem(start: str, end: str, now: datetime) -> str:
    """Return a user-facing description of what is wrong with the period, or an empty string."""
    if start and parse_time(start) < now - START_GRACE:
        return "The arrival time is in the past."
    if end and parse_time(end) <= now:
        return "The departure time is in the past."
    if start and parse_time(start) > now + MAX_ADVANCE:
        return f"Reservations can be made at most {MAX_ADVANCE.days} days ahead."
    if start and end and parse_time(end) <= parse_time(start):
        return "The departure must be after the arrival."
    if start and end and parse_time(end) - parse_time(start) > MAX_STAY:
        return f"The maximum reservation length is {MAX_STAY.days} days."
    return ""


def _apply_period(
    draft: ReservationDraft, updates: SlotUpdates, now: datetime
) -> tuple[ReservationDraft, tuple[str, ...]]:
    try:
        start = format_time(parse_time(updates.start)) if updates.start else draft.start
        end = format_time(parse_time(updates.end)) if updates.end else draft.end
    except InvalidSlotError as error:
        return draft, (str(error),)
    problem = period_problem(start, end, now)
    if problem:
        return draft, (problem,)
    return replace(draft, start=start, end=end), ()


def _apply_text_field(
    draft: ReservationDraft, field: str, raw: str, normalize: Callable[[str], str]
) -> tuple[ReservationDraft, tuple[str, ...]]:
    if not raw:
        return draft, ()
    try:
        return replace(draft, **{field: normalize(raw)}), ()
    except InvalidSlotError as error:
        return draft, (str(error),)


def apply_updates(
    draft: ReservationDraft, updates: SlotUpdates, now: datetime
) -> tuple[ReservationDraft, tuple[str, ...]]:
    steps = (
        lambda d: _apply_text_field(
            d, "first_name", updates.first_name, lambda v: normalize_name(v, "first name")
        ),
        lambda d: _apply_text_field(
            d, "last_name", updates.last_name, lambda v: normalize_name(v, "last name")
        ),
        lambda d: _apply_text_field(d, "car_number", updates.car_number, normalize_car_number),
        lambda d: _apply_period(d, updates, now),
    )
    problems: tuple[str, ...] = ()
    for step in steps:
        draft, found = step(draft)
        problems = (*problems, *found)
    return draft, problems


def advance(draft: ReservationDraft, updates: SlotUpdates, now: datetime) -> Outcome:
    updated, problems = apply_updates(replace(draft, active=True), updates, now)
    if missing_fields(updated):
        return Outcome(replace(updated, awaiting_confirmation=False), Step.ASK, problems)
    if draft.awaiting_confirmation and updated == draft and not problems:
        if updates.confirmation == "yes":
            return Outcome(updated, Step.BOOK, ())
        if updates.confirmation == "no":
            return Outcome(replace(updated, awaiting_confirmation=False), Step.REVISE, ())
    return Outcome(replace(updated, awaiting_confirmation=True), Step.CONFIRM, problems)


def display_time(value: str) -> str:
    return parse_time(value).strftime(DISPLAY_FORMAT)


def _with_problems(problems: tuple[str, ...], text: str) -> str:
    return "\n".join([*problems, text]) if problems else text


def render_question(
    draft: ReservationDraft, problems: tuple[str, ...] = (), introduce: bool = False
) -> str:
    question = QUESTIONS[missing_fields(draft)[0]]
    if introduce and not problems:
        question = f"I can reserve a space for you. I need a few details.\n{question}"
    return _with_problems(problems, question)


def render_summary(draft: ReservationDraft, problems: tuple[str, ...] = ()) -> str:
    summary = (
        "Please check your reservation:\n"
        f"- Name: {draft.first_name} {draft.last_name}\n"
        f"- Car number: {draft.car_number}\n"
        f"- Arrival: {display_time(draft.start)}\n"
        f"- Departure: {display_time(draft.end)}\n"
        "Shall I submit it? (yes / no)"
    )
    return _with_problems(problems, summary)


def render_revision_request() -> str:
    return "No problem. What would you like to change?"


def render_unavailable(draft: ReservationDraft) -> str:
    return (
        f"Sorry, there are no reservable spaces for {display_time(draft.start)} - "
        f"{display_time(draft.end)}. Please give me a different period."
    )


def render_cancelled() -> str:
    return "I cancelled the reservation in progress. Nothing was submitted."


def render_pending_prompt(draft: ReservationDraft) -> str:
    if draft.awaiting_confirmation:
        return "Back to your reservation: shall I submit it? (yes / no)"
    missing = missing_fields(draft)
    if not missing:
        return "Back to your reservation: what would you like to change?"
    return f"Back to your reservation: {QUESTIONS[missing[0]]}"
