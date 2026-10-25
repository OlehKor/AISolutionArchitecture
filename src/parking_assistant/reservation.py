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
DATE_FORMAT = "%a %d %b %Y"
MAX_STAY = timedelta(days=30)
MAX_ADVANCE = timedelta(days=90)
START_GRACE = timedelta(minutes=5)
MAX_ANSWER_WORDS = {"first_name": 2, "last_name": 3, "car_number": 2}
NOT_AN_ANSWER = frozenset(
    {"yes", "no", "y", "n", "yeah", "yep", "nope", "ok", "okay", "cancel", "stop", "hi", "hello"}
)

NAME_PATTERN = re.compile(r"[^\W\d_]+(?:[ '\-][^\W\d_]+)*")
PLATE_PATTERN = re.compile(r"[A-Z0-9]{3,10}")
TIME_ONLY_PATTERN = re.compile(r"(?:at\s+)?(\d{1,2})(?:[:.h](\d{2}))?\s*(am|pm)?", re.IGNORECASE)

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
    start: str = Field(
        description="Arrival as YYYY-MM-DDTHH:MM in local time, YYYY-MM-DD if no time was given, or empty string."
    )
    end: str = Field(
        description="Departure as YYYY-MM-DDTHH:MM in local time, YYYY-MM-DD if no time was given, or empty string."
    )
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
    start_date: str = ""
    end_date: str = ""


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
    if len(value.strip()) <= len("YYYY-MM-DD"):
        raise InvalidSlotError("Please give the time as well as the date, for example 09:00.")
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


def _split_moment(raw: str) -> tuple[str, str]:
    """Split a value from the model into (date and time, date only); one of them is empty."""
    if not raw:
        return "", ""
    if len(raw.strip()) <= len("YYYY-MM-DD"):
        try:
            return "", datetime.fromisoformat(raw.strip()).strftime("%Y-%m-%d")
        except ValueError as error:
            raise InvalidSlotError(f"I could not understand the date '{raw}'.") from error
    return format_time(parse_time(raw)), ""


def date_problem(start_date: str, end_date: str, start: str, now: datetime) -> str:
    """Like `period_problem` for a date whose time is still unknown."""
    today = now.strftime("%Y-%m-%d")
    horizon = (now + MAX_ADVANCE).strftime("%Y-%m-%d")
    if start_date and start_date < today:
        return "The arrival date is in the past."
    if start_date and start_date > horizon:
        return f"Reservations can be made at most {MAX_ADVANCE.days} days ahead."
    if end_date and end_date < today:
        return "The departure date is in the past."
    if end_date and end_date < (start_date or start[:10]):
        return "The departure must be after the arrival."
    return ""


def _apply_period(
    draft: ReservationDraft, updates: SlotUpdates, now: datetime
) -> tuple[ReservationDraft, tuple[str, ...]]:
    try:
        new_start, new_start_date = _split_moment(updates.start)
        new_end, new_end_date = _split_moment(updates.end)
    except InvalidSlotError as error:
        return draft, (str(error),)
    start = new_start or ("" if new_start_date else draft.start)
    end = new_end or ("" if new_end_date else draft.end)
    start_date = new_start_date or ("" if new_start else draft.start_date)
    end_date = new_end_date or ("" if new_end else draft.end_date)
    problem = period_problem(start, end, now) or date_problem(start_date, end_date, start, now)
    if problem:
        return draft, (problem,)
    return replace(draft, start=start, end=end, start_date=start_date, end_date=end_date), ()


def _apply_text_field(
    draft: ReservationDraft, field: str, raw: str, normalize: Callable[[str], str]
) -> tuple[ReservationDraft, tuple[str, ...]]:
    if not raw:
        return draft, ()
    try:
        return replace(draft, **{field: normalize(raw)}), ()
    except InvalidSlotError as error:
        return draft, (str(error),)


def _plain(value: str) -> str:
    return re.sub(r"[\s-]", "", value).casefold()


def _adds_information(draft: ReservationDraft, updates: SlotUpdates) -> bool:
    known = asdict(draft)
    return any(
        getattr(updates, field) and _plain(getattr(updates, field)) != _plain(known[field])
        for field in FIELD_ORDER
    )


def bare_time(message: str) -> str:
    """Return HH:MM when the whole message is just a clock time such as 9, 09:30 or 5pm."""
    found = TIME_ONLY_PATTERN.fullmatch(message.strip().rstrip(".!"))
    if not found:
        return ""
    hour, minute, meridiem = int(found[1]), int(found[2] or 0), (found[3] or "").lower()
    if meridiem:
        if not 1 <= hour <= 12:
            return ""
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    if hour > 23 or minute > 59:
        return ""
    return f"{hour:02d}:{minute:02d}"


def fill_pending_answer(draft: ReservationDraft, updates: SlotUpdates, message: str) -> SlotUpdates:
    """Take a short plain reply as the answer to the pending question.

    The language model sometimes returns nothing for unusual words such as an unfamiliar name,
    and a bare time has to be joined with the date that was already given.
    """
    missing = missing_fields(draft)
    pending = missing[0] if draft.active and missing and not draft.awaiting_confirmation else ""
    known_date = {"start": draft.start_date, "end": draft.end_date}.get(pending, "")
    if known_date and bare_time(message):
        return updates.model_copy(update={pending: f"{known_date}T{bare_time(message)}"})
    words = message.split()
    spoken = {word.casefold().strip(".,!") for word in words}
    if (
        pending not in MAX_ANSWER_WORDS
        or not 0 < len(words) <= MAX_ANSWER_WORDS[pending]
        or "?" in message
        or spoken & NOT_AN_ANSWER
        or _adds_information(draft, updates)
    ):
        return updates
    if pending == "first_name" and len(words) == 2:
        return updates.model_copy(update={"first_name": words[0], "last_name": words[1]})
    return updates.model_copy(update={pending: " ".join(words)})


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


def display_date(value: str) -> str:
    return datetime.fromisoformat(value).strftime(DATE_FORMAT)


def pending_question(draft: ReservationDraft) -> str:
    """The next question, asking only for the time when the date is already known."""
    field = missing_fields(draft)[0]
    known_date = {"start": draft.start_date, "end": draft.end_date}.get(field, "")
    if known_date:
        verb = "arrive" if field == "start" else "leave"
        return f"What time will you {verb} on {display_date(known_date)}?"
    return QUESTIONS[field]


def _with_problems(problems: tuple[str, ...], text: str) -> str:
    return "\n".join([*problems, text]) if problems else text


def render_question(
    draft: ReservationDraft, problems: tuple[str, ...] = (), introduce: bool = False
) -> str:
    question = pending_question(draft)
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
    return f"Back to your reservation: {pending_question(draft)}"
