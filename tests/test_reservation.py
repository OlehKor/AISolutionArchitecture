from dataclasses import replace

import pytest

from fakes import NOW, slot_updates
from parking_assistant.reservation import (
    InvalidSlotError,
    ReservationDraft,
    Step,
    advance,
    apply_updates,
    fill_pending_answer,
    missing_fields,
    normalize_car_number,
    normalize_name,
    period_problem,
    render_question,
    render_summary,
)

COMPLETE = ReservationDraft(
    first_name="Anna",
    last_name="Kowalska",
    car_number="ABC123",
    start="2026-10-25T09:00",
    end="2026-10-25T17:30",
    active=True,
)
COMPLETE_UPDATES = {
    "first_name": "Anna",
    "last_name": "Kowalska",
    "car_number": "abc 123",
    "start": "2026-10-25T09:00",
    "end": "2026-10-25T17:30",
}


def test_missing_fields_lists_everything_for_empty_draft():
    assert missing_fields(ReservationDraft()) == (
        "first_name",
        "last_name",
        "car_number",
        "start",
        "end",
    )


def test_normalize_name_collapses_whitespace_and_rejects_digits():
    assert normalize_name("  Anna   Maria ", "first name") == "Anna Maria"
    with pytest.raises(InvalidSlotError):
        normalize_name("R2D2", "first name")


def test_normalize_car_number_uppercases_and_strips_separators():
    assert normalize_car_number("abc-12 3") == "ABC123"
    with pytest.raises(InvalidSlotError):
        normalize_car_number("!!")


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2026-10-24T09:00", "", "The arrival time is in the past."),
        ("2026-10-25T09:00", "2026-10-25T08:00", "The departure must be after the arrival."),
        ("2026-10-25T09:00", "2026-12-25T09:00", "The maximum reservation length is 30 days."),
        ("2027-06-01T09:00", "2027-06-02T09:00", "Reservations can be made at most 90 days ahead."),
        ("2026-10-25T09:00", "2026-10-25T17:00", ""),
    ],
)
def test_period_problem(start, end, expected):
    assert period_problem(start, end, NOW) == expected


def test_apply_updates_reports_invalid_values_and_keeps_draft():
    updated, problems = apply_updates(ReservationDraft(), slot_updates(first_name="R2D2"), NOW)
    assert updated == ReservationDraft()
    assert problems == ("The first name should contain letters only.",)


def test_advance_marks_the_reservation_active_even_without_details():
    outcome = advance(ReservationDraft(), slot_updates(), NOW)

    assert outcome.step is Step.ASK
    assert outcome.draft.active
    assert render_question(outcome.draft, introduce=True).startswith("I can reserve")
    assert render_question(outcome.draft) == "What is your first name?"


def test_advance_asks_for_missing_details_first():
    outcome = advance(ReservationDraft(), slot_updates(first_name="Anna"), NOW)
    assert outcome.step is Step.ASK
    assert outcome.draft.first_name == "Anna"
    assert render_question(outcome.draft) == "What is your last name?"


def test_advance_confirms_then_books_only_after_yes():
    summary = advance(ReservationDraft(), slot_updates(**COMPLETE_UPDATES), NOW)
    assert summary.step is Step.CONFIRM
    assert summary.draft == replace(COMPLETE, awaiting_confirmation=True)

    booked = advance(summary.draft, slot_updates(confirmation="yes"), NOW)
    assert booked.step is Step.BOOK


def test_advance_returns_to_revision_on_no_and_resummarizes_after_change():
    awaiting = replace(COMPLETE, awaiting_confirmation=True)
    declined = advance(awaiting, slot_updates(confirmation="no"), NOW)
    assert declined.step is Step.REVISE
    assert not declined.draft.awaiting_confirmation

    changed = advance(declined.draft, slot_updates(end="2026-10-25T18:00"), NOW)
    assert changed.step is Step.CONFIRM
    assert changed.draft.end == "2026-10-25T18:00"


def test_advance_never_books_when_details_change_together_with_yes():
    awaiting = replace(COMPLETE, awaiting_confirmation=True)
    outcome = advance(awaiting, slot_updates(end="2026-10-25T18:00", confirmation="yes"), NOW)
    assert outcome.step is Step.CONFIRM


def test_summary_message_shows_the_details():
    summary = render_summary(COMPLETE)
    assert "Anna Kowalska" in summary
    assert "ABC123" in summary
    assert "25 Oct 2026, 09:00" in summary


@pytest.mark.parametrize(
    ("draft", "message", "expected"),
    [
        (ReservationDraft(active=True), "ol", {"first_name": "ol"}),
        (ReservationDraft(active=True), "ol porol", {"first_name": "ol", "last_name": "porol"}),
        (
            ReservationDraft(active=True, first_name="Ol"),
            "van der Berg",
            {"last_name": "van der Berg"},
        ),
        (
            ReservationDraft(active=True, first_name="Ol", last_name="Po"),
            "abc-123",
            {"car_number": "abc-123"},
        ),
    ],
)
def test_short_plain_reply_answers_the_pending_question(draft, message, expected):
    result = fill_pending_answer(draft, slot_updates(), message)

    assert {k: v for k, v in result.model_dump().items() if v not in ("", "none")} == expected


@pytest.mark.parametrize(
    "message", ["yes", "No", "what is your name?", "I do not want to say", "hello", ""]
)
def test_replies_that_are_not_answers_are_left_alone(message):
    assert (
        fill_pending_answer(ReservationDraft(active=True), slot_updates(), message)
        == slot_updates()
    )


def test_reply_is_not_forced_when_the_model_found_something_new():
    updates = slot_updates(first_name="Anna")

    assert fill_pending_answer(ReservationDraft(active=True), updates, "Anna") == updates


def test_reply_is_not_used_when_the_model_only_repeated_known_details():
    draft = ReservationDraft(active=True, first_name="Ol")

    result = fill_pending_answer(draft, slot_updates(first_name="ol"), "porol")

    assert result.last_name == "porol"


def test_reply_is_not_used_while_waiting_for_confirmation_or_dates():
    awaiting = replace(COMPLETE, awaiting_confirmation=True)
    needs_dates = ReservationDraft(active=True, first_name="A", last_name="B", car_number="ABC123")

    assert fill_pending_answer(awaiting, slot_updates(), "ok2") == slot_updates()
    assert fill_pending_answer(needs_dates, slot_updates(), "tomorrow") == slot_updates()


def test_the_message_that_starts_a_reservation_is_not_taken_as_an_answer():
    assert fill_pending_answer(ReservationDraft(), slot_updates(), "book") == slot_updates()
