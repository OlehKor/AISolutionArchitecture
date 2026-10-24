from parking_assistant.approval import (
    render_decision_notice,
    render_pending_list,
    render_request,
    render_saved_without_delivery,
    render_status,
    render_submitted,
    resolve_target,
)
from parking_assistant.database import CONFIRMED, PENDING, REJECTED, Reservation


def reservation(status: str = PENDING, comment: str = "") -> Reservation:
    return Reservation(
        "PK-AB12CD",
        "Anna",
        "Kowalska",
        "ABC123",
        "2026-10-25T09:00",
        "2026-10-25T17:30",
        status,
        comment,
    )


def test_replied_reference_wins_over_everything_else():
    target = resolve_target("PK-AAAAAA", "approve PK-BBBBBB", ["PK-AAAAAA", "PK-BBBBBB"])
    assert (target.reference, target.problem) == ("PK-AAAAAA", "")


def test_reference_in_the_text_is_found_case_insensitively():
    target = resolve_target("", "please approve pk-bbbbbb", ["PK-AAAAAA", "PK-BBBBBB"])
    assert target.reference == "PK-BBBBBB"


def test_single_pending_request_is_the_default_target():
    assert resolve_target("", "approve", ["PK-AAAAAA"]).reference == "PK-AAAAAA"


def test_ambiguous_or_missing_targets_explain_the_problem():
    ambiguous = resolve_target("", "approve", ["PK-AAAAAA", "PK-BBBBBB"])
    assert ambiguous.reference == ""
    assert "PK-AAAAAA, PK-BBBBBB" in ambiguous.problem
    assert "no pending" in resolve_target("", "approve", []).problem


def test_request_message_contains_all_details_and_instructions():
    text = render_request(reservation(), spaces_left=149, note="Looks routine.")

    for expected in ("PK-AB12CD", "Anna Kowalska", "ABC123", "09:00", "17:30", "149", "routine"):
        assert expected in text
    assert "Reply to this message" in text


def test_pending_list_names_each_request():
    assert "no pending" in render_pending_list([])
    assert "PK-AB12CD: Anna Kowalska" in render_pending_list([reservation()])


def test_customer_notices_cover_both_decisions():
    confirmed = render_decision_notice(reservation(CONFIRMED))
    rejected = render_decision_notice(reservation(REJECTED, "closed that day"))

    assert confirmed.startswith("Good news")
    assert "declined" in rejected
    assert "closed that day" in rejected


def test_status_lists_every_request_with_its_state():
    text = render_status([reservation(PENDING), reservation(REJECTED, "closed")])

    assert "waiting for the administrator" in text
    assert "declined (closed)" in text
    assert "no reservation requests" in render_status([])


def test_submission_messages_mention_the_reference():
    assert "PK-AB12CD" in render_submitted("PK-AB12CD")
    assert "could not reach the administrator" in render_saved_without_delivery("PK-AB12CD")
