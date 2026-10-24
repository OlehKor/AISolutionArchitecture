import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from parking_assistant.database import CONFIRMED, PENDING, REJECTED, Reservation
from parking_assistant.reservation import display_time

REFERENCE_PATTERN = re.compile(r"PK-[0-9A-F]{6}")


class AdminDeliveryError(Exception):
    """The administrator could not be reached through the configured channel."""


class Decision(BaseModel):
    """The administrator's answer to one reservation request."""

    verdict: Literal["approve", "reject", "unclear"] = Field(
        description="'approve' or 'reject' only when the reply clearly says so, otherwise 'unclear'."
    )
    reason: str = Field(description="The reason given for the decision, or an empty string.")


@dataclass(frozen=True)
class IncomingMessage:
    chat_id: int
    text: str
    reply_to_message_id: int = 0


@dataclass(frozen=True)
class Target:
    reference: str
    problem: str


def resolve_target(replied_reference: str, text: str, pending: Sequence[str]) -> Target:
    """Work out which reservation an administrator message refers to."""
    if replied_reference:
        return Target(replied_reference, "")
    mentioned = REFERENCE_PATTERN.search(text.upper())
    if mentioned:
        return Target(mentioned.group(), "")
    if len(pending) == 1:
        return Target(pending[0], "")
    if not pending:
        return Target("", "There are no pending reservation requests.")
    return Target(
        "",
        "Several requests are pending: "
        f"{', '.join(pending)}. Reply to the request message or include its reference.",
    )


def _period(reservation: Reservation) -> str:
    return f"{display_time(reservation.start_time)} - {display_time(reservation.end_time)}"


def render_request(reservation: Reservation, spaces_left: int, note: str) -> str:
    return (
        f"New reservation request {reservation.reference}\n"
        f"Customer: {reservation.first_name} {reservation.last_name}\n"
        f"Car number: {reservation.car_number}\n"
        f"Arrival: {display_time(reservation.start_time)}\n"
        f"Departure: {display_time(reservation.end_time)}\n"
        f"Reservable spaces left for this period: {spaces_left}\n"
        f"Note: {note.strip()}\n\n"
        'Reply to this message with your decision, for example "approve" '
        'or "reject, the garage is closed that day".'
    )


def render_pending_list(pending: Sequence[Reservation]) -> str:
    if not pending:
        return "There are no pending reservation requests."
    lines = [
        f"- {r.reference}: {r.first_name} {r.last_name}, {r.car_number}, {_period(r)}"
        for r in pending
    ]
    return "Pending reservation requests:\n" + "\n".join(lines)


def render_admin_help() -> str:
    return (
        "I forward reservation requests from the customer assistant.\n"
        'Reply to a request with your decision, for example "approve" or '
        '"reject, no spaces that day". Send /pending to list open requests.'
    )


def render_decision_ack(reservation: Reservation) -> str:
    verb = "confirmed" if reservation.status == CONFIRMED else "rejected"
    return f"Reservation {reservation.reference} {verb}. The customer will be told in the chat."


def render_already_decided(reservation: Reservation) -> str:
    return f"Reservation {reservation.reference} was already {reservation.status}."


def render_clarification(reference: str) -> str:
    return (
        f"I could not tell whether you approve or reject {reference}. "
        'Please reply "approve" or "reject" (with a reason).'
    )


def render_unknown_reference(reference: str) -> str:
    return f"I do not know a reservation with reference {reference}."


def render_submitted(reference: str) -> str:
    return (
        f"Thank you! Your reservation request {reference} was sent to our administrator for "
        "confirmation. I will tell you the decision here, and you can ask me for its status."
    )


def render_saved_without_delivery(reference: str) -> str:
    return (
        f"Your reservation request {reference} is saved and waiting for confirmation, but I could "
        "not reach the administrator just now. It will be handled as soon as they are available."
    )


def render_decision_notice(reservation: Reservation) -> str:
    if reservation.status == CONFIRMED:
        return (
            f"Good news: reservation {reservation.reference} for {_period(reservation)} "
            "was confirmed by the administrator."
        )
    reason = f" Reason: {reservation.admin_comment}." if reservation.admin_comment else ""
    return (
        f"Reservation request {reservation.reference} for {_period(reservation)} was declined "
        f"by the administrator.{reason} You are welcome to request another period."
    )


def render_status(reservations: Sequence[Reservation]) -> str:
    if not reservations:
        return "I have no reservation requests from you in this conversation."
    waiting = "waiting for the administrator's decision"
    lines = []
    for r in reservations:
        state = {PENDING: waiting, CONFIRMED: "confirmed", REJECTED: "declined"}[r.status]
        reason = f" ({r.admin_comment})" if r.status == REJECTED and r.admin_comment else ""
        lines.append(f"- {r.reference}, {_period(r)}: {state}{reason}")
    return "Your reservation requests:\n" + "\n".join(lines)
