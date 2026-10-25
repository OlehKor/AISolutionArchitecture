"""End-to-end tests of the whole pipeline: chat -> approval workflow -> Telegram agent -> MCP file.

Only the LLMs and Telegram are replaced by fakes; the graphs, SQLite (data and checkpoints),
the MCP server and its HTTP client are real. The chat side and the admin side each get their own
connections, as they would in separate processes.
"""

import threading

from conftest import SCHEMA_SQL, SEED_SQL
from fakes import (
    NOW,
    MaskingRedactor,
    RecordingChannel,
    ScriptedAdministratorModel,
    ScriptedModel,
    StaticKnowledge,
    slot_updates,
)
from parking_assistant.administrator import AdminDesk, AdministratorAgent
from parking_assistant.approval import ButtonPress, IncomingMessage
from parking_assistant.archive import McpReservationArchive
from parking_assistant.assistant import Dependencies, ParkingAssistant
from parking_assistant.conversation import Route
from parking_assistant.database import CONFIRMED, REJECTED, ParkingDatabase
from parking_assistant.guardrails import Guardrails
from parking_assistant.mcp_server import background_server, free_port
from parking_assistant.workflow import ReservationWorkflow, open_saver

TOKEN = "p" * 32
ADMIN_CHAT = 42
RESERVE = Route(intent="reserve", needs_live_data=False)


class Stack:
    """One process worth of components over shared files."""

    def __init__(self, tmp_path, mcp_url):
        self.database = ParkingDatabase.open(tmp_path / "parking.sqlite", SCHEMA_SQL, SEED_SQL)
        self.channel = RecordingChannel()
        self.model = ScriptedAdministratorModel()
        self.archive = McpReservationArchive(mcp_url, TOKEN, attempts=1)
        self.desk = AdminDesk(self.model, self.channel, self.database, lambda: NOW)
        self.workflow = ReservationWorkflow(
            self.database,
            self.desk,
            self.archive,
            lambda: NOW,
            open_saver(tmp_path / "workflow.sqlite"),
        )
        self.agent = AdministratorAgent(
            self.model, self.channel, self.database, ADMIN_CHAT, self.workflow
        )

    def customer(self, reference_factory, car="ABC123"):
        details = slot_updates(
            first_name="Anna",
            last_name="Kowalska",
            car_number=car,
            start="2026-11-01T09:00",
            end="2026-11-01T17:30",
        )
        model = ScriptedModel(
            [RESERVE] * 3, [slot_updates(), details, slot_updates(confirmation="yes")]
        )
        return ParkingAssistant(
            Dependencies(
                model=model,
                knowledge=StaticKnowledge([]),
                database=self.database,
                guardrails=Guardrails(MaskingRedactor()),
                workflow=self.workflow,
                clock=lambda: NOW,
                new_reference=reference_factory,
                history_messages=6,
            )
        )


def book(assistant, session="s1") -> str:
    assistant.reply(session, "I want to book a space")
    assistant.reply(session, "Anna Kowalska ABC123 tomorrow 9:00-17:30")
    return assistant.reply(session, "yes")


def tap(stack, action, reference, message_id=501):
    stack.agent.handle_button(ButtonPress(ADMIN_CHAT, "cb", message_id, action, reference))


def lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_a_customer_booking_travels_through_approval_to_the_reservations_file(tmp_path):
    with background_server(TOKEN, tmp_path / "confirmed.txt") as url:
        chat = Stack(tmp_path, url)
        admin = Stack(tmp_path, url)
        assistant = chat.customer(lambda: "PK-A1B2C3")

        submitted = book(assistant)

        assert "PK-A1B2C3" in submitted
        assert "New reservation request PK-A1B2C3" in chat.channel.sent[0]
        assert chat.channel.buttons_for == ["PK-A1B2C3"]
        assert lines(tmp_path / "confirmed.txt") == []

        tap(admin, "approve", "PK-A1B2C3")

        assert lines(tmp_path / "confirmed.txt") == [
            "Anna Kowalska | ABC123 | 2026-11-01 09:00 - 2026-11-01 17:30 | 2026-10-24 10:00"
        ]
        assert "saved to the confirmed reservations file" in admin.channel.sent[-1]
        assert assistant.announcements("s1").startswith("Good news")
        assert assistant.announcements("s1") == ""


def test_a_rejection_with_a_reason_reaches_the_customer_and_writes_nothing(tmp_path):
    with background_server(TOKEN, tmp_path / "confirmed.txt") as url:
        chat = Stack(tmp_path, url)
        admin = Stack(tmp_path, url)
        assistant = chat.customer(lambda: "PK-A1B2C3")
        book(assistant)

        tap(admin, "reject", "PK-A1B2C3")
        admin.agent.handle_admin_message(IncomingMessage(ADMIN_CHAT, "closed for an event"))

        assert lines(tmp_path / "confirmed.txt") == []
        notice = assistant.announcements("s1")
        assert "declined" in notice and "closed for an event" in notice
        assert chat.database.get_reservation("PK-A1B2C3").status == REJECTED


def test_a_confirmation_made_while_the_file_server_is_down_is_stored_once_it_returns(tmp_path):
    dead_url = f"http://127.0.0.1:{free_port()}/mcp"
    chat = Stack(tmp_path, dead_url)
    admin_while_down = Stack(tmp_path, dead_url)
    book(chat.customer(lambda: "PK-A1B2C3"))

    tap(admin_while_down, "approve", "PK-A1B2C3")

    assert "keep retrying" in admin_while_down.channel.sent[-1]
    assert chat.database.get_reservation("PK-A1B2C3").status == CONFIRMED
    assert [r.reference for r in chat.database.unarchived_confirmations()] == ["PK-A1B2C3"]

    with background_server(TOKEN, tmp_path / "confirmed.txt") as url:
        restarted_admin = Stack(tmp_path, url)

        assert restarted_admin.workflow.retry_pending() == 1
        assert restarted_admin.workflow.retry_pending() == 0

    assert len(lines(tmp_path / "confirmed.txt")) == 1
    assert chat.database.unarchived_confirmations() == []


def test_many_bookings_decided_in_parallel_all_end_up_in_the_file_exactly_once(tmp_path):
    count = 12
    with background_server(TOKEN, tmp_path / "confirmed.txt") as url:
        chat = Stack(tmp_path, url)
        references = [f"PK-{number:06X}" for number in range(1, count + 1)]
        for number, reference in enumerate(references):
            book(chat.customer(lambda ref=reference: ref, car=f"CAR{number:03d}"), f"s{number}")
        assert chat.database.free_reservable_spaces("2026-11-01T09:00", "2026-11-01T17:30") == (
            150 - count
        )
        errors: list[Exception] = []

        def approve(batch: list[str]) -> None:
            try:
                admin = Stack(tmp_path, url)
                for reference in batch:
                    tap(admin, "approve", reference)
            except Exception as error:
                errors.append(error)

        threads = [
            threading.Thread(target=approve, args=(references[offset::3],)) for offset in range(3)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        written = lines(tmp_path / "confirmed.txt")
        assert len(written) == count
        assert len(set(written)) == count
        assert chat.database.unarchived_confirmations() == []
        assert chat.database.pending_reservations() == []


def test_a_restarted_admin_process_still_decides_requests_started_before_it(tmp_path):
    with background_server(TOKEN, tmp_path / "confirmed.txt") as url:
        chat = Stack(tmp_path, url)
        book(chat.customer(lambda: "PK-A1B2C3"))

        second_admin = Stack(tmp_path, url)
        second_admin.agent.handle_admin_message(IncomingMessage(ADMIN_CHAT, "/pending"))
        tap(second_admin, "approve", "PK-A1B2C3")

        assert "PK-A1B2C3" in second_admin.channel.sent[0]
        assert len(lines(tmp_path / "confirmed.txt")) == 1
