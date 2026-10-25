from datetime import datetime

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from fakes import NOW, RecordingArchive, RecordingChannel, ScriptedAdministratorModel
from parking_assistant.administrator import AdministratorAgent, LlmAdministratorModel
from parking_assistant.approval import AdminDeliveryError, Decision, IncomingMessage
from parking_assistant.database import CONFIRMED, PENDING, REJECTED
from parking_assistant.reservation import ReservationDraft

ADMIN_CHAT = 42
APPROVE = Decision(verdict="approve", reason="")
REJECT = Decision(verdict="reject", reason="closed that day")
UNCLEAR = Decision(verdict="unclear", reason="")


def draft(car: str = "ABC123") -> ReservationDraft:
    return ReservationDraft(
        first_name="Anna",
        last_name="Kowalska",
        car_number=car,
        start="2026-10-25T09:00",
        end="2026-10-25T17:30",
    )


def message(text: str, reply_to: int = 0, chat_id: int = ADMIN_CHAT) -> IncomingMessage:
    return IncomingMessage(chat_id=chat_id, text=text, reply_to_message_id=reply_to)


def make_agent(database, channel=None, model=None, archive=None):
    channel = channel or RecordingChannel()
    model = model or ScriptedAdministratorModel()
    archive = archive or RecordingArchive()
    agent = AdministratorAgent(model, channel, database, lambda: NOW, ADMIN_CHAT, archive)
    return agent, channel, model


def submit(database, agent, reference="PK-TEST01", car="ABC123"):
    reservation = database.create_reservation(draft(car), reference, NOW)
    agent.request_approval(reservation)
    return reservation


def test_request_is_sent_to_the_administrator_and_remembered(database):
    agent, channel, _ = make_agent(database, model=ScriptedAdministratorModel(note="Long stay."))

    submit(database, agent)

    assert "New reservation request PK-TEST01" in channel.sent[0]
    assert "Long stay." in channel.sent[0]
    assert database.reference_for_request_message(101) == "PK-TEST01"
    assert database.unsent_requests() == []


def test_unsent_requests_are_delivered_later(database):
    agent, channel, _ = make_agent(database)
    database.create_reservation(draft(), "PK-TEST01", NOW)

    assert agent.resend_unsent() == 1
    assert agent.resend_unsent() == 0
    assert len(channel.sent) == 1


def test_approval_reply_confirms_the_only_pending_request(database):
    model = ScriptedAdministratorModel(APPROVE)
    agent, channel, _ = make_agent(database, model=model)
    submit(database, agent)

    agent.handle_admin_message(message("yes, go ahead"))

    assert database.get_reservation("PK-TEST01").status == CONFIRMED
    assert model.interpreted == [("PK-TEST01", "yes, go ahead")]
    assert "confirmed" in channel.sent[-1]


def test_rejection_reply_stores_the_reason(database):
    agent, _, _ = make_agent(database, model=ScriptedAdministratorModel(REJECT))
    submit(database, agent)

    agent.handle_admin_message(message("no, closed that day"))

    stored = database.get_reservation("PK-TEST01")
    assert (stored.status, stored.admin_comment) == (REJECTED, "closed that day")


def test_reply_to_a_request_message_targets_that_request(database):
    agent, _, _ = make_agent(database, model=ScriptedAdministratorModel(APPROVE))
    submit(database, agent, "PK-FIRST1", "ABC123")
    submit(database, agent, "PK-SECOND", "XYZ789")

    agent.handle_admin_message(message("approve", reply_to=102))

    assert database.get_reservation("PK-FIRST1").status == PENDING
    assert database.get_reservation("PK-SECOND").status == CONFIRMED


def test_ambiguous_reply_asks_which_request_without_calling_the_model(database):
    model = ScriptedAdministratorModel()
    agent, channel, _ = make_agent(database, model=model)
    submit(database, agent, "PK-FIRST1", "ABC123")
    submit(database, agent, "PK-SECOND", "XYZ789")

    agent.handle_admin_message(message("approve"))

    assert "PK-FIRST1, PK-SECOND" in channel.sent[-1]
    assert model.interpreted == []


def test_unclear_reply_asks_again_and_changes_nothing(database):
    agent, channel, _ = make_agent(database, model=ScriptedAdministratorModel(UNCLEAR))
    submit(database, agent)

    agent.handle_admin_message(message("hmm, who is this customer?"))

    assert database.get_reservation("PK-TEST01").status == PENDING
    assert "approve" in channel.sent[-1]


def test_second_decision_for_the_same_request_is_refused(database):
    agent, channel, _ = make_agent(database, model=ScriptedAdministratorModel(APPROVE, REJECT))
    submit(database, agent, "PK-A1B2C3")
    agent.handle_admin_message(message("approve PK-A1B2C3"))

    agent.handle_admin_message(message("reject PK-A1B2C3"))

    assert database.get_reservation("PK-A1B2C3").status == CONFIRMED
    assert "already confirmed" in channel.sent[-1]


def test_unknown_reference_is_reported(database):
    agent, channel, _ = make_agent(database)

    agent.handle_admin_message(message("approve PK-ABCDEF"))

    assert "PK-ABCDEF" in channel.sent[-1]
    assert "do not know" in channel.sent[-1]


def test_commands_list_pending_requests_and_show_help(database):
    agent, channel, _ = make_agent(database)
    submit(database, agent)

    agent.handle_admin_message(message("/pending"))
    agent.handle_admin_message(message("/start"))

    assert "PK-TEST01" in channel.sent[-2]
    assert "/pending" in channel.sent[-1]


def test_serve_ignores_other_chats_and_survives_channel_outages(database):
    model = ScriptedAdministratorModel(APPROVE)
    channel = RecordingChannel(
        [
            AdminDeliveryError("network down"),
            [message("approve", chat_id=999), message("approve")],
        ]
    )
    agent, channel, _ = make_agent(database, channel, model)
    submit(database, agent)
    pauses: list[float] = []

    agent.serve(should_stop=lambda: not channel._incoming, pause=pauses.append)

    assert pauses == [5]
    assert model.interpreted == [("PK-TEST01", "approve")]
    assert database.get_reservation("PK-TEST01").status == CONFIRMED


def test_serve_keeps_running_after_a_failure_while_processing(database):
    model = ScriptedAdministratorModel()  # no decisions queued: interpreting fails
    channel = RecordingChannel([[message("approve")], [message("/pending")]])
    agent, channel, _ = make_agent(database, channel, model)
    submit(database, agent)

    agent.serve(should_stop=lambda: not channel._incoming, pause=lambda seconds: None)

    assert "Pending reservation requests" in channel.sent[-1]


class StubChatModel(RunnableLambda):
    def __init__(self) -> None:
        super().__init__(lambda prompt: AIMessage("Short notice."))

    def with_structured_output(self, schema):
        return RunnableLambda(lambda prompt: schema(verdict="approve", reason="fine"))


def test_llm_model_writes_a_note_and_parses_the_decision(database):
    model = LlmAdministratorModel(StubChatModel())
    reservation = database.create_reservation(draft(), "PK-TEST01", NOW)

    note = model.write_note(reservation, 149, datetime(2026, 10, 24, 10, 0))
    decision = model.interpret_reply(reservation, "ok, approved")

    assert type(note) is str
    assert note == "Short notice."
    assert decision.verdict == "approve"


def test_approved_reservation_is_archived_with_its_approval_time(database):
    archive = RecordingArchive()
    agent, channel, _ = make_agent(
        database, model=ScriptedAdministratorModel(APPROVE), archive=archive
    )
    submit(database, agent)

    agent.handle_admin_message(message("approve"))

    archived = archive.archived[0]
    assert (archived.reference, archived.status) == ("PK-TEST01", CONFIRMED)
    assert archived.decided_at == NOW.strftime("%Y-%m-%dT%H:%M")
    assert database.unarchived_confirmations() == []
    assert "saved to the confirmed reservations file" in channel.sent[-1]


def test_rejected_reservation_is_not_archived(database):
    archive = RecordingArchive()
    agent, channel, _ = make_agent(
        database, model=ScriptedAdministratorModel(REJECT), archive=archive
    )
    submit(database, agent)

    agent.handle_admin_message(message("no"))

    assert archive.archived == []
    assert "saved" not in channel.sent[-1]


def test_confirmation_survives_an_unavailable_archive_and_is_retried(database):
    archive = RecordingArchive(available=False)
    agent, channel, _ = make_agent(
        database, model=ScriptedAdministratorModel(APPROVE), archive=archive
    )
    submit(database, agent)

    agent.handle_admin_message(message("approve"))

    assert database.get_reservation("PK-TEST01").status == CONFIRMED
    assert [r.reference for r in database.unarchived_confirmations()] == ["PK-TEST01"]
    assert "keep retrying" in channel.sent[-1]
    assert agent.retry_archiving() == 0

    archive.available = True

    assert agent.retry_archiving() == 1
    assert database.unarchived_confirmations() == []
    assert agent.retry_archiving() == 0


def test_serve_retries_archiving_before_every_poll(database):
    archive = RecordingArchive(available=False)
    channel = RecordingChannel([[], []])
    agent, channel, _ = make_agent(database, channel, ScriptedAdministratorModel(APPROVE), archive)
    submit(database, agent)
    agent.handle_admin_message(message("approve"))
    archive.available = True

    agent.serve(should_stop=lambda: not channel._incoming, pause=lambda seconds: None)

    assert len(archive.archived) == 1
