import pytest
from langgraph.checkpoint.memory import InMemorySaver

from conftest import SCHEMA_SQL, SEED_SQL
from fakes import NOW, RecordingArchive, RecordingDesk
from parking_assistant.database import CONFIRMED, PENDING, REJECTED, ParkingDatabase
from parking_assistant.reservation import ReservationDraft
from parking_assistant.workflow import DecisionResult, ReservationWorkflow, open_saver


def draft(car: str = "ABC123") -> ReservationDraft:
    return ReservationDraft(
        first_name="Anna",
        last_name="Kowalska",
        car_number=car,
        start="2026-11-01T09:00",
        end="2026-11-01T17:00",
    )


def make_workflow(database, archive=None, desk=None, saver=None):
    archive = archive or RecordingArchive()
    desk = desk or RecordingDesk(database)
    return ReservationWorkflow(database, desk, archive, lambda: NOW, saver or InMemorySaver())


def reserve(database, reference="PK-TEST01", car="ABC123"):
    return database.create_reservation(draft(car), reference, NOW)


def test_starting_a_run_asks_the_administrator_once_and_waits(database):
    desk = RecordingDesk(database)
    workflow = make_workflow(database, desk=desk)
    reserve(database)

    assert workflow.start("PK-TEST01") is True
    assert workflow.start("PK-TEST01") is True

    assert desk.requests == ["PK-TEST01"]
    assert database.get_reservation("PK-TEST01").status == PENDING


def test_an_unreachable_administrator_is_reported_but_the_run_still_waits(database):
    desk = RecordingDesk(database, reachable=False)
    workflow = make_workflow(database, desk=desk)
    reserve(database)

    assert workflow.start("PK-TEST01") is False

    desk._reachable = True
    result = workflow.decide("PK-TEST01", "approve", "")
    assert result.applied
    assert database.get_reservation("PK-TEST01").status == CONFIRMED


def test_approval_records_the_reservation_and_marks_it_archived(database):
    archive = RecordingArchive()
    workflow = make_workflow(database, archive)
    reserve(database)
    workflow.start("PK-TEST01")

    result = workflow.decide("PK-TEST01", "approve", "")

    assert result == DecisionResult(applied=True, status=CONFIRMED, archived=True)
    assert [r.reference for r in archive.archived] == ["PK-TEST01"]
    assert database.unarchived_confirmations() == []


def test_rejection_stores_the_reason_and_records_nothing(database):
    archive = RecordingArchive()
    workflow = make_workflow(database, archive)
    reserve(database)
    workflow.start("PK-TEST01")

    result = workflow.decide("PK-TEST01", "reject", "closed that day")

    assert result.status == REJECTED
    assert database.get_reservation("PK-TEST01").admin_comment == "closed that day"
    assert archive.archived == []


def test_a_second_decision_is_not_applied(database):
    workflow = make_workflow(database)
    reserve(database)
    workflow.start("PK-TEST01")
    workflow.decide("PK-TEST01", "approve", "")

    again = workflow.decide("PK-TEST01", "reject", "changed my mind")

    assert not again.applied
    assert again.status == CONFIRMED
    assert database.get_reservation("PK-TEST01").admin_comment == ""


def test_a_failed_recording_is_queued_and_completed_by_a_retry_run(database):
    archive = RecordingArchive(available=False)
    workflow = make_workflow(database, archive)
    reserve(database)
    workflow.start("PK-TEST01")

    result = workflow.decide("PK-TEST01", "approve", "")

    assert (result.applied, result.archived) == (True, False)
    assert [r.reference for r in database.unarchived_confirmations()] == ["PK-TEST01"]
    assert workflow.retry_pending() == 0

    archive.available = True

    assert workflow.retry_pending() == 1
    assert database.unarchived_confirmations() == []
    assert workflow.retry_pending() == 0
    assert len(archive.archived) == 1


def test_a_pending_request_without_a_run_can_still_be_decided(database):
    desk = RecordingDesk(database)
    workflow = make_workflow(database, desk=desk)
    reserve(database)

    result = workflow.decide("PK-TEST01", "approve", "")

    assert result.applied
    assert desk.requests == ["PK-TEST01"]


def test_a_decided_request_without_a_run_is_reported_not_reapplied(database):
    workflow = make_workflow(database)
    reserve(database)
    database.decide("PK-TEST01", REJECTED, "old", NOW)

    result = workflow.decide("PK-TEST01", "approve", "")

    assert result == DecisionResult(applied=False, status=REJECTED, archived=True)


def test_confirmations_approved_before_the_workflow_existed_are_recorded_by_retry(database):
    archive = RecordingArchive()
    workflow = make_workflow(database, archive)
    reserve(database)
    database.decide("PK-TEST01", CONFIRMED, "", NOW)
    assert [r.reference for r in database.unarchived_confirmations()] == ["PK-TEST01"]

    stored = workflow.retry_pending()

    assert stored == 1
    assert [r.reference for r in archive.archived] == ["PK-TEST01"]
    assert database.unarchived_confirmations() == []


def test_a_run_paused_by_one_process_is_resumed_by_another(tmp_path):
    checkpoints = tmp_path / "workflow.sqlite"
    chat_db = ParkingDatabase.open(tmp_path / "parking.sqlite", SCHEMA_SQL, SEED_SQL)
    chat_desk = RecordingDesk(chat_db)
    chat_side = make_workflow(chat_db, desk=chat_desk, saver=open_saver(checkpoints))
    reserve(chat_db)
    chat_side.start("PK-TEST01")

    archive = RecordingArchive()
    admin_db = ParkingDatabase.open(tmp_path / "parking.sqlite", SCHEMA_SQL, SEED_SQL)
    admin_desk = RecordingDesk(admin_db)
    admin_side = make_workflow(admin_db, archive, admin_desk, open_saver(checkpoints))
    result = admin_side.decide("PK-TEST01", "approve", "")

    assert result.applied and result.archived
    assert chat_desk.requests == ["PK-TEST01"]
    assert admin_desk.requests == []
    assert [r.reference for r in archive.archived] == ["PK-TEST01"]
    assert chat_db.get_reservation("PK-TEST01").status == CONFIRMED


def test_the_graph_has_the_pipeline_nodes_from_request_to_recording(database):
    diagram = make_workflow(database).mermaid()

    for node in ("request_approval", "await_decision", "apply_decision", "record_reservation"):
        assert node in diagram


def test_the_checkpoint_store_is_created_with_its_folder(tmp_path):
    saver = open_saver(tmp_path / "deep" / "workflow.sqlite")

    assert (tmp_path / "deep" / "workflow.sqlite").exists()
    assert saver is not None


@pytest.mark.parametrize("verdict", ["approve", "reject"])
def test_every_decision_leaves_nothing_pending(database, verdict):
    workflow = make_workflow(database)
    reserve(database)
    workflow.start("PK-TEST01")

    workflow.decide("PK-TEST01", verdict, "")

    assert database.pending_reservations() == []


def test_the_workflow_diagram_needs_no_running_components():
    from parking_assistant.workflow import workflow_diagram

    diagram = workflow_diagram()

    assert "await_decision" in diagram and "close_request" in diagram
