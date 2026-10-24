import sqlite3

import pytest

from fakes import NOW
from parking_assistant.database import (
    CONFIRMED,
    PENDING,
    REJECTED,
    NoAvailabilityError,
    ParkingDatabase,
    Reservation,
    UnknownReservationError,
)
from parking_assistant.reservation import ReservationDraft

DRAFT = ReservationDraft(
    first_name="Anna",
    last_name="Kowalska",
    car_number="ABC123",
    start="2026-10-25T09:00",
    end="2026-10-25T17:00",
)


def test_open_creates_file_and_seeds_reference_data(database, database_path):
    assert database_path.exists()
    assert len(database.working_hours()) == 7
    assert {zone.code for zone in database.zones()} == {"A", "B", "C", "EV"}


def test_zone_free_spaces_are_total_minus_occupied(database):
    zone_c = next(zone for zone in database.zones() if zone.code == "C")
    assert zone_c.free_spaces == 87


def test_create_reservation_stores_it_and_reduces_free_spaces(database):
    before = database.free_reservable_spaces(DRAFT.start, DRAFT.end)

    reservation = database.create_reservation(DRAFT, "PK-TEST01", NOW)

    assert reservation == Reservation(
        "PK-TEST01", "Anna", "Kowalska", "ABC123", DRAFT.start, DRAFT.end
    )
    assert database.free_reservable_spaces(DRAFT.start, DRAFT.end) == before - 1


def test_reservation_does_not_affect_other_periods(database):
    database.create_reservation(DRAFT, "PK-TEST01", NOW)
    assert database.free_reservable_spaces("2026-10-26T09:00", "2026-10-26T17:00") == 150


def test_create_reservation_fails_when_nothing_is_reservable(full_database):
    with pytest.raises(NoAvailabilityError):
        full_database.create_reservation(DRAFT, "PK-TEST01", NOW)


def test_new_reservations_start_pending_and_unsent(database):
    reservation = database.create_reservation(DRAFT, "PK-TEST01", NOW)

    assert reservation.status == PENDING
    assert [r.reference for r in database.pending_reservations()] == ["PK-TEST01"]
    assert [r.reference for r in database.unsent_requests()] == ["PK-TEST01"]


def test_recorded_request_message_is_found_again_and_no_longer_unsent(database):
    database.create_reservation(DRAFT, "PK-TEST01", NOW)

    database.record_request_message("PK-TEST01", 777)

    assert database.reference_for_request_message(777) == "PK-TEST01"
    assert database.reference_for_request_message(1) == ""
    assert database.unsent_requests() == []


def test_decision_is_applied_only_once(database):
    database.create_reservation(DRAFT, "PK-TEST01", NOW)

    assert database.decide("PK-TEST01", CONFIRMED, "ok", NOW)
    assert not database.decide("PK-TEST01", REJECTED, "changed my mind", NOW)

    decided = database.get_reservation("PK-TEST01")
    assert (decided.status, decided.admin_comment) == (CONFIRMED, "ok")
    assert database.pending_reservations() == []


def test_rejected_reservations_release_their_space(database):
    database.create_reservation(DRAFT, "PK-TEST01", NOW)
    assert database.free_reservable_spaces(DRAFT.start, DRAFT.end) == 149

    database.decide("PK-TEST01", REJECTED, "closed", NOW)

    assert database.free_reservable_spaces(DRAFT.start, DRAFT.end) == 150


def test_decisions_are_claimed_once_and_only_for_the_given_references(database):
    database.create_reservation(DRAFT, "PK-TEST01", NOW)
    assert database.claim_decisions(["PK-TEST01"]) == []
    database.decide("PK-TEST01", CONFIRMED, "", NOW)

    assert database.claim_decisions(["PK-OTHER1"]) == []
    claimed = database.claim_decisions(["PK-TEST01"])
    assert [(r.reference, r.status) for r in claimed] == [("PK-TEST01", CONFIRMED)]
    assert database.claim_decisions(["PK-TEST01"]) == []


def test_unknown_reference_raises(database):
    with pytest.raises(UnknownReservationError):
        database.get_reservation("PK-NOPE00")


def test_database_without_approval_columns_is_migrated(database_path, tmp_path):
    from conftest import SCHEMA_SQL, SEED_SQL

    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE reservations (reference TEXT PRIMARY KEY, first_name TEXT NOT NULL,"
            " last_name TEXT NOT NULL, car_number TEXT NOT NULL, start_time TEXT NOT NULL,"
            " end_time TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO reservations VALUES ('PK-OLD001', 'Old', 'Booking', 'ABC123',"
            " '2026-10-25T09:00', '2026-10-25T10:00', '2026-10-24T09:00')"
        )

    migrated = ParkingDatabase.open(database_path, SCHEMA_SQL, SEED_SQL)

    legacy = migrated.get_reservation("PK-OLD001")
    assert legacy.status == CONFIRMED
    assert migrated.pending_reservations() == []
    assert migrated.claim_decisions(["PK-OLD001"]) == []
    fresh = migrated.create_reservation(DRAFT, "PK-NEW001", NOW)
    assert fresh.status == PENDING
    assert [r.reference for r in migrated.pending_reservations()] == ["PK-NEW001"]
    migrated.close()
