import pytest

from fakes import NOW
from parking_assistant.database import NoAvailabilityError, Reservation
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
