import pytest

from parking_assistant.archive import ArchiveError, UnconfiguredArchive, to_tool_arguments
from parking_assistant.database import CONFIRMED, Reservation


def test_tool_arguments_use_the_file_formats():
    reservation = Reservation(
        "PK-ABC123",
        "Anna",
        "Kowalska",
        "ABC123",
        "2026-10-25T09:00",
        "2026-10-25T17:30",
        CONFIRMED,
        "",
        "2026-10-24T22:15",
    )

    assert to_tool_arguments(reservation) == {
        "name": "Anna Kowalska",
        "car_number": "ABC123",
        "period_start": "2026-10-25 09:00",
        "period_end": "2026-10-25 17:30",
        "approved_at": "2026-10-24 22:15",
    }


def test_unconfigured_archive_fails_so_the_confirmation_stays_queued():
    with pytest.raises(ArchiveError, match="not configured"):
        UnconfiguredArchive().archive(Reservation("PK-ABC123", "A", "B", "ABC123", "s", "e"))
