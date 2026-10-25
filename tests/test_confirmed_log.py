import os
import stat
import threading

import pytest
from pydantic import ValidationError

from parking_assistant.confirmed_log import (
    ConfirmedReservation,
    ReservationFile,
    format_entry,
    invalid_fields,
    timestamp_for_file,
)


def entry(**overrides) -> ConfirmedReservation:
    values = {
        "name": "Anna Kowalska",
        "car_number": "ABC123",
        "start": "2026-10-25 09:00",
        "end": "2026-10-25 17:30",
        "approved_at": "2026-10-24 22:15",
    }
    return ConfirmedReservation(**{**values, **overrides})


def test_entry_is_formatted_as_name_car_period_approval_time():
    assert format_entry(entry()) == (
        "Anna Kowalska | ABC123 | 2026-10-25 09:00 - 2026-10-25 17:30 | 2026-10-24 22:15"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "Anna | Kowalska"),
        ("name", "Anna\nEvil | X"),
        ("name", "=cmd()"),
        ("name", "A" * 200),
        ("car_number", "abc123"),
        ("car_number", "AB"),
        ("car_number", "ABC 123\n"),
        ("start", "2026-10-25"),
        ("start", "25/10/2026 09:00"),
        ("end", "2026-10-25 09:00"),
        ("approved_at", "2026-13-40 99:99"),
    ],
)
def test_values_that_could_corrupt_the_file_are_rejected(field, value):
    with pytest.raises(ValidationError):
        entry(**{field: value})


def test_unicode_names_and_hyphens_are_accepted():
    assert format_entry(entry(name="Žaneta O'Brien-Šimkutė")).startswith("Žaneta O'Brien-Šimkutė |")


def test_invalid_fields_name_the_fields_without_echoing_values():
    with pytest.raises(ValidationError) as failure:
        entry(name="Anna | X", car_number="x")

    assert invalid_fields(failure.value) == "car_number, name"


def test_appending_creates_the_file_privately_and_adds_one_line(tmp_path):
    reservations = ReservationFile(tmp_path / "nested" / "confirmed.txt")

    assert reservations.append(entry())

    path = tmp_path / "nested" / "confirmed.txt"
    assert path.read_text(encoding="utf-8") == format_entry(entry()) + "\n"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_identical_entry_is_not_written_twice(tmp_path):
    reservations = ReservationFile(tmp_path / "confirmed.txt")

    assert reservations.append(entry())
    assert not reservations.append(entry())
    assert reservations.append(entry(car_number="XYZ789"))

    assert len((tmp_path / "confirmed.txt").read_text().splitlines()) == 2


def test_concurrent_writers_never_corrupt_or_lose_entries(tmp_path):
    reservations = ReservationFile(tmp_path / "confirmed.txt")
    plates = [f"CAR{number:03d}" for number in range(40)]

    threads = [
        threading.Thread(target=reservations.append, args=(entry(car_number=plate),))
        for plate in plates
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = (tmp_path / "confirmed.txt").read_text().splitlines()
    assert sorted(line.split(" | ")[1] for line in lines) == plates
    assert all(line.count(" | ") == 3 for line in lines)


def test_database_timestamps_are_converted_for_the_file():
    assert timestamp_for_file("2026-10-25T09:00") == "2026-10-25 09:00"
