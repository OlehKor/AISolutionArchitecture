import fcntl
import os
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator, model_validator

from parking_assistant.reservation import NAME_PATTERN, PLATE_PATTERN

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M"
MAX_NAME_LENGTH = 101
FILE_MODE = 0o600
SEPARATOR = " | "


def _check_timestamp(value: str) -> str:
    if datetime.strptime(value, TIMESTAMP_FORMAT).strftime(TIMESTAMP_FORMAT) != value:
        raise ValueError("timestamp must look like 2026-10-25 09:00")
    return value


class ConfirmedReservation(BaseModel, frozen=True):
    """A confirmed reservation; the strict patterns keep delimiters and newlines out of the file."""

    name: str
    car_number: str
    start: str
    end: str
    approved_at: str

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if len(value) > MAX_NAME_LENGTH or not NAME_PATTERN.fullmatch(value):
            raise ValueError("name must contain letters, spaces, apostrophes or hyphens only")
        return value

    @field_validator("car_number")
    @classmethod
    def _valid_car_number(cls, value: str) -> str:
        if not PLATE_PATTERN.fullmatch(value):
            raise ValueError("car number must be 3 to 10 capital letters or digits")
        return value

    @field_validator("start", "end", "approved_at")
    @classmethod
    def _valid_timestamp(cls, value: str) -> str:
        return _check_timestamp(value)

    @model_validator(mode="after")
    def _end_after_start(self) -> "ConfirmedReservation":
        if self.end <= self.start:
            raise ValueError("the departure must be after the arrival")
        return self


def format_entry(entry: ConfirmedReservation) -> str:
    period = f"{entry.start} - {entry.end}"
    return SEPARATOR.join([entry.name, entry.car_number, period, entry.approved_at])


def invalid_fields(error: ValidationError) -> str:
    """Names of the invalid fields only, so rejected values are never echoed back."""
    return ", ".join(sorted({str(item["loc"][0]) for item in error.errors() if item["loc"]}))


class ReservationFile:
    """Append-only text file of confirmed reservations, safe for concurrent writers."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, entry: ConfirmedReservation) -> bool:
        """Write the entry; False when the identical entry is already in the file."""
        line = format_entry(entry)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._path, os.O_RDWR | os.O_CREAT | os.O_APPEND, FILE_MODE)
        with os.fdopen(descriptor, "a+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            file.seek(0)
            if line in file.read().splitlines():
                return False
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())
        return True


def timestamp_for_file(stored: str) -> str:
    """Convert the database format (2026-10-25T09:00) into the file format."""
    return stored.replace("T", " ", 1)
