import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from parking_assistant.reservation import TIME_FORMAT, ReservationDraft


class NoAvailabilityError(Exception):
    pass


@dataclass(frozen=True)
class WorkingHours:
    weekday: int
    open_time: str
    close_time: str


@dataclass(frozen=True)
class Zone:
    code: str
    description: str
    total_spaces: int
    occupied_spaces: int
    reservable_spaces: int
    hourly_price: float
    daily_price: float

    @property
    def free_spaces(self) -> int:
        return self.total_spaces - self.occupied_spaces


@dataclass(frozen=True)
class Reservation:
    reference: str
    first_name: str
    last_name: str
    car_number: str
    start_time: str
    end_time: str


class ParkingDatabase:
    """Dynamic data (hours, prices, availability, reservations) kept in SQLite."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, path: Path, schema_sql: Path, seed_sql: Path) -> "ParkingDatabase":
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        database = cls(connection)
        connection.executescript(schema_sql.read_text(encoding="utf-8"))
        if not database.zones():
            connection.executescript(seed_sql.read_text(encoding="utf-8"))
        return database

    def working_hours(self) -> list[WorkingHours]:
        rows = self._connection.execute(
            "SELECT weekday, open_time, close_time FROM working_hours ORDER BY weekday"
        )
        return [WorkingHours(**dict(row)) for row in rows]

    def zones(self) -> list[Zone]:
        rows = self._connection.execute(
            "SELECT code, description, total_spaces, occupied_spaces, reservable_spaces,"
            " hourly_price, daily_price FROM zones ORDER BY code"
        )
        return [Zone(**dict(row)) for row in rows]

    def free_reservable_spaces(self, start: str, end: str) -> int:
        capacity = self._connection.execute("SELECT SUM(reservable_spaces) FROM zones").fetchone()[
            0
        ]
        return (capacity or 0) - self._overlapping_reservations(start, end)

    def create_reservation(
        self, draft: ReservationDraft, reference: str, created_at: datetime
    ) -> Reservation:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if self.free_reservable_spaces(draft.start, draft.end) <= 0:
                raise NoAvailabilityError(f"{draft.start} - {draft.end}")
            self._connection.execute(
                "INSERT INTO reservations"
                " (reference, first_name, last_name, car_number, start_time, end_time, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    reference,
                    draft.first_name,
                    draft.last_name,
                    draft.car_number,
                    draft.start,
                    draft.end,
                    created_at.strftime(TIME_FORMAT),
                ),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")
        return self.get_reservation(reference)

    def get_reservation(self, reference: str) -> Reservation:
        row = self._connection.execute(
            "SELECT reference, first_name, last_name, car_number, start_time, end_time"
            " FROM reservations WHERE reference = ?",
            (reference,),
        ).fetchone()
        return Reservation(**dict(row))

    def _overlapping_reservations(self, start: str, end: str) -> int:
        return self._connection.execute(
            "SELECT COUNT(*) FROM reservations WHERE start_time < ? AND end_time > ?",
            (end, start),
        ).fetchone()[0]

    def close(self) -> None:
        self._connection.close()
