import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from parking_assistant.reservation import TIME_FORMAT, ReservationDraft

PENDING = "pending"
CONFIRMED = "confirmed"
REJECTED = "rejected"
ACTIVE_STATUSES = (PENDING, CONFIRMED)

RESERVATION_COLUMNS = (
    "reference, first_name, last_name, car_number, start_time, end_time, status, admin_comment,"
    " COALESCE(decided_at, '') AS decided_at"
)
# Reservations stored before approval existed count as confirmed and already reported.
LEGACY_COLUMNS = {
    "status": "TEXT NOT NULL DEFAULT 'confirmed'",
    "admin_comment": "TEXT NOT NULL DEFAULT ''",
    "request_message_id": "INTEGER",
    "customer_notified": "INTEGER NOT NULL DEFAULT 1",
    "decided_at": "TEXT",
    "archived": "INTEGER NOT NULL DEFAULT 1",
}
BUSY_TIMEOUT_MS = 5000


class NoAvailabilityError(Exception):
    pass


class UnknownReservationError(KeyError):
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
    status: str = PENDING
    admin_comment: str = ""
    decided_at: str = ""


class ParkingDatabase:
    """Dynamic data (hours, prices, availability, reservations) kept in SQLite."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, path: Path, schema_sql: Path, seed_sql: Path) -> "ParkingDatabase":
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        database = cls(connection)
        connection.executescript(schema_sql.read_text(encoding="utf-8"))
        database._add_missing_reservation_columns()
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
                " (reference, first_name, last_name, car_number, start_time, end_time, created_at,"
                " status, customer_notified) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    reference,
                    draft.first_name,
                    draft.last_name,
                    draft.car_number,
                    draft.start,
                    draft.end,
                    created_at.strftime(TIME_FORMAT),
                    PENDING,
                ),
            )
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")
        return self.get_reservation(reference)

    def get_reservation(self, reference: str) -> Reservation:
        row = self._connection.execute(
            f"SELECT {RESERVATION_COLUMNS} FROM reservations WHERE reference = ?", (reference,)
        ).fetchone()
        if row is None:
            raise UnknownReservationError(reference)
        return Reservation(**dict(row))

    def reservations(self, references: Sequence[str]) -> list[Reservation]:
        return self._select_reservations(
            f"reference IN ({_placeholders(references)})", tuple(references)
        )

    def pending_reservations(self) -> list[Reservation]:
        return self._select_reservations("status = ?", (PENDING,))

    def unsent_requests(self) -> list[Reservation]:
        return self._select_reservations("status = ? AND request_message_id IS NULL", (PENDING,))

    def record_request_message(self, reference: str, message_id: int) -> None:
        self._connection.execute(
            "UPDATE reservations SET request_message_id = ? WHERE reference = ?",
            (message_id, reference),
        )

    def request_message_id_for(self, reference: str) -> int:
        row = self._connection.execute(
            "SELECT request_message_id FROM reservations WHERE reference = ?", (reference,)
        ).fetchone()
        return row["request_message_id"] or 0 if row else 0

    def reference_for_request_message(self, message_id: int) -> str:
        row = self._connection.execute(
            "SELECT reference FROM reservations WHERE request_message_id = ?", (message_id,)
        ).fetchone()
        return row["reference"] if row else ""

    def decide(self, reference: str, status: str, comment: str, decided_at: datetime) -> bool:
        """Record the administrator's decision; False when the request was already decided."""
        cursor = self._connection.execute(
            "UPDATE reservations SET status = ?, admin_comment = ?, decided_at = ?"
            " WHERE reference = ? AND status = ?",
            (status, comment, decided_at.strftime(TIME_FORMAT), reference, PENDING),
        )
        return cursor.rowcount == 1

    def claim_decisions(self, references: Sequence[str]) -> list[Reservation]:
        """Return decided requests not yet reported to the customer and mark them reported."""
        rows = self._connection.execute(
            "UPDATE reservations SET customer_notified = 1"
            " WHERE status != ? AND customer_notified = 0"
            f" AND reference IN ({_placeholders(references)})"
            f" RETURNING {RESERVATION_COLUMNS}",
            (PENDING, *references),
        ).fetchall()
        return sorted((Reservation(**dict(row)) for row in rows), key=lambda r: r.reference)

    def unarchived_confirmations(self) -> list[Reservation]:
        return self._select_reservations("status = ? AND archived = 0", (CONFIRMED,))

    def mark_archived(self, reference: str) -> None:
        self._connection.execute(
            "UPDATE reservations SET archived = 1 WHERE reference = ?", (reference,)
        )

    def mark_notified(self, references: Sequence[str]) -> None:
        self._connection.execute(
            "UPDATE reservations SET customer_notified = 1"
            f" WHERE reference IN ({_placeholders(references)})",
            tuple(references),
        )

    def _select_reservations(self, condition: str, parameters: tuple) -> list[Reservation]:
        rows = self._connection.execute(
            f"SELECT {RESERVATION_COLUMNS} FROM reservations WHERE {condition}"
            " ORDER BY created_at, reference",
            parameters,
        )
        return [Reservation(**dict(row)) for row in rows]

    def _add_missing_reservation_columns(self) -> None:
        existing = {
            row["name"] for row in self._connection.execute("PRAGMA table_info(reservations)")
        }
        for column, definition in LEGACY_COLUMNS.items():
            if column not in existing:
                self._connection.execute(
                    f"ALTER TABLE reservations ADD COLUMN {column} {definition}"
                )
        if "archived" not in existing:
            self._connection.execute(
                "UPDATE reservations SET archived = 0"
                " WHERE status = 'confirmed' AND decided_at IS NOT NULL"
            )

    def _overlapping_reservations(self, start: str, end: str) -> int:
        return self._connection.execute(
            f"SELECT COUNT(*) FROM reservations WHERE status IN ({_placeholders(ACTIVE_STATUSES)})"
            " AND start_time < ? AND end_time > ?",
            (*ACTIVE_STATUSES, end, start),
        ).fetchone()[0]

    def close(self) -> None:
        self._connection.close()


def _placeholders(values: Sequence[object]) -> str:
    return ",".join("?" for _ in values)
