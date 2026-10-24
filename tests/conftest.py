import sqlite3
from pathlib import Path

import pytest

from parking_assistant.config import PROJECT_ROOT
from parking_assistant.database import ParkingDatabase
from parking_assistant.guardrails import PiiRedactor

SCHEMA_SQL = PROJECT_ROOT / "data" / "dynamic" / "schema.sql"
SEED_SQL = PROJECT_ROOT / "data" / "dynamic" / "seed.sql"
PUBLIC_CONTACTS = ("+370 5 210 4455", "info@skypark-central.com")


@pytest.fixture
def database_path(tmp_path: Path) -> Path:
    return tmp_path / "nested" / "parking.sqlite"


@pytest.fixture
def database(database_path: Path):
    db = ParkingDatabase.open(database_path, SCHEMA_SQL, SEED_SQL)
    yield db
    db.close()


@pytest.fixture
def full_database(database_path: Path):
    """Database where no space can be reserved."""
    db = ParkingDatabase.open(database_path, SCHEMA_SQL, SEED_SQL)
    with sqlite3.connect(database_path) as connection:
        connection.execute("UPDATE zones SET reservable_spaces = 0")
    yield db
    db.close()


@pytest.fixture(scope="session")
def redactor() -> PiiRedactor:
    return PiiRedactor(PUBLIC_CONTACTS)
