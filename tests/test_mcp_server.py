import pytest
from starlette.testclient import TestClient

from fakes import NOW
from parking_assistant.archive import ArchiveError, McpReservationArchive
from parking_assistant.database import CONFIRMED, Reservation
from parking_assistant.mcp_server import (
    FailureThrottle,
    background_server,
    build_app,
    free_port,
    has_valid_token,
)

TOKEN = "t" * 32


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def confirmed(**overrides) -> Reservation:
    values = {
        "reference": "PK-ABC123",
        "first_name": "Anna",
        "last_name": "Kowalska",
        "car_number": "ABC123",
        "start_time": "2026-10-25T09:00",
        "end_time": "2026-10-25T17:30",
        "status": CONFIRMED,
        "decided_at": NOW.strftime("%Y-%m-%dT%H:%M"),
    }
    return Reservation(**{**values, **overrides})


@pytest.fixture
def server(tmp_path):
    path = tmp_path / "confirmed.txt"
    with background_server(TOKEN, path) as url:
        yield url, path


def test_throttle_blocks_a_client_after_too_many_failures_and_recovers():
    clock = FakeClock()
    throttle = FailureThrottle(limit=3, window_seconds=60, clock=clock)

    for _ in range(3):
        assert not throttle.blocked("10.0.0.1")
        throttle.record_failure("10.0.0.1")

    assert throttle.blocked("10.0.0.1")
    assert not throttle.blocked("10.0.0.2")
    clock.now = 61
    assert not throttle.blocked("10.0.0.1")


@pytest.mark.parametrize(
    ("header", "valid"),
    [
        (f"Bearer {TOKEN}", True),
        (f"bearer {TOKEN}", True),
        ("Bearer wrong", False),
        (TOKEN, False),
        (f"Basic {TOKEN}", False),
        ("", False),
    ],
)
def test_only_the_exact_bearer_token_is_accepted(header, valid):
    assert has_valid_token(header, TOKEN) is valid


def test_server_refuses_to_start_with_a_weak_token(tmp_path):
    with pytest.raises(ValueError, match="at least"):
        build_app("short", tmp_path / "confirmed.txt", "127.0.0.1", 8765)


def test_unauthenticated_requests_are_rejected_but_the_health_probe_is_open(tmp_path):
    client = TestClient(build_app(TOKEN, tmp_path / "confirmed.txt", "127.0.0.1", 8765))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/mcp", json={}).status_code == 401
    wrong = client.post("/mcp", json={}, headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    assert wrong.headers["www-authenticate"] == "Bearer"
    assert not (tmp_path / "confirmed.txt").exists()


def test_repeated_wrong_tokens_lock_the_client_out_even_with_the_right_one(tmp_path):
    client = TestClient(build_app(TOKEN, tmp_path / "confirmed.txt", "127.0.0.1", 8765))

    statuses = [
        client.post("/mcp", json={}, headers={"Authorization": "Bearer nope"}).status_code
        for _ in range(11)
    ]
    locked = client.post("/mcp", json={}, headers={"Authorization": f"Bearer {TOKEN}"})

    assert statuses[:10] == [401] * 10
    assert statuses[10] == 429
    assert locked.status_code == 429


def test_confirmed_reservation_is_written_through_the_mcp_tool(server):
    url, path = server

    McpReservationArchive(url, TOKEN).archive(confirmed())

    assert path.read_text(encoding="utf-8").splitlines() == [
        "Anna Kowalska | ABC123 | 2026-10-25 09:00 - 2026-10-25 17:30 | 2026-10-24 10:00"
    ]


def test_retrying_the_same_reservation_does_not_duplicate_the_entry(server):
    url, path = server
    archive = McpReservationArchive(url, TOKEN)

    archive.archive(confirmed())
    archive.archive(confirmed())

    assert len(path.read_text().splitlines()) == 1


def test_wrong_token_is_refused_and_nothing_is_written(server):
    url, path = server

    with pytest.raises(ArchiveError):
        McpReservationArchive(url, "w" * 32, attempts=1).archive(confirmed())

    assert not path.exists()


def test_data_that_could_break_the_file_format_is_rejected_by_the_server(server):
    url, path = server
    archive = McpReservationArchive(url, TOKEN, attempts=1)

    with pytest.raises(ArchiveError, match="rejected"):
        archive.archive(confirmed(first_name="Anna | Admin", last_name="X\nEvil"))

    assert not path.exists()


def test_unreachable_server_is_retried_then_reported():
    pauses: list[float] = []
    archive = McpReservationArchive(
        f"http://127.0.0.1:{free_port()}/mcp", TOKEN, attempts=3, pause=pauses.append
    )

    with pytest.raises(ArchiveError, match="unavailable"):
        archive.archive(confirmed())

    assert pauses == [1, 1]


def test_extra_hosts_are_accepted_by_the_host_header_check(tmp_path):
    app = build_app(TOKEN, tmp_path / "c.txt", "0.0.0.0", 8765, extra_hosts=("mcp",))
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}

    with TestClient(app) as client:
        allowed = client.post("/mcp", json=body, headers={**headers, "Host": "mcp:8765"})
        stranger = client.post("/mcp", json=body, headers={**headers, "Host": "evil.example:8765"})

    assert allowed.status_code != 421
    assert stranger.status_code == 421
