import hmac
import logging
import socket
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from parking_assistant.confirmed_log import ConfirmedReservation, ReservationFile, invalid_fields

logger = logging.getLogger(__name__)

MIN_TOKEN_LENGTH = 24
MAX_BODY_BYTES = 8 * 1024
FAILURE_LIMIT = 10
FAILURE_WINDOW_SECONDS = 60
HEALTH_PATH = "/health"
STARTUP_SECONDS = 10
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")


class FailureThrottle:
    """Blocks a client that keeps sending wrong credentials."""

    def __init__(
        self,
        limit: int = FAILURE_LIMIT,
        window_seconds: float = FAILURE_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def blocked(self, client: str) -> bool:
        return len(self._recent(client)) >= self._limit

    def record_failure(self, client: str) -> None:
        self._recent(client).append(self._clock())

    def _recent(self, client: str) -> deque[float]:
        failures = self._failures[client]
        while failures and self._clock() - failures[0] > self._window:
            failures.popleft()
        return failures


def has_valid_token(authorization: str, token: str) -> bool:
    scheme, _, presented = authorization.partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(
        presented.strip().encode(), token.encode()
    )


class BearerAuthMiddleware:
    """Rejects every request without the shared secret; only the health probe is public."""

    def __init__(self, app: ASGIApp, token: str, throttle: FailureThrottle) -> None:
        self._app = app
        self._token = token
        self._throttle = throttle

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        if scope["path"] == HEALTH_PATH:
            await JSONResponse({"status": "ok"})(scope, receive, send)
            return
        client = scope["client"][0] if scope.get("client") else "unknown"
        if self._throttle.blocked(client):
            await Response("Too many failed attempts", status_code=429)(scope, receive, send)
            return
        headers = dict(scope["headers"])
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        if not has_valid_token(authorization, self._token):
            self._throttle.record_failure(client)
            logger.warning("Rejected an unauthenticated request from %s", client)
            await Response("Unauthorized", status_code=401, headers={"WWW-Authenticate": "Bearer"})(
                scope, receive, send
            )
            return
        await self._app(scope, receive, send)


def build_mcp_server(
    reservations: ReservationFile, host: str, port: int, extra_hosts: tuple[str, ...] = ()
) -> FastMCP:
    server = FastMCP(
        "reservation-archive",
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
        max_request_body_size=MAX_BODY_BYTES,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[f"{name}:*" for name in (*LOOPBACK_HOSTS, host, *extra_hosts)],
            allowed_origins=[],
        ),
    )

    @server.tool()
    def record_confirmed_reservation(
        name: str, car_number: str, period_start: str, period_end: str, approved_at: str
    ) -> str:
        """Append an administrator-approved reservation to the confirmed reservations file.

        Times use the format YYYY-MM-DD HH:MM. Returns "written", or "duplicate" when the
        identical entry already exists, so retries are safe.
        """
        try:
            entry = ConfirmedReservation(
                name=name,
                car_number=car_number,
                start=period_start,
                end=period_end,
                approved_at=approved_at,
            )
        except ValidationError as error:
            raise ValueError(f"Invalid reservation fields: {invalid_fields(error)}") from None
        return "written" if reservations.append(entry) else "duplicate"

    return server


def build_app(
    token: str, reservations_path: Path, host: str, port: int, extra_hosts: tuple[str, ...] = ()
) -> Starlette:
    if len(token) < MIN_TOKEN_LENGTH:
        raise ValueError(f"The MCP token must have at least {MIN_TOKEN_LENGTH} characters.")
    server = build_mcp_server(ReservationFile(reservations_path), host, port, extra_hosts)
    app = server.streamable_http_app()
    app.add_middleware(BearerAuthMiddleware, token=token, throttle=FailureThrottle())
    return app


def serve(
    token: str, reservations_path: Path, host: str, port: int, extra_hosts: tuple[str, ...] = ()
) -> None:
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning("Listening on %s without TLS; put the server behind HTTPS.", host)
    app = build_app(token, reservations_path, host, port, extra_hosts)
    uvicorn.run(app, host=host, port=port)


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket() as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


@contextmanager
def background_server(
    token: str, reservations_path: Path, host: str = "127.0.0.1"
) -> Iterator[str]:
    """Run the server in a thread for tests and load tests; yields its URL."""
    port = free_port(host)
    config = uvicorn.Config(
        build_app(token, reservations_path, host, port), host=host, port=port, log_level="error"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + STARTUP_SECONDS
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("The MCP server did not start in time")
    try:
        yield f"http://{host}:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=STARTUP_SECONDS)
