import asyncio
import time
from collections.abc import Callable
from typing import Protocol

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from parking_assistant.confirmed_log import timestamp_for_file
from parking_assistant.database import Reservation

TOOL_NAME = "record_confirmed_reservation"


class ArchiveError(Exception):
    """The confirmed reservation could not be stored; it will be retried later."""


class ArchiveRejectedError(Exception):
    """The server was reachable but refused the data; retrying will not help."""


class ReservationArchive(Protocol):
    def archive(self, reservation: Reservation) -> None: ...


def to_tool_arguments(reservation: Reservation) -> dict[str, str]:
    return {
        "name": f"{reservation.first_name} {reservation.last_name}",
        "car_number": reservation.car_number,
        "period_start": timestamp_for_file(reservation.start_time),
        "period_end": timestamp_for_file(reservation.end_time),
        "approved_at": timestamp_for_file(reservation.decided_at),
    }


class McpReservationArchive:
    """Writes confirmed reservations through the MCP server's tool, authenticating with a token."""

    def __init__(
        self,
        url: str,
        token: str,
        timeout_seconds: float = 10,
        attempts: int = 2,
        pause: Callable[[float], None] = time.sleep,
    ) -> None:
        self._url = url
        self._headers = {"Authorization": f"Bearer {token}"}
        self._timeout = timeout_seconds
        self._attempts = attempts
        self._pause = pause

    def archive(self, reservation: Reservation) -> None:
        arguments = to_tool_arguments(reservation)
        last_failure = ""
        for attempt in range(1, self._attempts + 1):
            try:
                asyncio.run(asyncio.wait_for(self._call_tool(arguments), self._timeout))
                return
            except ArchiveRejectedError as error:
                raise ArchiveError(str(error)) from None
            except Exception as error:
                # The MCP client raises assorted network and exception-group types; retry then translate.
                last_failure = f"MCP server unavailable ({type(error).__name__})"
                if attempt < self._attempts:
                    self._pause(1)
        raise ArchiveError(last_failure)

    async def _call_tool(self, arguments: dict[str, str]) -> None:
        async with (
            httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as http_client,
            streamable_http_client(
                self._url, http_client=http_client, terminate_on_close=False
            ) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(TOOL_NAME, arguments)
        if result.isError:
            detail = " ".join(getattr(part, "text", "") for part in result.content)
            raise ArchiveRejectedError(f"The MCP server rejected the reservation: {detail}")


class UnconfiguredArchive:
    """Stands in when no MCP token is set, so confirmations wait until it is configured."""

    def archive(self, reservation: Reservation) -> None:
        raise ArchiveError("The MCP server is not configured")
