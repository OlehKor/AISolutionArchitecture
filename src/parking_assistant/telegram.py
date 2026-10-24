import httpx

from parking_assistant.approval import AdminDeliveryError, IncomingMessage

API_ROOT = "https://api.telegram.org"
SEND_TIMEOUT_SECONDS = 15
POLL_MARGIN_SECONDS = 10


class TelegramChannel:
    """Talks to the administrator through a Telegram bot (Bot API over HTTPS)."""

    def __init__(
        self, token: str, admin_chat_id: int, client: httpx.Client, poll_seconds: int = 30
    ) -> None:
        self._base_url = f"{API_ROOT}/bot{token}"
        self._admin_chat_id = admin_chat_id
        self._client = client
        self._poll_seconds = poll_seconds
        self._offset = 0

    def send(self, text: str) -> int:
        result = self._call(
            "sendMessage",
            {"chat_id": self._admin_chat_id, "text": text},
            timeout=SEND_TIMEOUT_SECONDS,
        )
        return int(result["message_id"])

    def poll(self) -> list[IncomingMessage]:
        updates = self._call(
            "getUpdates",
            {"offset": self._offset, "timeout": self._poll_seconds, "allowed_updates": ["message"]},
            timeout=self._poll_seconds + POLL_MARGIN_SECONDS,
        )
        if updates:
            self._offset = max(update["update_id"] for update in updates) + 1
        return [_to_message(update["message"]) for update in updates if _has_text(update)]

    def _call(self, method: str, payload: dict, timeout: float):
        try:
            response = self._client.post(
                f"{self._base_url}/{method}", json=payload, timeout=timeout
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as error:
            # The request URL contains the bot token, so it must not leak through the chained error.
            raise AdminDeliveryError(
                f"Telegram {method} failed with HTTP {error.response.status_code}"
            ) from None
        except httpx.HTTPError as error:
            raise AdminDeliveryError(f"Telegram {method} failed: {type(error).__name__}") from None
        if not body.get("ok"):
            raise AdminDeliveryError(
                f"Telegram {method} was refused: {body.get('description', '')}"
            )
        return body["result"]


def _has_text(update: dict) -> bool:
    return bool(update.get("message", {}).get("text"))


def _to_message(message: dict) -> IncomingMessage:
    return IncomingMessage(
        chat_id=int(message["chat"]["id"]),
        text=message["text"],
        reply_to_message_id=int(message.get("reply_to_message", {}).get("message_id", 0)),
    )


class UnconfiguredChannel:
    """Stands in when no Telegram bot is configured, so requests stay pending."""

    def send(self, text: str) -> int:
        raise AdminDeliveryError("Telegram is not configured")

    def poll(self) -> list[IncomingMessage]:
        raise AdminDeliveryError("Telegram is not configured")
