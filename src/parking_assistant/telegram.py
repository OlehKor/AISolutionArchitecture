import httpx

from parking_assistant.approval import (
    AdminDeliveryError,
    ButtonPress,
    IncomingMessage,
    button_data,
    parse_button_data,
)

API_ROOT = "https://api.telegram.org"
SEND_TIMEOUT_SECONDS = 15
POLL_MARGIN_SECONDS = 10
UPDATE_KINDS = ["message", "callback_query"]


def decision_keyboard(reference: str) -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "Approve", "callback_data": button_data("approve", reference)},
                {"text": "Reject", "callback_data": button_data("reject", reference)},
            ]
        ]
    }


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

    def send(self, text: str, decision_for: str = "") -> int:
        payload: dict = {"chat_id": self._admin_chat_id, "text": text}
        if decision_for:
            payload["reply_markup"] = decision_keyboard(decision_for)
        result = self._call("sendMessage", payload, timeout=SEND_TIMEOUT_SECONDS)
        return int(result["message_id"])

    def poll(self) -> list[IncomingMessage | ButtonPress]:
        updates = self._call(
            "getUpdates",
            {
                "offset": self._offset,
                "timeout": self._poll_seconds,
                "allowed_updates": UPDATE_KINDS,
            },
            timeout=self._poll_seconds + POLL_MARGIN_SECONDS,
        )
        if updates:
            self._offset = max(update["update_id"] for update in updates) + 1
        return [event for update in updates if (event := _to_event(update))]

    def answer_press(self, callback_id: str, text: str) -> None:
        self._call(
            "answerCallbackQuery",
            {"callback_query_id": callback_id, "text": text},
            timeout=SEND_TIMEOUT_SECONDS,
        )

    def clear_buttons(self, message_id: int) -> None:
        self._call(
            "editMessageReplyMarkup",
            {
                "chat_id": self._admin_chat_id,
                "message_id": message_id,
                "reply_markup": {"inline_keyboard": []},
            },
            timeout=SEND_TIMEOUT_SECONDS,
        )

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


def _to_event(update: dict) -> IncomingMessage | ButtonPress | None:
    message = update.get("message", {})
    if message.get("text"):
        return IncomingMessage(
            chat_id=int(message["chat"]["id"]),
            text=message["text"],
            reply_to_message_id=int(message.get("reply_to_message", {}).get("message_id", 0)),
        )
    query = update.get("callback_query", {})
    if query.get("message"):
        action, reference = parse_button_data(query.get("data", ""))
        return ButtonPress(
            chat_id=int(query["message"]["chat"]["id"]),
            callback_id=str(query["id"]),
            message_id=int(query["message"]["message_id"]),
            action=action,
            reference=reference,
        )
    return None


class UnconfiguredChannel:
    """Stands in when no Telegram bot is configured, so requests stay pending."""

    def send(self, text: str, decision_for: str = "") -> int:
        raise AdminDeliveryError("Telegram is not configured")

    def poll(self) -> list[IncomingMessage | ButtonPress]:
        raise AdminDeliveryError("Telegram is not configured")

    def answer_press(self, callback_id: str, text: str) -> None:
        raise AdminDeliveryError("Telegram is not configured")

    def clear_buttons(self, message_id: int) -> None:
        raise AdminDeliveryError("Telegram is not configured")
