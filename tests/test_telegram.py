import json

import httpx
import pytest

from parking_assistant.approval import AdminDeliveryError
from parking_assistant.telegram import TelegramChannel, UnconfiguredChannel

TOKEN = "123456:secret-token"


def make_channel(handler, poll_seconds: int = 30) -> TelegramChannel:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TelegramChannel(TOKEN, admin_chat_id=42, client=client, poll_seconds=poll_seconds)


def test_send_posts_to_the_admin_chat_and_returns_the_message_id():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 77}})

    message_id = make_channel(handler).send("hello")

    assert message_id == 77
    assert seen[0].url.path == f"/bot{TOKEN}/sendMessage"
    assert json.loads(seen[0].content) == {"chat_id": 42, "text": "hello"}


def test_poll_returns_text_messages_and_acknowledges_updates():
    requests: list[dict] = []
    updates = [
        {
            "update_id": 10,
            "message": {
                "message_id": 1,
                "chat": {"id": 42},
                "text": "approve",
                "reply_to_message": {"message_id": 77},
            },
        },
        {"update_id": 11, "message": {"message_id": 2, "chat": {"id": 42}, "photo": []}},
        {"update_id": 12, "message": {"message_id": 3, "chat": {"id": 7}, "text": "hi"}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"ok": True, "result": updates if len(requests) == 1 else []}
        )

    channel = make_channel(handler)
    messages = channel.poll()
    channel.poll()

    assert [(m.chat_id, m.text, m.reply_to_message_id) for m in messages] == [
        (42, "approve", 77),
        (7, "hi", 0),
    ]
    assert requests[0]["offset"] == 0
    assert requests[1]["offset"] == 13


def test_http_errors_become_delivery_errors_without_leaking_the_token():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"ok": False})

    with pytest.raises(AdminDeliveryError) as failure:
        make_channel(handler).send("hello")

    assert "401" in str(failure.value)
    assert TOKEN not in str(failure.value)
    assert failure.value.__cause__ is None


def test_network_errors_become_delivery_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(AdminDeliveryError, match="ConnectError"):
        make_channel(handler).poll()


def test_refusal_from_telegram_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "description": "chat not found"})

    with pytest.raises(AdminDeliveryError, match="chat not found"):
        make_channel(handler).send("hello")


def test_unconfigured_channel_refuses_to_send_or_poll():
    channel = UnconfiguredChannel()

    with pytest.raises(AdminDeliveryError):
        channel.send("hello")
    with pytest.raises(AdminDeliveryError):
        channel.poll()
