import re

from pydantic import SecretStr

from parking_assistant.archive import McpReservationArchive, UnconfiguredArchive
from parking_assistant.composition import (
    build_admin_channel,
    build_archive,
    new_reservation_reference,
)
from parking_assistant.config import Settings
from parking_assistant.telegram import TelegramChannel, UnconfiguredChannel


def test_reservation_references_have_a_readable_format():
    assert re.fullmatch(r"PK-[0-9A-F]{6}", new_reservation_reference())


def test_reservation_references_are_not_repeated():
    assert len({new_reservation_reference() for _ in range(50)}) == 50


def test_without_telegram_settings_requests_go_nowhere():
    channel = build_admin_channel(
        Settings(telegram_bot_token=SecretStr(""), telegram_admin_chat_id=0)
    )

    assert isinstance(channel, UnconfiguredChannel)


def test_telegram_settings_build_the_telegram_channel():
    settings = Settings(telegram_bot_token=SecretStr("123:abc"), telegram_admin_chat_id=42)

    assert isinstance(build_admin_channel(settings), TelegramChannel)


def test_token_without_chat_id_still_builds_the_telegram_channel_for_setup():
    settings = Settings(telegram_bot_token=SecretStr("123:abc"), telegram_admin_chat_id=0)

    assert isinstance(build_admin_channel(settings), TelegramChannel)


def test_archive_is_unconfigured_without_an_mcp_token_and_mcp_with_one():
    assert isinstance(build_archive(Settings(mcp_token=SecretStr(""))), UnconfiguredArchive)
    assert isinstance(build_archive(Settings(mcp_token=SecretStr("x" * 32))), McpReservationArchive)
