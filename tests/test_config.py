from pydantic import SecretStr

from parking_assistant.config import Settings


def test_defaults_point_to_project_data():
    settings = Settings()

    assert settings.retrieval_k == 4
    assert settings.static_data_dir.is_dir()
    assert settings.schema_sql.is_file()
    assert settings.seed_sql.is_file()
    assert "info@skypark-central.com" in settings.public_contacts


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("PARKING_RETRIEVAL_K", "7")
    monkeypatch.setenv("PARKING_CHAT_MODEL", "gpt-4o")

    settings = Settings()

    assert settings.retrieval_k == 7
    assert settings.chat_model == "gpt-4o"


def test_telegram_is_configured_only_with_token_and_chat_id():
    assert not Settings(
        telegram_bot_token=SecretStr(""), telegram_admin_chat_id=0
    ).telegram_configured
    assert not Settings(
        telegram_bot_token=SecretStr("123:abc"), telegram_admin_chat_id=0
    ).telegram_configured
    assert Settings(
        telegram_bot_token=SecretStr("123:abc"), telegram_admin_chat_id=42
    ).telegram_configured


def test_bot_token_is_hidden_when_settings_are_printed():
    assert "123:abc" not in repr(Settings(telegram_bot_token=SecretStr("123:abc")))


def test_mcp_defaults_are_local_and_the_token_is_hidden():
    settings = Settings(mcp_token=SecretStr("super-secret-token"))

    assert settings.mcp_url == "http://127.0.0.1:8765/mcp"
    assert settings.reservations_file.name == "confirmed_reservations.txt"
    assert "super-secret-token" not in repr(settings)


def test_the_mcp_url_can_be_overridden_for_containers_and_extra_hosts_are_split():
    default = Settings(mcp_token=SecretStr("x"))
    container = Settings(
        mcp_server_url="http://mcp:8765/mcp", mcp_allowed_hosts=" mcp , parking.internal ,"
    )

    assert default.mcp_url == "http://127.0.0.1:8765/mcp"
    assert default.mcp_extra_hosts == ()
    assert container.mcp_url == "http://mcp:8765/mcp"
    assert container.mcp_extra_hosts == ("mcp", "parking.internal")
