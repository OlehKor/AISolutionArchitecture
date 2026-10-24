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
