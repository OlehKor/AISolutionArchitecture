import threading
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from fakes import MaskingRedactor, RecordingChannel, StaticKnowledge
from parking_assistant import __main__ as cli
from parking_assistant.approval import AdminDeliveryError, IncomingMessage
from parking_assistant.config import Settings
from parking_assistant.guardrails import Guardrails

UNCONFIGURED = Settings(telegram_bot_token=SecretStr(""), telegram_admin_chat_id=0)
CONFIGURED = Settings(telegram_bot_token=SecretStr("123:abc"), telegram_admin_chat_id=42)


class ClosableDatabase:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class RecordingAssistant:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def reply(self, session_id: str, text: str) -> str:
        self.messages.append(text)
        return f"echo: {text}"

    def announcements(self, session_id: str, database=None) -> str:
        return ""


@pytest.fixture(autouse=True)
def no_real_database(monkeypatch):
    monkeypatch.setattr(cli, "build_database", lambda settings: ClosableDatabase())


def test_cli_parses_chat_ingest_and_evaluate_commands():
    assert cli.parse_args(["chat"]).command == "chat"
    assert cli.parse_args(["ingest"]).command == "ingest"
    evaluate = cli.parse_args(["evaluate", "--end-to-end"])
    assert (evaluate.command, evaluate.end_to_end) == ("evaluate", True)


def test_cli_requires_a_command_and_defaults_to_offline_evaluation():
    with pytest.raises(SystemExit):
        cli.parse_args([])
    assert cli.parse_args(["evaluate"]).end_to_end is False


def test_chat_loop_skips_blank_lines_and_stops_on_exit(monkeypatch, capsys):
    assistant = RecordingAssistant()
    monkeypatch.setattr(cli, "build_assistant", lambda settings: assistant)
    typed = iter(["hello", "   ", "exit", "never reached"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))

    cli.chat(UNCONFIGURED)

    assert assistant.messages == ["hello"]
    assert "Assistant: echo: hello" in capsys.readouterr().out


def test_chat_loop_ends_on_end_of_input(monkeypatch):
    assistant = RecordingAssistant()
    monkeypatch.setattr(cli, "build_assistant", lambda settings: assistant)

    def end_of_input(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", end_of_input)

    cli.chat(UNCONFIGURED)

    assert assistant.messages == []


def test_cli_parses_the_admin_command():
    assert cli.parse_args(["admin"]).command == "admin"


def test_admin_command_refuses_to_start_without_telegram_settings():
    with pytest.raises(SystemExit) as stopped:
        cli.admin(UNCONFIGURED)

    assert "PARKING_TELEGRAM_BOT_TOKEN" in str(stopped.value)


def test_admin_command_resends_unsent_requests_and_then_serves(monkeypatch):
    calls: list[str] = []

    class FakeDesk:
        def resend_unsent(self) -> int:
            calls.append("resend")
            return 2

    class FakeAgent:
        def serve(self) -> None:
            calls.append("serve")

    service = SimpleNamespace(desk=FakeDesk(), agent=FakeAgent())
    monkeypatch.setattr(cli, "build_database", lambda settings: object())
    monkeypatch.setattr(cli, "build_admin_service", lambda settings, database: service)

    cli.admin(CONFIGURED)

    assert calls == ["resend", "serve"]


TOKEN_ONLY = Settings(telegram_bot_token=SecretStr("123:abc"), telegram_admin_chat_id=0)


def test_admin_command_without_chat_id_prints_the_chat_id_and_exits(monkeypatch, capsys):
    channel = RecordingChannel([[], [IncomingMessage(chat_id=555, text="hi")]])
    monkeypatch.setattr(cli, "build_admin_channel", lambda settings: channel)

    cli.admin(TOKEN_ONLY)

    assert "Your chat id is 555" in capsys.readouterr().out


def test_admin_setup_mode_reports_telegram_failures(monkeypatch):
    channel = RecordingChannel([AdminDeliveryError("Telegram getUpdates failed with HTTP 401")])
    monkeypatch.setattr(cli, "build_admin_channel", lambda settings: channel)

    with pytest.raises(SystemExit) as stopped:
        cli.admin(TOKEN_ONLY)

    assert "401" in str(stopped.value)


def test_watcher_prints_decisions_as_they_arrive_and_closes_its_database(capsys):
    stop = threading.Event()
    database = ClosableDatabase()

    class AnnouncingAssistant:
        calls = 0

        def announcements(self, session_id: str, db) -> str:
            self.calls += 1
            if self.calls == 2:
                stop.set()
                return "Good news: reservation PK-AAAAAA was confirmed."
            return ""

    cli.announce_decisions(AnnouncingAssistant(), "s1", lambda: database, stop, interval=0)

    assert "Assistant: Good news: reservation PK-AAAAAA was confirmed." in capsys.readouterr().out
    assert database.closed


def test_watcher_survives_a_failed_check():
    stop = threading.Event()

    class FlakyAssistant:
        calls = 0

        def announcements(self, session_id: str, db) -> str:
            self.calls += 1
            if self.calls == 2:
                stop.set()
            raise RuntimeError("database is locked")

    assistant = FlakyAssistant()

    cli.announce_decisions(assistant, "s1", ClosableDatabase, stop, interval=0)

    assert assistant.calls == 2


def test_cli_parses_the_mcp_server_command():
    assert cli.parse_args(["mcp-server"]).command == "mcp-server"


def test_mcp_server_command_refuses_a_missing_token():
    with pytest.raises(SystemExit) as stopped:
        cli.mcp_server(Settings(mcp_token=SecretStr("")))

    assert "at least" in str(stopped.value)


def test_mcp_server_command_serves_with_the_configured_settings(monkeypatch):
    started: list[tuple] = []
    monkeypatch.setattr(cli, "serve", lambda *args: started.append(args))
    settings = Settings(mcp_token=SecretStr("x" * 32), mcp_port=9999)

    cli.mcp_server(settings)

    assert started == [("x" * 32, settings.reservations_file, "127.0.0.1", 9999, ())]


def test_chat_greets_the_customer_before_the_first_prompt(monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_assistant", lambda settings: RecordingAssistant())
    monkeypatch.setattr("builtins.input", lambda prompt="": "exit")

    cli.chat(UNCONFIGURED)

    output = capsys.readouterr().out
    assert output.startswith("Assistant: Hello! I'm the SkyPark Central parking assistant.")
    assert "reserve a parking space" in output


@pytest.mark.parametrize(
    ("command", "level"), [("chat", "WARNING"), ("admin", "INFO"), ("ingest", "INFO")]
)
def test_only_the_customer_chat_hides_informational_log_lines(monkeypatch, command, level):
    configured: list[int] = []
    monkeypatch.setattr(
        cli.logging, "basicConfig", lambda **kwargs: configured.append(kwargs["level"])
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(cli, "get_settings", lambda: UNCONFIGURED)
    monkeypatch.setattr(cli, command, lambda *args: None)

    cli.main([command])

    assert configured == [getattr(cli.logging, level)]


def test_cli_parses_the_new_commands_and_their_defaults():
    load = cli.parse_args(["loadtest"])

    assert cli.parse_args(["run"]).command == "run"
    assert cli.parse_args(["graph"]).command == "graph"
    assert (load.sessions, load.concurrency, load.mcp_requests) == (24, 8, 100)
    assert load.mcp_concurrency == [1, 4, 16] and load.real_llm is False


class FakeThread:
    started: list[str] = []

    def __init__(self, target, args=(), daemon=False, name=""):
        self.name = name

    def start(self) -> None:
        FakeThread.started.append(self.name)


def test_run_starts_the_background_components_that_are_configured(monkeypatch, capsys):
    FakeThread.started = []
    chats: list[object] = []
    monkeypatch.setattr(cli.threading, "Thread", FakeThread)
    monkeypatch.setattr(cli, "chat", chats.append)
    settings = Settings(
        telegram_bot_token=SecretStr("123:abc"),
        telegram_admin_chat_id=42,
        mcp_token=SecretStr("m" * 32),
    )

    cli.run_all(settings)

    assert FakeThread.started == ["mcp", "admin"]
    assert chats == [settings]
    assert "MCP server started" in capsys.readouterr().out


def test_run_says_what_is_missing_instead_of_starting_it(monkeypatch, capsys):
    FakeThread.started = []
    monkeypatch.setattr(cli.threading, "Thread", FakeThread)
    monkeypatch.setattr(cli, "chat", lambda settings: None)

    cli.run_all(Settings(telegram_bot_token=SecretStr(""), mcp_token=SecretStr("short")))

    output = capsys.readouterr().out
    assert FakeThread.started == []
    assert "PARKING_MCP_TOKEN is missing or too short" in output
    assert "Telegram is not configured" in output


def test_graph_command_prints_both_diagrams(capsys):
    cli.show_graphs()

    output = capsys.readouterr().out
    for node in ("screen_input", "escalate", "await_decision", "record_reservation"):
        assert node in output


def test_loadtest_command_writes_its_report(monkeypatch, tmp_path, capsys):
    written: dict[str, str] = {}
    monkeypatch.setattr(cli, "build_knowledge", lambda settings: StaticKnowledge([]))
    monkeypatch.setattr(cli, "build_guardrails", lambda settings: Guardrails(MaskingRedactor()))
    monkeypatch.setattr(cli, "write_report", lambda path, text: written.update({path.name: text}))
    args = cli.parse_args(
        [
            "loadtest",
            "--sessions",
            "2",
            "--concurrency",
            "2",
            "--mcp-requests",
            "3",
            "--mcp-concurrency",
            "1",
            "2",
        ]
    )

    cli.loadtest(Settings(reports_dir=tmp_path), args)

    report = written["load_test_report.md"]
    assert "2 chat sessions at concurrency 2 (stub LLM)" in report
    for section in (
        "Chatbot in interactive dialogue mode",
        "Administrator confirmation",
        "MCP server",
    ):
        assert section in report
    assert report in capsys.readouterr().out


class FailingAssistant:
    def __init__(self) -> None:
        self.calls = 0

    def reply(self, session_id: str, text: str) -> str:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("provider said: key sk-secret was wrong")
        return f"echo: {text}"

    def announcements(self, session_id: str, database=None) -> str:
        return ""


def test_a_failed_turn_gets_an_apology_and_the_conversation_continues(monkeypatch, capsys, caplog):
    assistant = FailingAssistant()
    monkeypatch.setattr(cli, "build_assistant", lambda settings: assistant)
    typed = iter(["first", "second", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))

    cli.chat(UNCONFIGURED)

    output = capsys.readouterr().out
    assert f"Assistant: {cli.APOLOGY}" in output
    assert "Assistant: echo: second" in output
    assert "sk-secret" not in output + caplog.text
    assert "RuntimeError" in caplog.text
