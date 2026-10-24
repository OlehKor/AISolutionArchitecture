import pytest

from parking_assistant import __main__ as cli


class RecordingAssistant:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def reply(self, session_id: str, text: str) -> str:
        self.messages.append(text)
        return f"echo: {text}"


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

    cli.chat(settings=None)

    assert assistant.messages == ["hello"]
    assert "Assistant: echo: hello" in capsys.readouterr().out


def test_chat_loop_ends_on_end_of_input(monkeypatch):
    assistant = RecordingAssistant()
    monkeypatch.setattr(cli, "build_assistant", lambda settings: assistant)

    def end_of_input(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", end_of_input)

    cli.chat(settings=None)

    assert assistant.messages == []
