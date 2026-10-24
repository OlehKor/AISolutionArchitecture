import re

from parking_assistant.__main__ import parse_args
from parking_assistant.composition import new_reservation_reference


def test_reservation_references_have_a_readable_format():
    assert re.fullmatch(r"PK-[0-9A-F]{6}", new_reservation_reference())


def test_reservation_references_are_not_repeated():
    assert len({new_reservation_reference() for _ in range(50)}) == 50


def test_cli_parses_chat_ingest_and_evaluate_commands():
    assert parse_args(["chat"]).command == "chat"
    assert parse_args(["ingest"]).command == "ingest"
    evaluate = parse_args(["evaluate", "--end-to-end"])
    assert (evaluate.command, evaluate.end_to_end) == ("evaluate", True)
