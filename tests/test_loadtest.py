from datetime import date, datetime

import pytest

from conftest import SCHEMA_SQL, SEED_SQL
from fakes import MaskingRedactor, StaticKnowledge
from parking_assistant.conversation import Route
from parking_assistant.evaluation import LatencyStats
from parking_assistant.guardrails import Guardrails
from parking_assistant.loadtest import (
    Measurement,
    Outcome,
    ScenarioResult,
    StubConversationModel,
    load_environment,
    measurements,
    render_load_report,
    run_administrator_load,
    run_chat_load,
    run_jobs,
    run_mcp_load,
    stub_script,
    synthetic_reservation,
)
from parking_assistant.reservation import ReservationDraft

NOW = datetime(2026, 10, 25, 10, 0)


def test_the_stub_model_follows_the_scripted_dialogue():
    model = StubConversationModel()
    idle, active = ReservationDraft(), ReservationDraft(active=True)

    assert model.classify([], "What are the opening hours?", idle) == Route(
        intent="info", needs_live_data=True
    )
    assert model.classify([], "book", idle).intent == "reserve"
    assert model.classify([], "name: Anna Load", active).intent == "reserve"
    assert model.extract_slots("name: Anna Load", active, NOW).last_name == "Load"
    assert model.extract_slots("plate: LD00001", active, NOW).car_number == "LD00001"
    assert model.extract_slots("arrive: 2026-11-01T09:00", active, NOW).start == "2026-11-01T09:00"
    assert model.extract_slots("confirm", active, NOW).confirmation == "yes"
    assert "2 excerpts" in model.answer([], "q", [1, 2], "", NOW)


def test_the_stub_script_ends_with_the_confirmation():
    script = stub_script(7, date(2026, 11, 1))

    assert script[-1] == (
        "Chat: confirmation (booking + workflow start + request to administrator)",
        "confirm",
    )
    assert any("LD00007" in text for _, text in script)


def test_failed_jobs_are_recorded_without_stopping_the_run():
    def good() -> list[tuple[str, float]]:
        return [("op", 1.0)]

    def bad() -> list[tuple[str, float]]:
        raise ValueError("boom")

    outcome = run_jobs([good, bad, good], concurrency=2)

    assert len(outcome.samples) == 2
    assert outcome.errors == ("ValueError: boom",)


def test_measurements_are_grouped_by_operation_label():
    outcome = Outcome((("a", 10.0), ("b", 5.0), ("a", 30.0)), (), 1.0)

    grouped = {m.label: m.latency for m in measurements(outcome)}

    assert grouped["a"].count == 2 and grouped["a"].max_ms == 30.0
    assert grouped["b"].count == 1


def test_synthetic_reservations_are_valid_and_distinct():
    first, second = synthetic_reservation(1), synthetic_reservation(2)

    assert first.car_number != second.car_number
    assert first.reference != second.reference
    assert first.start_time < first.end_time


def test_chat_and_approval_load_complete_without_errors_and_keep_the_file_consistent():
    guardrails = Guardrails(MaskingRedactor())
    with load_environment(SCHEMA_SQL, SEED_SQL, StaticKnowledge([]), guardrails) as environment:
        chat = run_chat_load(environment, sessions=6, concurrency=3)
        approvals = run_administrator_load(environment, concurrency=3)

    assert chat.errors == () and chat.jobs == 6
    assert "6 reservation requests are waiting" in chat.checks[0]
    assert {m.label.split(":")[0] for m in chat.measurements} == {"Chat"}
    assert approvals.errors == () and approvals.jobs == 6
    assert approvals.checks[0] == "0 requests still pending"
    assert "5 confirmed, 0 not yet archived" in approvals.checks[1]
    assert "5 lines in the reservations file (expected 5)" in approvals.checks[2]


def test_mcp_load_stores_every_reservation_once_even_when_replayed():
    result = run_mcp_load(requests_per_level=6, concurrencies=[1, 3])

    assert result.errors == ()
    assert result.jobs == 12
    assert "12 lines stored for 12 distinct reservations" in result.checks
    assert "0 duplicate lines after replaying 6 requests" in result.checks
    assert len(result.measurements) == 2


def test_the_report_lists_every_scenario_with_its_latency_table():
    latency = LatencyStats(count=3, mean_ms=10.0, p50_ms=9.0, p95_ms=12.0, max_ms=13.0)
    result = ScenarioResult(
        name="Scenario one",
        jobs=3,
        errors=("ValueError: boom",),
        seconds=1.5,
        measurements=(Measurement("Operation", latency),),
        checks=("everything consistent",),
    )

    report = render_load_report([result], "3 jobs", NOW)

    assert "## Scenario one" in report
    assert "Jobs: 3, errors: 1, duration: 1.50 s, throughput: 2.0 jobs/s" in report
    assert "Error sample: ValueError: boom" in report
    assert "| Operation | 3 | 10.0 | 9.0 | 12.0 | 13.0 |" in report
    assert "Parameters: 3 jobs." in report


@pytest.mark.parametrize("seconds", [0.0])
def test_throughput_is_zero_for_an_empty_run(seconds):
    empty = ScenarioResult("x", 0, (), seconds, (), ())

    assert empty.throughput == 0.0
