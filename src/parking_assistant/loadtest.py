import itertools
import secrets
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from parking_assistant.administrator import AdminDesk
from parking_assistant.archive import McpReservationArchive
from parking_assistant.assistant import Dependencies, ParkingAssistant
from parking_assistant.composition import new_reservation_reference
from parking_assistant.conversation import ConversationModel, Route
from parking_assistant.database import CONFIRMED, ParkingDatabase, Reservation
from parking_assistant.evaluation import LatencyStats, summarize_latencies
from parking_assistant.guardrails import Guardrails
from parking_assistant.knowledge import KnowledgeSearch
from parking_assistant.mcp_server import background_server
from parking_assistant.reservation import ReservationDraft, SlotUpdates
from parking_assistant.workflow import ReservationWorkflow, open_saver

INFO_LABEL = "Chat: information question (guard rails + RAG + live data)"
STEP_LABEL = "Chat: reservation step (slot filling)"
BOOKING_LABEL = "Chat: confirmation (booking + workflow start + request to administrator)"
DECISION_LABEL = "Administrator decision (resume workflow + database + MCP write)"
MCP_LABEL = "MCP tool call (authenticated HTTP + validated write)"
REJECT_EVERY = 5


@dataclass(frozen=True)
class Measurement:
    label: str
    latency: LatencyStats


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    jobs: int
    errors: tuple[str, ...]
    seconds: float
    measurements: tuple[Measurement, ...]
    checks: tuple[str, ...]

    @property
    def throughput(self) -> float:
        return self.jobs / self.seconds if self.seconds else 0.0


@dataclass(frozen=True)
class Outcome:
    samples: tuple[tuple[str, float], ...]
    errors: tuple[str, ...]
    seconds: float


class StubConversationModel:
    """Rule-based stand-in for the LLM so the load test measures the system, not the API."""

    def classify(self, history, message: str, draft: ReservationDraft) -> Route:
        if draft.active or message.startswith(("book", "name:", "plate:", "arrive:", "leave:")):
            return Route(intent="reserve", needs_live_data=False)
        return Route(intent="info", needs_live_data=True)

    def extract_slots(self, message: str, draft: ReservationDraft, now: datetime) -> SlotUpdates:
        fields = {"first_name": "", "last_name": "", "car_number": "", "start": "", "end": ""}
        confirmation = "yes" if message == "confirm" else "none"
        key, _, value = message.partition(": ")
        if key == "name":
            fields["first_name"], fields["last_name"] = value.split()
        elif key in ("plate", "arrive", "leave"):
            fields[{"plate": "car_number", "arrive": "start", "leave": "end"}[key]] = value
        return SlotUpdates(**fields, confirmation=confirmation)

    def answer(self, history, question: str, passages, live_data: str, now: datetime) -> str:
        return f"Stub answer using {len(passages)} excerpts."


class LoadChannel:
    """Administrator channel that accepts everything, like a Telegram bot that always answers."""

    def __init__(self) -> None:
        self._ids = itertools.count(1)

    def send(self, text: str, decision_for: str = "") -> int:
        return next(self._ids)


class StubAdministratorModel:
    def write_note(self, reservation: Reservation, spaces_left: int, now: datetime) -> str:
        return "Load test request."


@dataclass
class Stack:
    """One worker thread's components; SQLite connections cannot be shared between threads."""

    database: ParkingDatabase
    workflow: ReservationWorkflow
    assistant: ParkingAssistant


@dataclass
class LoadEnvironment:
    directory: Path
    mcp_url: str
    token: str
    schema_sql: Path
    seed_sql: Path
    knowledge: KnowledgeSearch
    guardrails: Guardrails
    model_factory: Callable[[], ConversationModel]
    main_database: ParkingDatabase
    _local: threading.local = field(default_factory=threading.local)
    _stacks: list[Stack] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def stack(self) -> Stack:
        if not hasattr(self._local, "stack"):
            self._local.stack = self._build_stack()
            with self._lock:
                self._stacks.append(self._local.stack)
        return self._local.stack

    def close(self) -> None:
        # Worker connections belong to their threads and are dropped with the temporary directory.
        self.main_database.close()

    def _build_stack(self) -> Stack:
        database = ParkingDatabase.open(
            self.directory / "parking.sqlite", self.schema_sql, self.seed_sql
        )
        desk = AdminDesk(StubAdministratorModel(), LoadChannel(), database, datetime.now)
        workflow = ReservationWorkflow(
            database,
            desk,
            McpReservationArchive(self.mcp_url, self.token, attempts=1),
            datetime.now,
            open_saver(self.directory / "workflow.sqlite"),
        )
        assistant = ParkingAssistant(
            Dependencies(
                model=self.model_factory(),
                knowledge=self.knowledge,
                database=database,
                guardrails=self.guardrails,
                workflow=workflow,
                clock=datetime.now,
                new_reference=new_reservation_reference,
                history_messages=6,
            )
        )
        return Stack(database, workflow, assistant)


@contextmanager
def load_environment(
    schema_sql: Path,
    seed_sql: Path,
    knowledge: KnowledgeSearch,
    guardrails: Guardrails,
    model_factory: Callable[[], ConversationModel] = StubConversationModel,
) -> Iterator[LoadEnvironment]:
    """Isolated temporary data and a real MCP server; nothing touches the real database."""
    with tempfile.TemporaryDirectory(prefix="parking-load-") as name:
        directory = Path(name)
        token = secrets.token_urlsafe(32)
        with background_server(token, directory / "confirmed.txt") as url:
            environment = LoadEnvironment(
                directory=directory,
                mcp_url=url,
                token=token,
                schema_sql=schema_sql,
                seed_sql=seed_sql,
                knowledge=knowledge,
                guardrails=guardrails,
                model_factory=model_factory,
                main_database=ParkingDatabase.open(
                    directory / "parking.sqlite", schema_sql, seed_sql
                ),
            )
            try:
                yield environment
            finally:
                environment.close()


def run_jobs(jobs: Sequence[Callable[[], list[tuple[str, float]]]], concurrency: int) -> Outcome:
    """Run jobs concurrently; failures are recorded instead of stopping the run."""
    samples: list[tuple[str, float]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def execute(job: Callable[[], list[tuple[str, float]]]) -> None:
        try:
            result = job()
        except Exception as error:
            with lock:
                errors.append(f"{type(error).__name__}: {error}"[:200])
            return
        with lock:
            samples.extend(result)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(execute, jobs))
    return Outcome(tuple(samples), tuple(errors), time.perf_counter() - started)


def measurements(outcome: Outcome) -> tuple[Measurement, ...]:
    labels = list(dict.fromkeys(label for label, _ in outcome.samples))
    return tuple(
        Measurement(
            label, summarize_latencies([ms for name, ms in outcome.samples if name == label])
        )
        for label in labels
    )


def _timed(label: str, call: Callable[[], object]) -> tuple[str, float]:
    started = time.perf_counter()
    call()
    return label, (time.perf_counter() - started) * 1000


def stub_script(index: int, day: date) -> list[tuple[str, str]]:
    return [
        (INFO_LABEL, "What are the opening hours and the prices?"),
        (INFO_LABEL, "Where is the garage located?"),
        (STEP_LABEL, "book"),
        (STEP_LABEL, "name: Anna Load"),
        (STEP_LABEL, f"plate: LD{index:05d}"),
        (STEP_LABEL, f"arrive: {day}T09:00"),
        (STEP_LABEL, f"leave: {day}T17:00"),
        (BOOKING_LABEL, "confirm"),
    ]


def natural_script(index: int, day: date) -> list[tuple[str, str]]:
    return [
        (INFO_LABEL, "What are the opening hours and the prices?"),
        (INFO_LABEL, "Where is the garage located?"),
        (STEP_LABEL, "I would like to book a parking space"),
        (STEP_LABEL, "Anna Load"),
        (STEP_LABEL, f"LD{index:05d}"),
        (STEP_LABEL, f"arriving on {day} at 09:00"),
        (STEP_LABEL, f"leaving on {day} at 17:00"),
        (BOOKING_LABEL, "yes"),
    ]


def chat_session(
    environment: LoadEnvironment, index: int, script: Callable[[int, date], list[tuple[str, str]]]
) -> list[tuple[str, float]]:
    assistant = environment.stack().assistant
    session = f"load-{index}"
    day = date.today() + timedelta(days=1 + index % 60)
    samples = []
    for label, text in script(index, day):
        started = time.perf_counter()
        reply = assistant.reply(session, text)
        samples.append((label, (time.perf_counter() - started) * 1000))
        if not reply.strip():
            raise AssertionError(f"empty reply to {text!r}")
    if "reservation request PK-" not in reply:
        raise AssertionError(f"the booking was not submitted: {reply[:80]!r}")
    return samples


def run_chat_load(
    environment: LoadEnvironment, sessions: int, concurrency: int, natural: bool = False
) -> ScenarioResult:
    script = natural_script if natural else stub_script
    jobs = [
        lambda index=index: chat_session(environment, index, script) for index in range(sessions)
    ]
    outcome = run_jobs(jobs, concurrency)
    pending = len(environment.main_database.pending_reservations())
    return ScenarioResult(
        name="Chatbot in interactive dialogue mode",
        jobs=sessions,
        errors=outcome.errors,
        seconds=outcome.seconds,
        measurements=measurements(outcome),
        checks=(f"{pending} reservation requests are waiting for the administrator",),
    )


def run_administrator_load(environment: LoadEnvironment, concurrency: int) -> ScenarioResult:
    pending = environment.main_database.pending_reservations()

    def decide(position: int, reference: str) -> list[tuple[str, float]]:
        verdict = "reject" if position % REJECT_EVERY == REJECT_EVERY - 1 else "approve"
        workflow = environment.stack().workflow
        return [_timed(DECISION_LABEL, lambda: workflow.decide(reference, verdict, "load test"))]

    jobs = [lambda p=p, r=r.reference: decide(p, r) for p, r in enumerate(pending)]
    outcome = run_jobs(jobs, concurrency)
    database = environment.main_database
    confirmed = sum(
        1 for r in database.reservations([r.reference for r in pending]) if r.status == CONFIRMED
    )
    written = _count_lines(environment.directory / "confirmed.txt")
    return ScenarioResult(
        name="Administrator confirmation (workflow resume)",
        jobs=len(pending),
        errors=outcome.errors,
        seconds=outcome.seconds,
        measurements=measurements(outcome),
        checks=(
            f"{len(database.pending_reservations())} requests still pending",
            f"{confirmed} confirmed, {len(database.unarchived_confirmations())} not yet archived",
            f"{written} lines in the reservations file (expected {confirmed})",
        ),
    )


def _count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def synthetic_reservation(index: int) -> Reservation:
    day = date.today() + timedelta(days=1 + index % 60)
    return Reservation(
        reference=f"PK-{index:06X}",
        first_name="Anna",
        last_name="Load",
        car_number=f"MC{index:05d}",
        start_time=f"{day}T09:00",
        end_time=f"{day}T17:00",
        status=CONFIRMED,
        decided_at=f"{day}T08:00",
    )


def run_mcp_load(requests_per_level: int, concurrencies: Sequence[int]) -> ScenarioResult:
    """Write distinct reservations at several client counts, then replay some to test idempotency."""
    with tempfile.TemporaryDirectory(prefix="parking-mcp-load-") as name:
        path = Path(name) / "confirmed.txt"
        token = secrets.token_urlsafe(32)
        with background_server(token, path) as url:
            archive = McpReservationArchive(url, token, attempts=1)
            samples: list[Measurement] = []
            errors: list[str] = []
            notes: list[str] = []
            seconds = 0.0
            for position, level in enumerate(concurrencies):
                first = position * requests_per_level
                label = f"{MCP_LABEL}, {level} parallel clients"
                jobs = [
                    lambda i=i, label=label: [
                        _timed(label, lambda: archive.archive(synthetic_reservation(i)))
                    ]
                    for i in range(first, first + requests_per_level)
                ]
                outcome = run_jobs(jobs, level)
                samples.extend(measurements(outcome))
                errors.extend(outcome.errors)
                seconds += outcome.seconds
                notes.append(
                    f"{level} parallel clients: {requests_per_level / outcome.seconds:.1f} calls/s"
                )
            replay_count = min(20, requests_per_level)
            replayed = run_jobs(
                [
                    lambda i=i: [
                        _timed("replay", lambda: archive.archive(synthetic_reservation(i)))
                    ]
                    for i in range(replay_count)
                ],
                max(concurrencies),
            )
            lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    total = requests_per_level * len(concurrencies)
    return ScenarioResult(
        name="MCP server recording and storage",
        jobs=total,
        errors=(*errors, *replayed.errors),
        seconds=seconds,
        measurements=tuple(samples),
        checks=(
            *notes,
            f"{len(lines)} lines stored for {total} distinct reservations",
            f"{len(lines) - len(set(lines))} duplicate lines after replaying {replay_count} requests",
        ),
    )


def render_load_report(
    results: Sequence[ScenarioResult], parameters: str, generated: datetime
) -> str:
    lines = [
        "# Load test report",
        "",
        f"Generated: {generated:%Y-%m-%d %H:%M}",
        "",
        f"Parameters: {parameters}. Telegram is replaced by a stand-in that always answers, and the "
        "chat language model by a rule-based stub unless the parameters say 'real LLM'; LangGraph, "
        "SQLite, the vector store, the guard rails and the MCP server are real.",
        "",
    ]
    for result in results:
        lines += [
            f"## {result.name}",
            "",
            f"- Jobs: {result.jobs}, errors: {len(result.errors)}, "
            f"duration: {result.seconds:.2f} s, throughput: {result.throughput:.1f} jobs/s",
            *[f"- {check}" for check in result.checks],
            *[f"- Error sample: {error}" for error in result.errors[:3]],
            "",
            "| Operation | Requests | Mean ms | p50 ms | p95 ms | Max ms |",
            "|---|---|---|---|---|---|",
            *[
                f"| {m.label} | {m.latency.count} | {m.latency.mean_ms:.1f} | {m.latency.p50_ms:.1f} "
                f"| {m.latency.p95_ms:.1f} | {m.latency.max_ms:.1f} |"
                for m in result.measurements
            ],
            "",
        ]
    return "\n".join(lines)
