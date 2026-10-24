import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

from parking_assistant.assistant import ParkingAssistant
from parking_assistant.guardrails import Guardrails, InputVerdict
from parking_assistant.knowledge import KnowledgeSearch

T = TypeVar("T")

DEFAULT_KS = (1, 3, 5)


class RetrievalCase(BaseModel, frozen=True):
    question: str
    relevant: tuple[str, ...]


class RedactionCase(BaseModel, frozen=True):
    text: str
    forbidden: tuple[str, ...]


class GuardrailDataset(BaseModel, frozen=True):
    blocked: tuple[str, ...]
    allowed: tuple[str, ...]
    redaction: tuple[RedactionCase, ...]
    leak_probes: tuple[str, ...]
    secrets: tuple[str, ...]


@dataclass(frozen=True)
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


@dataclass(frozen=True)
class CutoffScore:
    k: int
    recall: float
    precision: float
    hit_rate: float


@dataclass(frozen=True)
class RetrievalReport:
    cutoffs: tuple[CutoffScore, ...]
    mrr: float
    misses: tuple[str, ...]
    latency: LatencyStats


@dataclass(frozen=True)
class GuardrailReport:
    blocked_caught: int
    blocked_total: int
    allowed_passed: int
    allowed_total: int
    redaction_clean: int
    redaction_total: int
    leak_free: int
    leak_total: int
    failures: tuple[str, ...]
    latency: LatencyStats


def load_retrieval_cases(path: Path) -> tuple[RetrievalCase, ...]:
    return tuple(TypeAdapter(list[RetrievalCase]).validate_json(path.read_text(encoding="utf-8")))


def load_guardrail_dataset(path: Path) -> GuardrailDataset:
    return GuardrailDataset.model_validate_json(path.read_text(encoding="utf-8"))


def unique_in_order(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    found = set(unique_in_order(retrieved)[:k]) & set(relevant)
    return len(found) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top = unique_in_order(retrieved)[:k]
    return len(set(top) & set(relevant)) / len(top) if top else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    for rank, chunk_id in enumerate(unique_in_order(retrieved), start=1):
        if chunk_id in relevant:
            return 1 / rank
    return 0.0


def _mean(values: Iterable[float]) -> float:
    return statistics.fmean(list(values) or [0.0])


def score_retrieval(
    cases: Sequence[RetrievalCase],
    retrieved_ids: Sequence[Sequence[str]],
    ks: Sequence[int],
    latency: LatencyStats,
) -> RetrievalReport:
    pairs = list(zip(retrieved_ids, (case.relevant for case in cases), strict=True))
    cutoffs = tuple(
        CutoffScore(
            k=k,
            recall=_mean(recall_at_k(got, want, k) for got, want in pairs),
            precision=_mean(precision_at_k(got, want, k) for got, want in pairs),
            hit_rate=_mean(float(recall_at_k(got, want, k) > 0) for got, want in pairs),
        )
        for k in ks
    )
    misses = tuple(
        case.question
        for case, (got, want) in zip(cases, pairs, strict=True)
        if recall_at_k(got, want, max(ks)) < 1
    )
    return RetrievalReport(
        cutoffs=cutoffs,
        mrr=_mean(reciprocal_rank(got, want) for got, want in pairs),
        misses=misses,
        latency=latency,
    )


def summarize_latencies(samples_ms: Sequence[float]) -> LatencyStats:
    ordered = sorted(samples_ms)
    if not ordered:
        return LatencyStats(0, 0.0, 0.0, 0.0, 0.0)
    p95_index = min(len(ordered) - 1, max(0, round(0.95 * len(ordered)) - 1))
    return LatencyStats(
        count=len(ordered),
        mean_ms=statistics.fmean(ordered),
        p50_ms=statistics.median(ordered),
        p95_ms=ordered[p95_index],
        max_ms=ordered[-1],
    )


def contains_any(text: str, fragments: Iterable[str]) -> tuple[str, ...]:
    return tuple(fragment for fragment in fragments if fragment.lower() in text.lower())


def _timed(call: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - started) * 1000


def run_retrieval(
    cases: Sequence[RetrievalCase], knowledge: KnowledgeSearch, ks: Sequence[int] = DEFAULT_KS
) -> RetrievalReport:
    timed = [_timed(lambda c=case: knowledge.search(c.question, k=max(ks))) for case in cases]
    retrieved_ids = [[p.chunk_id for p in passages] for passages, _ in timed]
    return score_retrieval(cases, retrieved_ids, ks, summarize_latencies([ms for _, ms in timed]))


def run_guardrails(
    dataset: GuardrailDataset, guardrails: Guardrails, knowledge: KnowledgeSearch
) -> GuardrailReport:
    failures: list[str] = []
    latencies: list[float] = []

    def check(text: str) -> InputVerdict:
        verdict, ms = _timed(lambda: guardrails.check_input(text))
        latencies.append(ms)
        return verdict

    blocked_caught = 0
    for text in dataset.blocked:
        if check(text).allowed:
            failures.append(f"not blocked: {text}")
        else:
            blocked_caught += 1

    allowed_passed = 0
    for text in dataset.allowed:
        if check(text).allowed:
            allowed_passed += 1
        else:
            failures.append(f"wrongly blocked: {text}")

    redaction_clean = 0
    for case in dataset.redaction:
        verdict = check(case.text)
        passed_input = verdict.text if verdict.allowed else ""
        masked_output = guardrails.sanitize_output(case.text).text
        leaked = contains_any(passed_input + "\n" + masked_output, case.forbidden)
        if leaked:
            failures.append(f"not redacted {leaked}: {case.text}")
        else:
            redaction_clean += 1

    leak_free = 0
    for probe in dataset.leak_probes:
        retrieved = " ".join(p.text for p in knowledge.search(probe))
        leaked = contains_any(retrieved, dataset.secrets)
        if leaked:
            failures.append(f"retrieval leaked {leaked}: {probe}")
        else:
            leak_free += 1

    return GuardrailReport(
        blocked_caught=blocked_caught,
        blocked_total=len(dataset.blocked),
        allowed_passed=allowed_passed,
        allowed_total=len(dataset.allowed),
        redaction_clean=redaction_clean,
        redaction_total=len(dataset.redaction),
        leak_free=leak_free,
        leak_total=len(dataset.leak_probes),
        failures=tuple(failures),
        latency=summarize_latencies(latencies),
    )


def run_end_to_end(assistant: ParkingAssistant, questions: Sequence[str]) -> LatencyStats:
    samples = [
        _timed(lambda q=question, i=index: assistant.reply(f"eval-{i}", q))[1]
        for index, question in enumerate(questions)
    ]
    return summarize_latencies(samples)


def _latency_row(name: str, stats: LatencyStats) -> str:
    return (
        f"| {name} | {stats.count} | {stats.mean_ms:.1f} | {stats.p50_ms:.1f} "
        f"| {stats.p95_ms:.1f} | {stats.max_ms:.1f} |"
    )


def _ratio(done: int, total: int) -> str:
    return f"{done}/{total} ({done / total:.0%})" if total else "n/a"


def render_report(
    retrieval: RetrievalReport,
    guardrails: GuardrailReport,
    end_to_end: LatencyStats | None,
    generated_at: datetime,
) -> str:
    cutoff_rows = [
        f"| {c.k} | {c.recall:.3f} | {c.precision:.3f} | {c.hit_rate:.3f} |"
        for c in retrieval.cutoffs
    ]
    latency_rows = [
        _latency_row("Retrieval (vector search)", retrieval.latency),
        _latency_row("Guard rails (input check)", guardrails.latency),
        *([_latency_row("End to end (LLM answer)", end_to_end)] if end_to_end else []),
    ]
    lines = [
        "# Evaluation report",
        "",
        f"Generated: {generated_at:%Y-%m-%d %H:%M}",
        "",
        "## Retrieval accuracy",
        "",
        "| K | Recall@K | Precision@K | Hit rate@K |",
        "|---|---|---|---|",
        *cutoff_rows,
        "",
        f"Mean reciprocal rank: {retrieval.mrr:.3f}",
        "",
        "Questions with a relevant chunk missing from the top results:",
        *([f"- {question}" for question in retrieval.misses] or ["- none"]),
        "",
        "## Performance (milliseconds per request)",
        "",
        "| Stage | Requests | Mean | p50 | p95 | Max |",
        "|---|---|---|---|---|---|",
        *latency_rows,
        "",
        "## Data protection",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Attacks and sensitive requests refused | {_ratio(guardrails.blocked_caught, guardrails.blocked_total)} |",
        f"| Legitimate requests passed | {_ratio(guardrails.allowed_passed, guardrails.allowed_total)} |",
        f"| Personal data fully redacted | {_ratio(guardrails.redaction_clean, guardrails.redaction_total)} |",
        f"| Probes with no confidential text retrieved | {_ratio(guardrails.leak_free, guardrails.leak_total)} |",
        "",
        "Failures:",
        *([f"- {failure}" for failure in guardrails.failures] or ["- none"]),
        "",
    ]
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
