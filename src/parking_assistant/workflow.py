import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from parking_assistant.approval import AdminDeliveryError
from parking_assistant.archive import ArchiveError, ReservationArchive
from parking_assistant.database import (
    CONFIRMED,
    PENDING,
    REJECTED,
    ParkingDatabase,
    Reservation,
)

PHASE_START = "start"
PHASE_RETRY = "retry"
CHECKPOINT_TIMEOUT_SECONDS = 30


class ApprovalDesk(Protocol):
    """The human-in-the-loop agent's outgoing side: puts a request in front of the administrator."""

    def request_approval(self, reservation: Reservation) -> None: ...


class WorkflowState(TypedDict, total=False):
    reference: str
    phase: str
    delivered: bool
    verdict: str
    reason: str
    applied: bool
    status: str
    archived: bool


@dataclass(frozen=True)
class DecisionResult:
    applied: bool
    status: str
    archived: bool


def open_saver(path: Path) -> SqliteSaver:
    """Checkpoint store shared by every process that touches reservation workflows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False, timeout=CHECKPOINT_TIMEOUT_SECONDS)
    connection.execute("PRAGMA journal_mode = WAL")
    return SqliteSaver(connection)


def _entry_node(state: WorkflowState) -> str:
    return "record_reservation" if state.get("phase") == PHASE_RETRY else "request_approval"


def _after_decision(state: WorkflowState) -> str:
    recordable = state["applied"] and state["status"] == CONFIRMED
    return "record_reservation" if recordable else "close_request"


def build_workflow_graph(
    database: ParkingDatabase,
    desk: ApprovalDesk,
    archive: ReservationArchive,
    clock: Callable[[], datetime],
) -> StateGraph:
    def request_approval(state: WorkflowState) -> dict:
        reference = state["reference"]
        if database.request_message_id_for(reference):
            return {"delivered": True}
        try:
            desk.request_approval(database.get_reservation(reference))
        except AdminDeliveryError:
            return {"delivered": False}
        return {"delivered": True}

    def await_decision(state: WorkflowState) -> dict:
        answer = interrupt({"reference": state["reference"]})
        return {"verdict": answer["verdict"], "reason": answer["reason"]}

    def apply_decision(state: WorkflowState) -> dict:
        status = CONFIRMED if state["verdict"] == "approve" else REJECTED
        applied = database.decide(state["reference"], status, state["reason"], clock())
        return {"applied": applied, "status": database.get_reservation(state["reference"]).status}

    def record_reservation(state: WorkflowState) -> dict:
        reference = state["reference"]
        try:
            archive.archive(database.get_reservation(reference))
        except ArchiveError:
            return {"archived": False}
        database.mark_archived(reference)
        return {"archived": True}

    def close_request(state: WorkflowState) -> dict:
        return {"status": database.get_reservation(state["reference"]).status}

    graph = StateGraph(WorkflowState)
    graph.add_node("request_approval", request_approval)
    graph.add_node("await_decision", await_decision)
    graph.add_node("apply_decision", apply_decision)
    graph.add_node("record_reservation", record_reservation)
    graph.add_node("close_request", close_request)

    graph.add_conditional_edges(
        START,
        _entry_node,
        {"request_approval": "request_approval", "record_reservation": "record_reservation"},
    )
    graph.add_edge("request_approval", "await_decision")
    graph.add_edge("await_decision", "apply_decision")
    graph.add_conditional_edges(
        "apply_decision",
        _after_decision,
        {"record_reservation": "record_reservation", "close_request": "close_request"},
    )
    graph.add_edge("record_reservation", "close_request")
    graph.add_edge("close_request", END)
    return graph


class ReservationWorkflow:
    """One durable LangGraph run per reservation: request, wait for the human, record."""

    def __init__(
        self,
        database: ParkingDatabase,
        desk: ApprovalDesk,
        archive: ReservationArchive,
        clock: Callable[[], datetime],
        saver: BaseCheckpointSaver,
    ) -> None:
        self._database = database
        self._graph = build_workflow_graph(database, desk, archive, clock).compile(
            checkpointer=saver
        )

    def start(self, reference: str) -> bool:
        """Ask the administrator and pause; returns False when the request could not be delivered."""
        config = _config(reference)
        if self._graph.get_state(config).next:
            return True
        result = self._graph.invoke({"reference": reference, "phase": PHASE_START}, config)
        return result["delivered"]

    def decide(self, reference: str, verdict: str, reason: str) -> DecisionResult:
        """Resume the paused run with the administrator's decision."""
        config = _config(reference)
        if not self._graph.get_state(config).next:
            current = self._database.get_reservation(reference)
            if current.status != PENDING:
                return DecisionResult(applied=False, status=current.status, archived=True)
            self.start(reference)
        result = self._graph.invoke(Command(resume={"verdict": verdict, "reason": reason}), config)
        return DecisionResult(
            applied=result["applied"],
            status=result["status"],
            archived=result.get("archived", True),
        )

    def retry_pending(self) -> int:
        """Run the recording step again for confirmed reservations that could not be stored."""
        stored = 0
        for reservation in self._database.unarchived_confirmations():
            phase = {"reference": reservation.reference, "phase": PHASE_RETRY}
            result = self._graph.invoke(phase, _config(reservation.reference))
            stored += bool(result.get("archived"))
        return stored

    def mermaid(self) -> str:
        return self._graph.get_graph().draw_mermaid()


def _config(reference: str) -> dict:
    return {"configurable": {"thread_id": f"reservation-{reference}"}}


def workflow_diagram() -> str:
    """Mermaid source of the workflow graph; nothing runs, so no dependencies are needed."""
    graph = build_workflow_graph(
        cast(ParkingDatabase, None),
        cast(ApprovalDesk, None),
        cast(ReservationArchive, None),
        datetime.now,
    )
    return graph.compile().get_graph().draw_mermaid()
