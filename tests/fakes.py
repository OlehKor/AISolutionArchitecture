from collections import deque
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from parking_assistant.approval import AdminDeliveryError, Decision, IncomingMessage
from parking_assistant.conversation import Route
from parking_assistant.database import Reservation
from parking_assistant.guardrails import Redaction
from parking_assistant.knowledge import Passage
from parking_assistant.reservation import ReservationDraft, SlotUpdates

NOW = datetime(2026, 10, 24, 10, 0)


def slot_updates(**values: str) -> SlotUpdates:
    fields = {
        "first_name": "",
        "last_name": "",
        "car_number": "",
        "start": "",
        "end": "",
        "confirmation": "none",
    }
    return SlotUpdates(**{**fields, **values})


class ScriptedModel:
    """Conversation model that replays prepared routes and slot updates."""

    def __init__(self, routes: list[Route], updates: list[SlotUpdates] | None = None) -> None:
        self._routes = deque(routes)
        self._updates = deque(updates or [])
        self.answer_text = "scripted answer"
        self.answer_calls: list[dict] = []

    def classify(self, history: list[BaseMessage], message: str, draft: ReservationDraft) -> Route:
        return self._routes.popleft()

    def extract_slots(self, message: str, draft: ReservationDraft, now: datetime) -> SlotUpdates:
        return self._updates.popleft()

    def answer(
        self,
        history: list[BaseMessage],
        question: str,
        passages: list[Passage],
        live_data: str,
        now: datetime,
    ) -> str:
        self.answer_calls.append(
            {"history": history, "question": question, "passages": passages, "live_data": live_data}
        )
        return self.answer_text


class StaticKnowledge:
    def __init__(self, passages: list[Passage]) -> None:
        self._passages = passages
        self.queries: list[str] = []

    def search(self, query: str, k: int = 0) -> list[Passage]:
        self.queries.append(query)
        return self._passages[:k] if k else self._passages


class MaskingRedactor:
    """Redactor stand-in that hides the given secrets."""

    def __init__(self, *secrets: str) -> None:
        self._secrets = secrets

    def redact(self, text: str, entities: tuple[str, ...]) -> Redaction:
        found = tuple(secret for secret in self._secrets if secret in text)
        for secret in found:
            text = text.replace(secret, "[REDACTED]")
        return Redaction(text=text, entities=found)


class FakeVectorStore:
    def __init__(self, hits: list[tuple[Document, float]]) -> None:
        self._hits = hits
        self.calls: list[dict] = []

    def similarity_search_with_score(self, query: str, k: int, expr: str):
        self.calls.append({"query": query, "k": k, "expr": expr})
        return self._hits


class RecordingAdministrator:
    """Gateway stand-in that records escalations, optionally failing like an unreachable admin."""

    def __init__(self, unreachable: bool = False) -> None:
        self.requests: list[Reservation] = []
        self._unreachable = unreachable

    def request_approval(self, reservation: Reservation) -> None:
        if self._unreachable:
            raise AdminDeliveryError("unreachable")
        self.requests.append(reservation)


class RecordingChannel:
    """Admin channel that stores what was sent and replays queued incoming messages."""

    def __init__(self, incoming: list[list[IncomingMessage]] | None = None) -> None:
        self.sent: list[str] = []
        self._incoming = deque(incoming or [])
        self._next_id = 100

    def send(self, text: str) -> int:
        self.sent.append(text)
        self._next_id += 1
        return self._next_id

    def poll(self) -> list[IncomingMessage]:
        if not self._incoming:
            raise AdminDeliveryError("no more messages")
        batch = self._incoming.popleft()
        if isinstance(batch, Exception):
            raise batch
        return batch


class ScriptedAdministratorModel:
    def __init__(self, *decisions: Decision, note: str = "Looks routine.") -> None:
        self._decisions = deque(decisions)
        self._note = note
        self.interpreted: list[tuple[str, str]] = []

    def write_note(self, reservation: Reservation, spaces_left: int, now: datetime) -> str:
        return self._note

    def interpret_reply(self, reservation: Reservation, reply: str) -> Decision:
        self.interpreted.append((reservation.reference, reply))
        return self._decisions.popleft()
