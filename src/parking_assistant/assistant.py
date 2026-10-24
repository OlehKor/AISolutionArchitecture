from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from parking_assistant.conversation import ConversationModel
from parking_assistant.database import NoAvailabilityError, ParkingDatabase
from parking_assistant.guardrails import REFUSALS, Guardrails
from parking_assistant.knowledge import KnowledgeSearch, Passage
from parking_assistant.live_data import format_live_data
from parking_assistant.reservation import (
    Outcome,
    ReservationDraft,
    Step,
    advance,
    is_started,
    render_booked,
    render_cancelled,
    render_pending_prompt,
    render_question,
    render_revision_request,
    render_summary,
    render_unavailable,
)

BLOCKED = "blocked"
EMPTY_DRAFT = ReservationDraft()
STATE_TYPES = (
    ("parking_assistant.reservation", "ReservationDraft"),
    ("parking_assistant.knowledge", "Passage"),
)


class ChatState(TypedDict, total=False):
    user_text: str
    messages: Annotated[list[BaseMessage], add_messages]
    draft: ReservationDraft
    intent: str
    needs_live_data: bool
    passages: list[Passage]
    live_data: str
    reply: str


@dataclass(frozen=True)
class Dependencies:
    model: ConversationModel
    knowledge: KnowledgeSearch
    database: ParkingDatabase
    guardrails: Guardrails
    clock: Callable[[], datetime]
    new_reference: Callable[[], str]
    history_messages: int


def _latest_message(state: ChatState) -> str:
    return str(state["messages"][-1].content)


def _history(state: ChatState, limit: int) -> list[BaseMessage]:
    return state["messages"][:-1][-limit:] if limit else []


def _draft(state: ChatState) -> ReservationDraft:
    return state.get("draft", EMPTY_DRAFT)


def _reply_for(outcome: Outcome) -> str:
    if outcome.step is Step.ASK:
        return render_question(outcome.draft, outcome.problems)
    if outcome.step is Step.CONFIRM:
        return render_summary(outcome.draft, outcome.problems)
    return render_revision_request()


def build_graph(deps: Dependencies) -> StateGraph:
    def screen_input(state: ChatState) -> dict:
        verdict = deps.guardrails.check_input(state["user_text"])
        if not verdict.allowed:
            return {"intent": BLOCKED, "reply": REFUSALS[verdict.reason]}
        return {"messages": [HumanMessage(verdict.text)], "intent": "", "reply": ""}

    def classify(state: ChatState) -> dict:
        route = deps.model.classify(
            _history(state, deps.history_messages), _latest_message(state), _draft(state)
        )
        return {"intent": route.intent, "needs_live_data": route.needs_live_data}

    def retrieve(state: ChatState) -> dict:
        passages = deps.knowledge.search(_latest_message(state))
        live_data = (
            format_live_data(deps.database.working_hours(), deps.database.zones(), deps.clock())
            if state["needs_live_data"]
            else ""
        )
        return {"passages": passages, "live_data": live_data}

    def answer(state: ChatState) -> dict:
        text = deps.model.answer(
            _history(state, deps.history_messages),
            _latest_message(state),
            state["passages"],
            state["live_data"],
            deps.clock(),
        )
        draft = _draft(state)
        if is_started(draft):
            text = f"{text}\n\n{render_pending_prompt(draft)}"
        return {"reply": text}

    def reserve(state: ChatState) -> dict:
        now = deps.clock()
        draft = _draft(state)
        updates = deps.model.extract_slots(_latest_message(state), draft, now)
        outcome = advance(draft, updates, now)
        if outcome.step is not Step.BOOK:
            return {"draft": outcome.draft, "reply": _reply_for(outcome)}
        try:
            reservation = deps.database.create_reservation(outcome.draft, deps.new_reference(), now)
        except NoAvailabilityError:
            retry = replace(outcome.draft, start="", end="", awaiting_confirmation=False)
            return {"draft": retry, "reply": render_unavailable(outcome.draft)}
        return {"draft": EMPTY_DRAFT, "reply": render_booked(reservation.reference)}

    def cancel(state: ChatState) -> dict:
        return {"draft": EMPTY_DRAFT, "reply": render_cancelled()}

    def finalize(state: ChatState) -> dict:
        safe_reply = deps.guardrails.sanitize_output(state["reply"]).text
        if state["intent"] == BLOCKED:
            return {"reply": safe_reply}
        return {"reply": safe_reply, "messages": [AIMessage(safe_reply)]}

    graph = StateGraph(ChatState)
    graph.add_node("screen_input", screen_input)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("reserve", reserve)
    graph.add_node("cancel", cancel)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "screen_input")
    graph.add_conditional_edges(
        "screen_input",
        lambda state: "finalize" if state["intent"] == BLOCKED else "classify",
        {"finalize": "finalize", "classify": "classify"},
    )
    graph.add_conditional_edges(
        "classify",
        lambda state: state["intent"],
        {"info": "retrieve", "other": "retrieve", "reserve": "reserve", "cancel": "cancel"},
    )
    graph.add_edge("retrieve", "answer")
    for node in ("answer", "reserve", "cancel"):
        graph.add_edge(node, "finalize")
    graph.add_edge("finalize", END)
    return graph


class ParkingAssistant:
    """Chat entry point; each session id keeps its own history and reservation draft."""

    def __init__(self, deps: Dependencies) -> None:
        saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES))
        self._graph = build_graph(deps).compile(checkpointer=saver)

    def reply(self, session_id: str, text: str) -> str:
        result = self._graph.invoke(
            {"user_text": text}, config={"configurable": {"thread_id": session_id}}
        )
        return result["reply"]
