from datetime import datetime

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from parking_assistant.conversation import (
    LlmConversationModel,
    Route,
    describe_collected,
    describe_pending_question,
    describe_reservation_state,
    format_context,
)
from parking_assistant.knowledge import Passage
from parking_assistant.reservation import ReservationDraft, SlotUpdates


def test_reservation_state_description():
    assert describe_reservation_state(ReservationDraft()) == "none"
    assert "still missing: first_name" in describe_reservation_state(ReservationDraft(active=True))
    assert "still missing: last_name" in describe_reservation_state(
        ReservationDraft(first_name="Anna", car_number="ABC123", start="s", end="e", active=True)
    )
    assert "waiting for the customer to confirm" in describe_reservation_state(
        ReservationDraft("Anna", "K", "ABC123", "s", "e", awaiting_confirmation=True, active=True)
    )


def test_collected_details_description():
    assert describe_collected(ReservationDraft()) == "nothing yet"
    assert describe_collected(ReservationDraft(first_name="Anna", car_number="ABC123")) == (
        "first_name=Anna, car_number=ABC123"
    )


def test_pending_question_follows_the_missing_field_order():
    assert describe_pending_question(ReservationDraft()) == "What is your first name?"
    assert "confirm" not in describe_pending_question(ReservationDraft(first_name="Anna"))
    assert "submit" in describe_pending_question(
        ReservationDraft("Anna", "K", "ABC123", "s", "e", awaiting_confirmation=True)
    )


def test_context_labels_every_excerpt_with_its_chunk_id():
    passages = [
        Passage("faq#a", "faq", "a", "Text A", 0.1),
        Passage("faq#b", "faq", "b", "Text B", 0.2),
    ]

    assert format_context(passages) == "[faq#a]\nText A\n\n[faq#b]\nText B"
    assert format_context([]) == "(no excerpts found)"


class StubChatModel(RunnableLambda):
    """Chat model stand-in that answers every prompt with a fixed reply."""

    def __init__(self) -> None:
        super().__init__(lambda prompt: AIMessage("stub reply"))

    def with_structured_output(self, schema):
        values = {
            Route: {"intent": "reserve", "needs_live_data": False},
            SlotUpdates: {
                "first_name": "Anna",
                "last_name": "",
                "car_number": "",
                "start": "",
                "end": "",
                "confirmation": "none",
            },
        }
        return RunnableLambda(lambda prompt: schema.model_validate(values[schema]))


def test_llm_model_formats_all_prompts_and_parses_structured_output():
    model = LlmConversationModel(StubChatModel())
    now = datetime(2026, 10, 24, 10, 0)
    draft = ReservationDraft()

    assert model.classify([], "I want to book", draft).intent == "reserve"
    assert model.extract_slots("I am Anna", draft, now).first_name == "Anna"
    answer = model.answer([], "Where are you?", [], "", now)
    assert type(answer) is str
    assert answer == "stub reply"


def test_known_dates_without_a_time_are_reported_to_the_model_and_asked_for():
    draft = ReservationDraft(
        active=True,
        first_name="Anna",
        last_name="Kowalska",
        car_number="ABC123",
        start_date="2026-11-01",
    )

    assert "arrival date=2026-11-01 (time still needed)" in describe_collected(draft)
    assert describe_pending_question(draft) == "What time will you arrive on Sun 01 Nov 2026?"
