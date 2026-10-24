import secrets
from datetime import datetime

from parking_assistant.assistant import Dependencies, ParkingAssistant
from parking_assistant.config import Settings
from parking_assistant.conversation import LlmConversationModel
from parking_assistant.database import ParkingDatabase
from parking_assistant.guardrails import Guardrails, PiiRedactor
from parking_assistant.knowledge import (
    VectorKnowledgeBase,
    ingest_static_data,
    is_ingested,
    open_vector_store,
)
from parking_assistant.models import build_chat_model, build_embeddings


def build_guardrails(settings: Settings) -> Guardrails:
    return Guardrails(PiiRedactor(settings.public_contacts))


def build_database(settings: Settings) -> ParkingDatabase:
    return ParkingDatabase.open(settings.sqlite_path, settings.schema_sql, settings.seed_sql)


def build_knowledge(settings: Settings) -> VectorKnowledgeBase:
    embeddings = build_embeddings(settings)
    store = open_vector_store(settings, embeddings)
    if not is_ingested(store, settings):
        ingest_static_data(settings, embeddings)
        store = open_vector_store(settings, embeddings)
    return VectorKnowledgeBase(store, settings.retrieval_k)


def new_reservation_reference() -> str:
    return f"PK-{secrets.token_hex(3).upper()}"


def build_assistant(settings: Settings) -> ParkingAssistant:
    return ParkingAssistant(
        Dependencies(
            model=LlmConversationModel(build_chat_model(settings)),
            knowledge=build_knowledge(settings),
            database=build_database(settings),
            guardrails=build_guardrails(settings),
            clock=datetime.now,
            new_reference=new_reservation_reference,
            history_messages=settings.history_messages,
        )
    )
