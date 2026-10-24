from langchain_openai import AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings

from parking_assistant.config import Settings
from parking_assistant.models import FastEmbedEmbeddings, build_chat_model, build_embeddings


def test_openai_provider_builds_deterministic_chat_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = build_chat_model(Settings(llm_provider="openai", chat_model="gpt-4o-mini"))

    assert isinstance(model, ChatOpenAI)
    assert model.temperature == 0


def test_azure_provider_builds_azure_chat_model(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-10-21")

    model = build_chat_model(Settings(llm_provider="azure", chat_model="my-deployment"))

    assert isinstance(model, AzureChatOpenAI)


def test_openai_embeddings_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    embeddings = build_embeddings(
        Settings(embeddings_provider="openai", embeddings_model="text-embedding-3-small")
    )

    assert isinstance(embeddings, OpenAIEmbeddings)


def test_default_embeddings_run_locally_without_api_key():
    embeddings = build_embeddings(Settings())

    assert isinstance(embeddings, FastEmbedEmbeddings)
    assert len(embeddings.embed_query("parking")) == 384
