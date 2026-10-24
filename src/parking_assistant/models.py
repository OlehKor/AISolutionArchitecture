from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

from parking_assistant.config import Settings


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX embeddings that need no API key."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "azure":
        return AzureChatOpenAI(azure_deployment=settings.chat_model, temperature=0)
    return ChatOpenAI(model=settings.chat_model, temperature=0)


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embeddings_provider == "azure":
        return AzureOpenAIEmbeddings(azure_deployment=settings.embeddings_model)
    if settings.embeddings_provider == "openai":
        return OpenAIEmbeddings(model=settings.embeddings_model)
    return FastEmbedEmbeddings(settings.embeddings_model)
