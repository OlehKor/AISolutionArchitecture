import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from parking_assistant.config import Settings

PUBLIC = "public"
PUBLIC_ONLY_FILTER = f'visibility == "{PUBLIC}"'


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    source: str
    section: str
    text: str
    distance: float


class KnowledgeSearch(Protocol):
    def search(self, query: str, k: int = 0) -> list[Passage]: ...


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def split_markdown(
    text: str, source: str, visibility: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=True
    ).split_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        chunk
        for section in sections
        for chunk in splitter.split_documents([_with_context(section, source, visibility)])
    ]


def _with_context(section: Document, source: str, visibility: str) -> Document:
    title = section.metadata["title"]
    heading = section.metadata.get("section", title)
    return Document(
        page_content=f"{title} / {heading}\n{section.page_content}",
        metadata={
            "chunk_id": f"{source}#{slugify(heading)}",
            "source": source,
            "section": heading,
            "visibility": visibility,
        },
    )


def load_documents(static_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Each sub-directory of `static_dir` names the visibility of the files inside it."""
    return [
        chunk
        for path in sorted(static_dir.glob("*/*.md"))
        for chunk in split_markdown(
            path.read_text(encoding="utf-8"), path.stem, path.parent.name, chunk_size, chunk_overlap
        )
    ]


def open_vector_store(settings: Settings, embeddings: Embeddings, drop_old: bool = False) -> Milvus:
    Path(settings.milvus_uri).parent.mkdir(parents=True, exist_ok=True)
    return Milvus(
        embedding_function=embeddings,
        collection_name=settings.milvus_collection,
        connection_args={"uri": settings.milvus_uri},
        auto_id=True,
        enable_dynamic_field=True,
        drop_old=drop_old,
    )


def ingest_static_data(settings: Settings, embeddings: Embeddings) -> int:
    documents = load_documents(
        settings.static_data_dir, settings.chunk_size, settings.chunk_overlap
    )
    store = open_vector_store(settings, embeddings, drop_old=True)
    store.add_documents(documents)
    return len(documents)


def is_ingested(store: Milvus, settings: Settings) -> bool:
    return store.client.has_collection(settings.milvus_collection)


class VectorKnowledgeBase:
    """Semantic search over the static knowledge, restricted to public chunks."""

    def __init__(self, store: Milvus, default_k: int) -> None:
        self._store = store
        self._default_k = default_k

    def search(self, query: str, k: int = 0) -> list[Passage]:
        hits = self._store.similarity_search_with_score(
            query, k=k or self._default_k, expr=PUBLIC_ONLY_FILTER
        )
        return [
            Passage(
                chunk_id=doc.metadata["chunk_id"],
                source=doc.metadata["source"],
                section=doc.metadata["section"],
                text=doc.page_content,
                distance=float(distance),
            )
            for doc, distance in hits
        ]
