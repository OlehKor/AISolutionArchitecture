from langchain_core.documents import Document

from fakes import FakeVectorStore
from parking_assistant.config import get_settings
from parking_assistant.knowledge import (
    PUBLIC_ONLY_FILTER,
    VectorKnowledgeBase,
    load_documents,
    slugify,
    split_markdown,
)

SAMPLE = "# Title\n\n## Section A\nbody a\n\n## Section B\nbody b\n"


def test_slugify_builds_stable_identifiers():
    assert slugify("What if I lose my ticket?") == "what-if-i-lose-my-ticket"


def test_split_markdown_makes_one_chunk_per_section_with_metadata():
    chunks = split_markdown(SAMPLE, "faq", "public", chunk_size=700, chunk_overlap=0)

    assert [c.metadata["chunk_id"] for c in chunks] == ["faq#section-a", "faq#section-b"]
    assert chunks[0].page_content.startswith("Title / Section A")
    assert all(c.metadata["visibility"] == "public" for c in chunks)


def test_split_markdown_splits_long_sections_under_the_same_id():
    text = "# Title\n\n## Long\n" + " ".join(["word"] * 200)
    chunks = split_markdown(text, "faq", "public", chunk_size=100, chunk_overlap=0)

    assert len(chunks) > 1
    assert {c.metadata["chunk_id"] for c in chunks} == {"faq#long"}


def test_visibility_comes_from_the_directory_name(tmp_path):
    (tmp_path / "public").mkdir()
    (tmp_path / "internal").mkdir()
    (tmp_path / "public" / "a.md").write_text(SAMPLE, encoding="utf-8")
    (tmp_path / "internal" / "b.md").write_text(SAMPLE, encoding="utf-8")

    chunks = load_documents(tmp_path, 700, 0)

    assert {(c.metadata["source"], c.metadata["visibility"]) for c in chunks} == {
        ("a", "public"),
        ("b", "internal"),
    }


def test_project_documents_include_confidential_chunks_marked_internal():
    settings = get_settings()
    chunks = load_documents(settings.static_data_dir, settings.chunk_size, settings.chunk_overlap)

    internal = [c for c in chunks if c.metadata["source"] == "internal_operations"]
    assert internal
    assert all(c.metadata["visibility"] == "internal" for c in internal)


def test_search_restricts_results_to_public_chunks_and_maps_hits():
    document = Document(
        page_content="Opening text",
        metadata={"chunk_id": "faq#x", "source": "faq", "section": "x"},
    )
    store = FakeVectorStore([(document, 0.25)])

    passages = VectorKnowledgeBase(store, default_k=4).search("when open?")

    assert store.calls == [{"query": "when open?", "k": 4, "expr": PUBLIC_ONLY_FILTER}]
    assert passages[0].chunk_id == "faq#x"
    assert passages[0].distance == 0.25
