from __future__ import annotations

from pathlib import Path

import yaml

from tutor.config import Settings
from tutor.ingestion.indexer import ingest
from tutor.ingestion.manifest import build_manifest, coverage_gaps, load_manifest
from tutor.retrieval.citations import build_citations
from tutor.retrieval.hybrid import Retriever


def test_manifest_has_license_url_and_topics(materials: Path):
    entries = {e.file: e for e in build_manifest(materials)}
    ps = entries["web/book/present-simple.md"]
    assert ps.license == "CC BY-SA 4.0" and ps.url == "https://example.org/present-simple"
    assert ps.topics == ["a1-present-simple"] and ps.level == "A1"
    data = yaml.safe_load((materials / "sources.yaml").read_text("utf-8"))
    assert len(data["files"]) == 2 and data["collections"][0]["files"] == 2
    assert [e.file for e in load_manifest(materials)] == sorted(entries)
    assert coverage_gaps(materials) == ["B1 · Грамматика · Пассивный залог"]


def test_manifest_pdf_collection_and_own_files(materials: Path):
    import pymupdf as fitz

    col = materials / "web" / "sheets"
    col.mkdir()
    (col / "_collection.yaml").write_text(
        "source: Sheets\nlicense: CC BY 4.0\nsource_url: https://example.org\nkind: worksheet\n"
        "filename_pattern: '^(?P<id>ESL-\\d+)_(?P<title>.+)_(?P<level>[ABC]\\d(?:-[ABC]\\d)?)$'\n", "utf-8")
    for name in ("ESL-0001_Articles-A-An-The_A1.pdf", "ESL-0002_Role-Play_A2-B1.pdf"):
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "text")
        doc.save(col / name)
    (materials / "own" / "my-notes.md").write_text("# Notes\n\n## Passive voice\n\ntext\n", "utf-8")
    entries = {e.file: e for e in build_manifest(materials)}
    sheet = entries["web/sheets/ESL-0001_Articles-A-An-The_A1.pdf"]
    assert (sheet.title, sheet.level, sheet.license, sheet.kind) == ("Articles A An The", "A1", "CC BY 4.0", "worksheet")
    assert sheet.topics == ["a1-articles"]
    assert entries["web/sheets/ESL-0002_Role-Play_A2-B1.pdf"].level == "A2–B1"
    own = entries["own/my-notes.md"]
    assert own.license == "собственные материалы" and own.topics == ["b1-passive"]
    assert coverage_gaps(materials) == []


def test_ingest_is_incremental(settings: Settings, embedder, store, materials: Path):
    first = ingest(settings, embedder, store)
    assert len(first.indexed) == 2 and first.skipped == 0 and store.count() == first.chunks_added > 0
    total = store.count()

    second = ingest(settings, embedder, store)
    assert second.indexed == [] and second.skipped == 2 and store.count() == total

    path = materials / "web/book/articles.md"
    path.write_text(path.read_text("utf-8") + "\n## Zero article\n\n" + "No article is used with plural nouns in general statements. " * 5, "utf-8")
    third = ingest(settings, embedder, store)
    assert third.indexed == ["web/book/articles.md"] and third.skipped == 1
    assert store.count() > total
    assert any(c.section == "Zero article" for c in store.all_chunks())

    path.unlink()
    fourth = ingest(settings, embedder, store)
    assert fourth.removed == ["web/book/articles.md"]
    assert {c.file for c in store.all_chunks()} == {"web/book/present-simple.md"}


def test_search_finds_section_and_builds_citation(settings: Settings, embedder, store):
    ingest(settings, embedder, store)
    retriever = Retriever(store, embedder, settings)
    result = retriever.search("third person singular ending -s or -es")
    assert result.relevant
    top = result.hits[0].chunk
    assert top.file == "web/book/present-simple.md" and top.section == "Third person singular › Spelling"

    citations = build_citations(result.hits)
    c = citations[0]
    assert [x.n for x in citations] == list(range(1, len(citations) + 1))
    assert (c.file, c.url, c.license) == (top.file, "https://example.org/present-simple", "CC BY-SA 4.0")
    label = c.label()
    assert label.startswith("[1] Present Simple › Third person singular › Spelling — web/book/present-simple.md")
    assert "Test Author, CC BY-SA 4.0" in label  # атрибуция обязательна для CC BY-SA


def test_search_off_topic_is_not_relevant(settings: Settings, embedder, store):
    ingest(settings, embedder, store)
    result = Retriever(store, embedder, settings).search("квантовая хромодинамика кварков")
    assert not result.relevant
    assert result.top_score < settings.relevance_threshold


def test_search_filters_by_kind_and_level(settings: Settings, embedder, store, materials: Path):
    (materials / "web/book/examples.md").write_text(
        "---\ntitle: Examples\nkind: examples\nlevel: B1\n---\n\n# Examples\n\n## B1\n\n"
        + "- **He works in a bank.** — Он работает в банке.\n" * 8, "utf-8")
    ingest(settings, embedder, store)
    retriever = Retriever(store, embedder, settings)
    assert all(h.chunk.kind != "examples" for h in retriever.search("works in a bank").hits)
    only_examples = retriever.search("works in a bank", kinds=("examples",))
    assert only_examples.hits and all(h.chunk.kind == "examples" for h in only_examples.hits)
    assert retriever.search("works in a bank", kinds=None, level="B1").hits[0].chunk.level == "B1"
    assert all(h.chunk.level == "A1" for h in retriever.search("works in a bank", level="A1").hits)


def test_search_on_empty_index(settings: Settings, embedder, store):
    result = Retriever(store, embedder, settings).search("present simple")
    assert result.hits == [] and not result.relevant


def test_search_is_hybrid_semantic_and_lexical(settings: Settings, embedder, store):
    """Гибридный поиск объединяет две выдачи; каждую можно включить отдельно."""
    ingest(settings, embedder, store)
    retriever = Retriever(store, embedder, settings)
    query = "third person singular ending -s or -es"
    hybrid = retriever.search(query)
    assert hybrid.hits[0].dense_rank == 1 and hybrid.hits[0].lexical_rank == 1
    assert hybrid.hits[0].found_by == "семантика+BM25"
    dense = retriever.search(query, mode="dense")
    assert dense.hits and all(h.lexical_rank is None and h.dense_rank for h in dense.hits)
    lexical = retriever.search(query, mode="lexical")
    assert lexical.hits and all(h.dense_rank is None and h.lexical_rank for h in lexical.hits)
    assert lexical.hits[0].chunk.id == dense.hits[0].chunk.id == hybrid.hits[0].chunk.id
