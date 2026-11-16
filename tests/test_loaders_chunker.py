from __future__ import annotations

from pathlib import Path

from tutor.ingestion.chunker import chunk_sections, clean_text, embedding_text, merge_small
from tutor.ingestion.loaders import Section, load_docx, load_markdown, load_pdf, markdown_sections
from tutor.models import SourceMeta, levels_in

META = SourceMeta(file="web/book/x.md", title="Doc", level="A1–A2", topics=["a1-present-simple"],
                  license="CC BY-SA 4.0", url="https://example.org/x")


def test_markdown_sections_follow_headings(materials: Path):
    meta, sections = load_markdown(materials / "web/book/present-simple.md")
    assert meta["license"] == "CC BY-SA 4.0"
    paths = [s.heading_path for s in sections]
    # название документа (H1) в путь раздела не входит, жирная строка считается подзаголовком
    assert paths == [("Usage",), ("Third person singular", "Spelling")]
    assert "habits" in sections[0].text
    assert sections[1].title == "Third person singular › Spelling"


def test_headings_inside_code_fence_are_ignored():
    sections = markdown_sections("## Real\n\ntext\n\n```\n## not a heading\n```\n")
    assert [s.heading_path for s in sections] == [("Real",)]
    assert "not a heading" in sections[0].text


def test_bold_line_inside_paragraph_is_not_a_heading():
    sections = markdown_sections("## A\n\nfirst line\n**bold inside**\nlast line\n")
    assert len(sections) == 1 and "**bold inside**" in sections[0].text


def test_long_section_is_split_with_metadata():
    text = " ".join(f"Sentence number {i} about the present simple tense." for i in range(60))
    chunks = chunk_sections([Section(("Usage",), text)], META, size=400, overlap=40)
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 for c in chunks)
    assert {c.section for c in chunks} == {"Usage"}
    assert chunks[0].file == META.file and chunks[0].level == "A1–A2" and chunks[0].topics == META.topics
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.order for c in chunks] == list(range(len(chunks)))


def test_small_sections_are_merged_into_next():
    sections = [Section(("Intro",), "Short note."), Section(("Rule",), "x" * 300)]
    merged = merge_small(sections)
    assert len(merged) == 1
    assert merged[0].heading_path == ("Rule",)
    assert merged[0].text.startswith("Intro\nShort note.")


def test_trailing_small_section_is_kept():
    merged = merge_small([Section(("Rule",), "x" * 300), Section(("Note",), "Remember this.")])
    assert len(merged) == 1 and "Remember this." in merged[0].text


def test_clean_text_collapses_blanks():
    assert clean_text("I \\_\\_\\_\\_\\_\\_\\_\\_ (go) home.") == "I ____ (go) home."


def test_embedding_text_has_title_and_section():
    chunk = chunk_sections([Section(("Usage",), "y" * 250)], META, 400, 40)[0]
    assert embedding_text(chunk).startswith("Doc › Usage\n\n")


def test_levels_in():
    assert levels_in("A1–B1") == ["A1", "A2", "B1"]
    assert levels_in("A2") == ["A2"]
    assert levels_in("A2–B1") == ["A2", "B1"]
    assert levels_in("") == ["A1", "A2", "B1"]


def test_pdf_pages_and_footer_removal(tmp_path: Path):
    import pymupdf as fitz

    path = tmp_path / "sheet.pdf"
    doc = fitz.open()
    for body in ("PAST SIMPLE VERBS\nWrite the past simple form of each verb.", "ANSWER KEY\n1. worked 2. went"):
        page = doc.new_page()
        page.insert_text((72, 72), body + "\nLicensed under CC BY 4.0\nESL-0008 | v1.1 | 1 / 2")
    doc.save(path)
    sections = load_pdf(path)
    assert [s.page for s in sections] == [1, 2]
    assert sections[0].heading_path == ("Past Simple Verbs",)
    assert "worked" in sections[1].text
    assert all("Licensed under" not in s.text and "ESL-0008" not in s.text for s in sections)


def test_docx_headings(tmp_path: Path):
    import docx

    path = tmp_path / "notes.docx"
    d = docx.Document()
    d.add_heading("Modal verbs", level=1)
    d.add_paragraph("Can expresses ability.")
    d.add_heading("Must", level=2)
    d.add_paragraph("Must expresses obligation.")
    d.save(str(path))
    sections = load_docx(path)
    assert [s.heading_path for s in sections] == [("Modal verbs",), ("Modal verbs", "Must")]
    assert "obligation" in sections[1].text
