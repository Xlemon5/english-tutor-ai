"""Чанкинг по смысловым разделам: один раздел — один чанк, длинные разделы делятся."""
from __future__ import annotations

import hashlib
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from tutor.ingestion.loaders import Section
from tutor.models import Chunk, SourceMeta

MIN_SECTION_CHARS = 200


def clean_text(text: str) -> str:
    text = re.sub(r"(\\?_){4,}", "____", text)  # пропуски в упражнениях
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def merge_small(sections: list[Section], min_chars: int = MIN_SECTION_CHARS) -> list[Section]:
    """Короткий раздел присоединяется к следующему на той же странице, заголовок сохраняется в тексте."""
    out: list[Section] = []
    carry: Section | None = None
    for sec in sections:
        if carry is not None:
            if carry.page == sec.page:
                head = f"{carry.heading_path[-1]}\n" if carry.heading_path else ""
                sec = Section(sec.heading_path, f"{head}{carry.text}\n\n{sec.text}", sec.page)
            else:
                out.append(carry)
            carry = None
        if len(sec.text) < min_chars:
            carry = sec
        else:
            out.append(sec)
    if carry is not None:
        if out and out[-1].page == carry.page:
            last = out[-1]
            head = f"{carry.heading_path[-1]}\n" if carry.heading_path else ""
            out[-1] = Section(last.heading_path, f"{last.text}\n\n{head}{carry.text}", last.page)
        else:
            out.append(carry)
    return out


def chunk_sections(sections: list[Section], meta: SourceMeta, size: int, overlap: int) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
    cleaned = [Section(s.heading_path, clean_text(s.text), s.page) for s in sections]
    chunks: list[Chunk] = []
    for sec in merge_small([s for s in cleaned if s.text]):
        parts = [sec.text] if len(sec.text) <= size else splitter.split_text(sec.text)
        for part in parts:
            part = part.strip()
            if len(part) < 20:
                continue
            order = len(chunks)
            digest = hashlib.sha1(f"{meta.file}|{order}|{part}".encode()).hexdigest()[:20]
            chunks.append(Chunk(
                id=digest, text=part, order=order, file=meta.file, doc_title=meta.title,
                section=sec.title, page=sec.page, level=meta.level, topics=meta.topics,
                language=meta.language, kind=meta.kind, source=meta.source, url=meta.url,
                author=meta.author, license=meta.license, license_url=meta.license_url,
            ))
    return chunks


def embedding_text(chunk: Chunk) -> str:
    """Текст для эмбеддинга: название документа и раздела помогают найти чанк по теме."""
    head = " › ".join(p for p in (chunk.doc_title, chunk.section) if p)
    return f"{head}\n\n{chunk.text}"
