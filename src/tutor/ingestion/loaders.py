"""Чтение материалов (Markdown, PDF, DOCX) в список смысловых разделов."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

SUPPORTED = (".md", ".pdf", ".docx")

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# Строка целиком жирным шрифтом — в части учебников так оформлены подзаголовки.
_BOLD_HEADING = re.compile(r"^\*\*([^*\n]{3,80}?)\*\*:?$")
_PAGE_FOOTER = re.compile(r"^(\S+ \| )?(v[\d.]+ \| )?\d+ ?/ ?\d+$|^(page|стр\.?) ?\d+( (of|из) \d+)?$", re.IGNORECASE)
_PSEUDO_LEVEL = 7


@dataclass
class Section:
    heading_path: tuple[str, ...]
    text: str
    page: int | None = None

    @property
    def title(self) -> str:
        return " › ".join(self.heading_path)


def split_front_matter(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return yaml.safe_load(text[4:end]) or {}, text[end + 5:]
    return {}, text


def markdown_sections(body: str, doc_title: str = "") -> list[Section]:
    """Делит Markdown по заголовкам. Путь раздела — цепочка заголовков без названия документа."""
    lines = body.splitlines()
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            path = tuple(t for lvl, t in stack if not (lvl == 1 and t == doc_title))
            sections.append(Section(path, text))
        buf.clear()

    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        level, title = None, ""
        if not in_fence:
            m = _HEADING.match(line)
            if m:
                level, title = len(m.group(1)), m.group(2)
            else:
                b = _BOLD_HEADING.match(line.strip())
                alone = (i == 0 or not lines[i - 1].strip()) and (i + 1 >= len(lines) or not lines[i + 1].strip())
                if b and alone:
                    level, title = _PSEUDO_LEVEL, b.group(1).strip().rstrip(":")
        if level is None:
            buf.append(line)
            continue
        flush()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, re.sub(r"[*_`]", "", title).strip()))
    flush()
    return sections


def load_markdown(path: Path) -> tuple[dict, list[Section]]:
    meta, body = split_front_matter(path.read_text("utf-8"))
    return meta, markdown_sections(body, str(meta.get("title", "")))


def _drop_repeated_lines(pages: list[list[str]]) -> list[list[str]]:
    """Убирает колонтитулы: строки, повторяющиеся почти на каждой странице, и номера страниц."""
    n = len(pages)
    repeated: set[str] = set()
    if n >= 2:
        counts = Counter(line for page in pages for line in set(page))
        need = n if n < 4 else max(2, int(n * 0.6))
        repeated = {line for line, c in counts.items() if c >= need and len(line) < 140}
    return [[ln for ln in page if ln not in repeated and not _PAGE_FOOTER.match(ln)] for page in pages]


def load_pdf(path: Path) -> list[Section]:
    """Одна страница — один раздел, чтобы в ссылке можно было указать номер страницы."""
    import pymupdf as fitz

    with fitz.open(path) as doc:
        raw = [[ln.strip() for ln in page.get_text("text").splitlines() if ln.strip()] for page in doc]
    sections = []
    for number, lines in enumerate(_drop_repeated_lines(raw), 1):
        if not lines:
            continue
        first = lines[0]
        heading = (first.title() if first.isupper() else first,) if len(first) <= 70 else ()
        sections.append(Section(heading, "\n".join(lines), page=number))
    return sections


def load_docx(path: Path) -> list[Section]:
    import docx

    document = docx.Document(str(path))
    lines: list[str] = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            lines.append("")
            continue
        style = (para.style.name or "").lower() if para.style is not None else ""
        m = re.match(r"heading (\d)", style)
        if m:
            lines += ["", "#" * int(m.group(1)) + " " + text, ""]
        elif style == "title":
            lines += ["# " + text, ""]
        else:
            lines += [text, ""]
    for table in document.tables:
        lines.append("")
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    return markdown_sections("\n".join(lines))


def load_sections(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return load_markdown(path)[1]
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    raise ValueError(f"Неподдерживаемый формат: {path.name}")
