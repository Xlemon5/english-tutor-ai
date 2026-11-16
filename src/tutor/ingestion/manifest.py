"""Манифест materials/sources.yaml и отчёт о покрытии карты тем materials/coverage.md.

Источники метаданных:
- Markdown из materials/web/ — front matter файла (его пишет загрузчик материалов);
- PDF и DOCX из materials/web/<коллекция>/ — файл _collection.yaml в корне коллекции
  (общие поля и шаблон имени файла);
- всё из materials/own/ — собственные материалы без лицензии.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

from tutor.ingestion.loaders import SUPPORTED, split_front_matter
from tutor.models import LEVELS, SourceMeta

# Уровень коллекций, где он известен из описания источника (оценка).
DIR_LEVELS = {
    "wikibooks-ru-english": "A1–A2",
    "wikibooks-en-english-grammar": "A2",
    "wikibooks-en-english-in-use": "A2–B1",
    "libretexts-explorations-1": "A1–A2",
    "libretexts-writing-for-success": "B1",
    "libretexts-bissonnette-esl-grammar/book-1": "A1",
    "libretexts-bissonnette-esl-grammar/book-2": "A2",
    "libretexts-bissonnette-esl-grammar/book-3": "A2–B1",
    "libretexts-bissonnette-esl-grammar/book-4": "B1",
}
AREAS = {"grammar": "Грамматика", "vocabulary": "Лексика", "writing": "Письмо"}
NO_TOPIC_KINDS = {"examples"}


def load_topics(materials: Path) -> list[dict]:
    path = materials / "topics.yaml"
    if not path.exists():
        return []
    topics = []
    for area, items in yaml.safe_load(path.read_text("utf-8")).items():
        for t in items:
            t = dict(t, area=area)
            t["re"] = re.compile("|".join(f"(?:{p})" for p in t["match"]), re.IGNORECASE | re.MULTILINE)
            topics.append(t)
    return topics


def _level_range(levels: list[str]) -> str:
    lv = sorted(set(levels), key=LEVELS.index)
    return lv[0] if len(lv) == 1 else f"{lv[0]}–{lv[-1]}"


def _collection_defaults(path: Path, web: Path) -> dict:
    for parent in path.parents:
        cfg = parent / "_collection.yaml"
        if cfg.exists():
            return yaml.safe_load(cfg.read_text("utf-8")) or {}
        if parent == web:
            break
    return {}


def _describe(path: Path, materials: Path) -> tuple[dict, str]:
    """Возвращает (метаданные, текст заголовков для сопоставления с темами)."""
    rel = path.relative_to(materials).as_posix()
    if rel.startswith("own/"):
        meta = {"title": path.stem, "source": "Собственные материалы", "author": "владелец проекта",
                "license": "собственные материалы"}
        headings = ""
        if path.suffix.lower() == ".md":
            fm, body = split_front_matter(path.read_text("utf-8"))
            meta.update({k: v for k, v in fm.items() if v})
            headings = "\n".join(ln for ln in body.splitlines() if ln.startswith("#"))
        return meta, headings
    if path.suffix.lower() == ".md":
        fm, body = split_front_matter(path.read_text("utf-8"))
        fm.setdefault("title", path.stem)
        return fm, "\n".join(ln for ln in body.splitlines() if ln.startswith("#"))
    meta = dict(_collection_defaults(path, materials / "web"))
    pattern = meta.pop("filename_pattern", None)
    meta["title"] = path.stem
    if pattern and (m := re.match(pattern, path.stem)):
        parts = m.groupdict()
        if parts.get("title"):
            meta["title"] = parts["title"].replace("-", " ").replace("_", " ")
        if parts.get("level"):
            meta["level"] = parts["level"].replace("-", "–")
    return meta, ""


def build_manifest(materials: Path) -> list[SourceMeta]:
    """Пересобирает sources.yaml и coverage.md, возвращает записи манифеста."""
    topics = load_topics(materials)
    entries: list[SourceMeta] = []
    covered: dict[str, list[str]] = defaultdict(list)
    files = sorted(
        p for base in ("web", "own") for p in (materials / base).rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED and "/data/" not in p.as_posix()
    )
    for path in files:
        rel = path.relative_to(materials).as_posix()
        meta, headings = _describe(path, materials)
        # LibreTexts нумерует страницы («3.1.1.1: Passive Voice») — в названии это только мешает
        title = re.sub(r"^\d+(?:\.\d+)*:\s*", "", str(meta.get("title") or path.stem))
        kind = str(meta.get("kind") or "theory")
        in_title, in_any = [], []
        if kind not in NO_TOPIC_KINDS:
            for t in topics:
                if t["re"].search(title):
                    in_title.append(t)
                    in_any.append(t)
                elif headings and t["re"].search(headings):
                    in_any.append(t)
        for t in in_any:
            covered[t["id"]].append(rel)
        sub = rel.split("/", 1)[1] if "/" in rel else rel
        level = meta.get("level") or next((v for k, v in DIR_LEVELS.items() if sub.startswith(k + "/")), "")
        if not level:
            basis = in_title or in_any
            level = _level_range([t["level"] for t in basis]) if basis else "A1–B1"
        entries.append(SourceMeta(
            file=rel, title=title, source=str(meta.get("source") or ""),
            url=str(meta.get("source_url") or meta.get("url") or ""), author=str(meta.get("author") or ""),
            license=str(meta.get("license") or ""), license_url=str(meta.get("license_url") or ""),
            retrieved=str(meta.get("retrieved") or ""), language=str(meta.get("language") or ""),
            kind=kind, level=str(level), topics=[t["id"] for t in in_any],
        ))

    collections: dict[str, dict] = {}
    for e in entries:
        parts = e.file.split("/")
        key = parts[1] if parts[0] == "web" and len(parts) > 2 else parts[0]
        c = collections.setdefault(key, {"id": key, "source": e.source, "license": e.license, "files": 0})
        c["files"] += 1
    manifest = {
        "generated": date.today().isoformat(),
        "collections": list(collections.values()),
        "files": [{k: v for k, v in e.model_dump().items() if v not in ("", [])} for e in entries],
    }
    header = "# Манифест учебных материалов. Генерируется командой `tutor manifest` — вручную не править.\n"
    (materials / "sources.yaml").write_text(
        header + yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=1000), "utf-8")
    if topics:
        _write_coverage(materials, topics, covered)
    return entries


def _write_coverage(materials: Path, topics: list[dict], covered: dict[str, list[str]]) -> None:
    note = ("Тема считается закрытой, если её ключевые слова найдены в названии или заголовках хотя бы "
            "одного файла. Это автоматическая оценка: глубину изложения она не проверяет.")
    lines = [f"# Покрытие карты тем материалами ({date.today().isoformat()})", "", note, ""]
    gaps = []
    for area, name in AREAS.items():
        lines += [f"## {name}", "", "| Уровень | Тема | Файлов | Примеры файлов |", "|---|---|---|---|"]
        for t in (t for t in topics if t["area"] == area):
            hits = covered.get(t["id"], [])
            sample = ", ".join(f"`{h.split('/', 1)[1]}`" for h in hits[:3])
            lines.append(f"| {t['level']} | {t['title']} | {len(hits)} | {sample or '**нет**'} |")
            if not hits:
                gaps.append(f"{t['level']} · {name} · {t['title']}")
        lines.append("")
    lines += ["## Пробелы", ""] + ([f"- {g}" for g in gaps] or ["Нет."]) + [""]
    (materials / "coverage.md").write_text("\n".join(lines), "utf-8")


def load_manifest(materials: Path) -> list[SourceMeta]:
    path = materials / "sources.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text("utf-8")) or {}
    return [SourceMeta(**f) for f in data.get("files", [])]


def coverage_gaps(materials: Path) -> list[str]:
    path = materials / "coverage.md"
    if not path.exists():
        return []
    text = path.read_text("utf-8")
    tail = text.split("## Пробелы", 1)[-1]
    return [ln[2:] for ln in tail.splitlines() if ln.startswith("- ")]
