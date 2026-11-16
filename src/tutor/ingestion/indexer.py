"""Индексация материалов: повторный запуск обрабатывает только изменившиеся файлы."""
from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

from loguru import logger

from tutor.config import Settings
from tutor.ingestion.chunker import chunk_sections, embedding_text
from tutor.ingestion.loaders import load_sections
from tutor.ingestion.manifest import build_manifest
from tutor.llm.base import Embedder
from tutor.models import SourceMeta
from tutor.retrieval.store import ChunkStore

BATCH = 16


@dataclass
class IngestReport:
    indexed: list[str] = field(default_factory=list)
    skipped: int = 0
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    chunks_added: int = 0

    def summary(self) -> str:
        return (f"проиндексировано файлов: {len(self.indexed)}, без изменений: {self.skipped}, "
                f"удалено: {len(self.removed)}, ошибок: {len(self.failed)}, новых чанков: {self.chunks_added}")


def _state(settings: Settings) -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS indexed_files ("
                 "file TEXT PRIMARY KEY, digest TEXT NOT NULL, chunks INTEGER NOT NULL, indexed_at TEXT NOT NULL)")
    return conn


def _digest(settings: Settings, meta: SourceMeta) -> str:
    h = hashlib.sha256((settings.materials_dir / meta.file).read_bytes())
    h.update(meta.model_dump_json().encode())
    h.update(f"{settings.chunk_size}|{settings.chunk_overlap}|{settings.embeddings_model}".encode())
    return h.hexdigest()


def ingest(settings: Settings, embedder: Embedder, store: ChunkStore, full: bool = False) -> IngestReport:
    report = IngestReport()
    manifest = build_manifest(settings.materials_dir)
    conn = _state(settings)
    known = dict(conn.execute("SELECT file, digest FROM indexed_files"))
    if full:
        known = {f: "" for f in known}

    for file in sorted(set(known) - {m.file for m in manifest}):
        store.delete_file(file)
        conn.execute("DELETE FROM indexed_files WHERE file = ?", (file,))
        conn.commit()
        report.removed.append(file)

    for n, meta in enumerate(manifest, 1):
        digest = _digest(settings, meta)
        if known.get(meta.file) == digest:
            report.skipped += 1
            continue
        try:
            sections = load_sections(settings.materials_dir / meta.file)
            chunks = chunk_sections(sections, meta, settings.chunk_size, settings.chunk_overlap)
            vectors: list[list[float]] = []
            for i in range(0, len(chunks), BATCH):
                vectors += embedder.embed_documents([embedding_text(c) for c in chunks[i:i + BATCH]])
            store.delete_file(meta.file)
            store.add(chunks, vectors)
        except Exception as e:  # noqa: BLE001 — один плохой файл не должен ронять индексацию
            logger.warning("не удалось проиндексировать {}: {}", meta.file, e)
            report.failed[meta.file] = str(e)
            continue
        conn.execute("INSERT OR REPLACE INTO indexed_files VALUES (?, ?, ?, ?)",
                     (meta.file, digest, len(chunks), datetime.now(UTC).isoformat()))
        conn.commit()
        report.indexed.append(meta.file)
        report.chunks_added += len(chunks)
        logger.info("[{}/{}] {} — чанков: {}", n, len(manifest), meta.file, len(chunks))
    conn.close()
    return report
