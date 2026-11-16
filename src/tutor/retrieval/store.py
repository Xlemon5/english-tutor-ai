"""Векторное хранилище чанков поверх ChromaDB (встроенный режим или отдельный сервер)."""
from __future__ import annotations

import chromadb

from tutor.config import Settings
from tutor.models import Chunk

PAGE = 500


class ChunkStore:
    def __init__(self, settings: Settings) -> None:
        if settings.chroma_host:
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        else:
            settings.chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.col = client.get_or_create_collection(
            settings.chroma_collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for i in range(0, len(chunks), PAGE):
            part = chunks[i:i + PAGE]
            self.col.add(
                ids=[c.id for c in part], documents=[c.text for c in part],
                metadatas=[c.to_metadata() for c in part], embeddings=embeddings[i:i + PAGE],
            )

    def delete_file(self, file: str) -> None:
        self.col.delete(where={"file": file})

    def count(self) -> int:
        return self.col.count()

    def query(self, embedding: list[float], n: int, where: dict | None = None) -> list[tuple[Chunk, float, list[float]]]:
        """Ближайшие чанки: (чанк, косинусная близость, вектор)."""
        if self.count() == 0:
            return []
        res = self.col.query(query_embeddings=[embedding], n_results=n, where=where or None,
                             include=["documents", "metadatas", "distances", "embeddings"])
        return [
            (Chunk.from_store(id_, doc, meta), 1.0 - float(dist), list(emb))
            for id_, doc, meta, dist, emb in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], res["embeddings"][0])
        ]

    def get(self, ids: list[str]) -> list[tuple[Chunk, list[float]]]:
        if not ids:
            return []
        res = self.col.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        return [(Chunk.from_store(i, d, m), list(e))
                for i, d, m, e in zip(res["ids"], res["documents"], res["metadatas"], res["embeddings"])]

    def all_chunks(self) -> list[Chunk]:
        out: list[Chunk] = []
        offset = 0
        while True:
            res = self.col.get(include=["documents", "metadatas"], limit=PAGE, offset=offset)
            if not res["ids"]:
                return out
            out += [Chunk.from_store(i, d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
            offset += PAGE
