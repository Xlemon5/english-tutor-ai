from __future__ import annotations

from pathlib import Path

import pytest

from tutor.config import Settings
from tutor.llm.base import FakeEmbedder
from tutor.retrieval.store import ChunkStore

PRESENT_SIMPLE = """---
title: Present Simple
source: Test Grammar Book
source_url: https://example.org/present-simple
author: Test Author
license: CC BY-SA 4.0
license_url: https://creativecommons.org/licenses/by-sa/4.0/
retrieved: '2026-11-16'
language: en
---

# Present Simple

## Usage

We use the present simple for habits, routines and facts that are always true. For example:
I drink coffee every morning. The sun rises in the east. She works in a bank and usually
starts at nine o'clock. Adverbs of frequency such as always, often and never are common here.

## Third person singular

**Spelling**

In the third person singular the verb takes the ending -s or -es: he works, she watches,
it goes. Verbs ending in a consonant plus y change y to ies: study becomes studies.
The verbs have, do and go are irregular in the third person singular: has, does, goes.
"""

ARTICLES = """---
title: Articles
source: Test Grammar Book
source_url: https://example.org/articles
author: Test Author
license: CC BY-SA 4.0
language: en
---

# Articles

## Indefinite article

The indefinite article a is used before consonant sounds and an before vowel sounds: a book,
an apple, an hour. We use it when we mention a singular countable noun for the first time.

## Definite article

The definite article the is used when the listener knows which thing we mean: the sun,
the book on the table, the best film of the year.
"""

TOPICS = """
grammar:
  - {id: a1-present-simple, level: A1, title: "Present Simple", match: ['present simple']}
  - {id: a1-articles, level: A1, title: "Артикли", match: ['\\barticles?\\b']}
  - {id: b1-passive, level: B1, title: "Пассивный залог", match: ['passive']}
"""


@pytest.fixture
def materials(tmp_path: Path) -> Path:
    root = tmp_path / "materials"
    (root / "web" / "book").mkdir(parents=True)
    (root / "own").mkdir()
    (root / "web" / "book" / "present-simple.md").write_text(PRESENT_SIMPLE, "utf-8")
    (root / "web" / "book" / "articles.md").write_text(ARTICLES, "utf-8")
    (root / "topics.yaml").write_text(TOPICS, "utf-8")
    return root


@pytest.fixture
def settings(tmp_path: Path, materials: Path) -> Settings:
    return Settings(
        _env_file=None, materials_dir=materials, data_dir=tmp_path / "data", embeddings_provider="fake",
        chunk_size=400, chunk_overlap=40, relevance_threshold=0.2, retrieval_top_k=3,
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(settings: Settings) -> ChunkStore:
    return ChunkStore(settings)
