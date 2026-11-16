# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "pyyaml>=6"]
# ///
"""Выборка пар предложений eng–rus из Tatoeba для уровней A1–B1.

Скачивает три файла выгрузки (≈61 МБ) в materials/.raw/tatoeba/, отбирает короткие
предложения, все слова которых входят в CEFR-J Vocabulary Profile не выше B1, и
пишет Markdown с указанием авторов (лицензия CC BY 2.0 FR требует атрибуции).

Уровень предложения = максимальный уровень его слов по CEFR-J. Грамматика при этом
не учитывается, поэтому уровень — оценка «по лексике».

Запуск:  uv run scripts/build_tatoeba_examples.py [--per-level 1200]
"""
from __future__ import annotations

import argparse
import bz2
import csv
import random
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "materials" / ".raw" / "tatoeba"
OUT = ROOT / "materials" / "web" / "tatoeba"
VOCAB = ROOT / "materials" / "web" / "cefr-j" / "data" / "cefrj-vocabulary-profile-1.5.csv"
UA = "EnglishTutorRAG/0.1 (personal educational project; rate-limited, cached)"
BASE = "https://downloads.tatoeba.org/exports/per_language/"
FILES = {
    "eng": "eng/eng_sentences_detailed.tsv.bz2",
    "rus": "rus/rus_sentences_detailed.tsv.bz2",
    "links": "eng/eng-rus_links.tsv.bz2",
}
RANK = {"A1": 1, "A2": 2, "B1": 3}
NAMES = {1: "A1", 2: "A2", 3: "B1"}

IRREGULAR = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be",
    "has": "have", "had": "have", "does": "do", "did": "do", "done": "do", "went": "go", "gone": "go",
    "came": "come", "saw": "see", "seen": "see", "got": "get", "gotten": "get", "made": "make",
    "said": "say", "took": "take", "taken": "take", "knew": "know", "known": "know", "thought": "think",
    "gave": "give", "given": "give", "found": "find", "told": "tell", "felt": "feel", "left": "leave",
    "kept": "keep", "began": "begin", "begun": "begin", "wrote": "write", "written": "write",
    "ran": "run", "bought": "buy", "brought": "bring", "sat": "sit", "stood": "stand", "lost": "lose",
    "paid": "pay", "met": "meet", "spoke": "speak", "spoken": "speak", "read": "read", "ate": "eat",
    "eaten": "eat", "drank": "drink", "drunk": "drink", "slept": "sleep", "sold": "sell", "sent": "send",
    "built": "build", "fell": "fall", "fallen": "fall", "heard": "hear", "held": "hold", "taught": "teach",
    "caught": "catch", "broke": "break", "broken": "break", "chose": "choose", "chosen": "choose",
    "drove": "drive", "driven": "drive", "flew": "fly", "flown": "fly", "forgot": "forget",
    "forgotten": "forget", "grew": "grow", "grown": "grow", "swam": "swim", "wore": "wear", "worn": "wear",
    "won": "win", "sang": "sing", "sung": "sing", "spent": "spend", "understood": "understand",
    "woke": "wake", "threw": "throw", "thrown": "throw", "rode": "ride", "became": "become",
    "children": "child", "men": "man", "women": "woman", "people": "person", "feet": "foot",
    "teeth": "tooth", "mice": "mouse", "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "me": "i", "him": "he", "her": "she", "us": "we", "them": "they", "my": "i", "your": "you",
    "his": "he", "its": "it", "our": "we", "their": "they", "mine": "i", "yours": "you", "ours": "we",
    "theirs": "they", "hers": "she", "myself": "i", "yourself": "you", "himself": "he", "herself": "she",
    "these": "this", "those": "that", "an": "a", "cannot": "can", "could": "can", "would": "will",
    "can't": "can", "won't": "will", "shan't": "shall", "let's": "let", "o'clock": "clock",
}


def load_vocab() -> dict[str, int]:
    vocab: dict[str, int] = {}
    with VOCAB.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rank = RANK.get(row["CEFR"])
            if rank is None:
                continue
            for word in row["headword"].lower().split("/"):
                word = word.strip()
                if word and rank < vocab.get(word, 9):
                    vocab[word] = rank
    return vocab


def candidates(w: str):
    yield w
    if w in IRREGULAR:
        yield IRREGULAR[w]
    if w.endswith("n't"):
        yield w[:-3]
    if "'" in w:
        yield w.split("'")[0]
    for suf, repls in (("ies", ["y"]), ("es", [""]), ("s", [""]), ("ied", ["y"]), ("ed", ["", "e"]),
                       ("ing", ["", "e"]), ("ier", ["y"]), ("iest", ["y"]), ("er", ["", "e"]),
                       ("est", ["", "e"]), ("ly", [""])):
        if w.endswith(suf) and len(w) - len(suf) >= 2:
            stem = w[: -len(suf)]
            for r in repls:
                yield stem + r
            if len(stem) > 2 and stem[-1] == stem[-2]:  # running -> run, bigger -> big
                yield stem[:-1]


def sentence_level(text: str, vocab: dict[str, int]) -> int | None:
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    if not 3 <= len(tokens) <= 12:
        return None
    level = 1
    for i, tok in enumerate(tokens):
        got = min((vocab[c] for c in candidates(tok.lower()) if c in vocab), default=None)
        if got is None:
            if tok[0].isupper() and i > 0:  # имя собственное
                continue
            return None
        level = max(level, got)
    return level


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": UA}, timeout=300, follow_redirects=True) as client:
        for rel in FILES.values():
            dest = RAW / Path(rel).name
            if dest.exists():
                continue
            print("скачиваю", rel, flush=True)
            with client.stream("GET", BASE + rel) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)


def rows(key: str):
    with bz2.open(RAW / Path(FILES[key]).name, "rt", encoding="utf-8", newline="") as f:
        for line in f:
            yield line.rstrip("\n").split("\t")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-level", type=int, default=1200)
    args = ap.parse_args()
    download()
    vocab = load_vocab()

    eng: dict[str, tuple[str, str, int]] = {}
    seen_text: set[str] = set()
    for r in rows("eng"):
        if len(r) < 4:
            continue
        sid, _, text, user = r[:4]
        if not text.isascii() or re.search(r"\d", text) or text[-1:] not in ".?!" or not text[0].isupper():
            continue
        if user in ("\\N", "") or text.lower() in seen_text:
            continue
        lvl = sentence_level(text, vocab)
        if lvl:
            eng[sid] = (text, user, lvl)
            seen_text.add(text.lower())
    print("кандидатов eng:", len(eng), flush=True)

    pairs: dict[str, str] = {}
    for r in rows("links"):
        if len(r) >= 2 and r[0] in eng and (r[0] not in pairs or int(r[1]) < int(pairs[r[0]])):
            pairs[r[0]] = r[1]
    wanted = set(pairs.values())
    rus: dict[str, tuple[str, str]] = {}
    for r in rows("rus"):
        if len(r) >= 4 and r[0] in wanted and r[3] not in ("\\N", ""):
            rus[r[0]] = (r[2], r[3])
    print("пар с переводом:", sum(1 for e in pairs if pairs[e] in rus), flush=True)

    by_level: dict[int, list[str]] = defaultdict(list)
    for sid in sorted(pairs, key=int):
        if pairs[sid] in rus:
            by_level[eng[sid][2]].append(sid)

    rng = random.Random(42)
    OUT.mkdir(parents=True, exist_ok=True)
    for lvl in (1, 2, 3):
        ids = by_level[lvl]
        rng.shuffle(ids)
        chosen, starts, names = [], defaultdict(int), 0
        for sid in ids:
            text = eng[sid][0]
            start = " ".join(text.lower().split()[:3])
            has_name = bool(re.search(r"\b(Tom|Mary)\b", text))
            if starts[start] >= 2 or (has_name and names >= args.per_level // 10):
                continue
            starts[start] += 1
            names += has_name
            chosen.append(sid)
            if len(chosen) >= args.per_level:
                break
        chosen.sort(key=lambda s: (len(eng[s][0].split()), eng[s][0]))
        name = NAMES[lvl]
        meta = {
            "title": f"Tatoeba: примеры предложений с переводом, уровень {name}",
            "source": "Tatoeba",
            "source_url": "https://tatoeba.org",
            "download_url": BASE,
            "author": "Участники проекта Tatoeba (автор указан у каждого предложения)",
            "license": "CC BY 2.0 FR",
            "license_url": "https://creativecommons.org/licenses/by/2.0/fr/",
            "retrieved": date.today().isoformat(),
            "language": "en, ru",
            "kind": "examples",
            "level": name,
            "note": "Уровень определён по лексике (CEFR-J Vocabulary Profile 1.5), грамматика не оценивалась.",
        }
        lines = []
        for i, sid in enumerate(chosen):
            if i % 40 == 0:
                lines.append(f"\n## {name}: примеры {i + 1}–{min(i + 40, len(chosen))}\n")
            en_text, en_user, _ = eng[sid]
            ru_text, ru_user = rus[pairs[sid]]
            lines.append(f"- **{en_text}** — {ru_text} (eng #{sid}, {en_user}; rus #{pairs[sid]}, {ru_user})")
        fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=1000).strip()
        path = OUT / f"examples-{name.lower()}.md"
        path.write_text(f"---\n{fm}\n---\n\n# {meta['title']}\n" + "\n".join(lines) + "\n", "utf-8")
        print(f"{path.relative_to(ROOT)}: {len(chosen)} пар (доступно {len(ids)})", flush=True)


if __name__ == "__main__":
    main()
