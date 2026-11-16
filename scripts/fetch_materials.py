# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "beautifulsoup4>=4.12", "lxml>=5", "markdownify>=0.13", "pyyaml>=6"]
# ///
"""Загрузка открытых учебных материалов в materials/web/ (этап 1).

Правила: проверяем robots.txt каждого хоста, между запросами к одному хосту
пауза не меньше 5 с (или Crawl-delay, если он больше), сырые ответы кэшируются
в materials/.raw/, поэтому повторный запуск сеть не трогает.

Запуск:  uv run scripts/fetch_materials.py [--only wikibooks_ru,wikipedia_en,...]
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from datetime import date
from pathlib import Path
from urllib import robotparser
from urllib.parse import quote, unquote, urlsplit

import httpx
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "materials" / ".raw"
WEB = ROOT / "materials" / "web"
UA = "EnglishTutorRAG/0.1 (personal educational project; rate-limited, cached)"
MIN_DELAY = 5.0
TODAY = date.today().isoformat()

CC_URLS = {
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC BY-NC 4.0": "https://creativecommons.org/licenses/by-nc/4.0/",
    "CC BY-NC-SA 4.0": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "CC BY-NC-SA 3.0": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
}


# ---------------------------------------------------------------- сеть


class Fetcher:
    def __init__(self) -> None:
        self.client = httpx.Client(headers={"User-Agent": UA}, timeout=90, follow_redirects=True)
        self.robots: dict[str, robotparser.RobotFileParser] = {}
        self.last: dict[str, float] = {}
        self.network_requests = 0

    def _robots(self, host: str) -> robotparser.RobotFileParser:
        if host not in self.robots:
            rp = robotparser.RobotFileParser()
            try:
                r = self.client.get(f"https://{host}/robots.txt")
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                elif r.status_code in (401, 403) or r.status_code >= 500:
                    rp.disallow_all = True
                else:  # 404 — ограничений нет
                    rp.allow_all = True
            except httpx.HTTPError:
                rp.disallow_all = True
            self.robots[host] = rp
            self.last[host] = time.monotonic()
        return self.robots[host]

    def get(self, url: str, binary: bool = False) -> bytes | str:
        host = urlsplit(url).netloc
        key = hashlib.sha1(url.encode()).hexdigest()[:16]
        path = RAW / "http" / host / (key + (".bin" if binary else ".html"))
        if path.exists():
            return path.read_bytes() if binary else path.read_text("utf-8")
        rp = self._robots(host)
        if not rp.can_fetch(UA, url):
            raise PermissionError(f"robots.txt запрещает {url}")
        delay = max(MIN_DELAY, float(rp.crawl_delay(UA) or 0))
        wait = self.last.get(host, 0.0) + delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        r = self.client.get(url)
        self.last[host] = time.monotonic()
        self.network_requests += 1
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        return r.content if binary else r.content.decode("utf-8", errors="replace")


F = Fetcher()

# ---------------------------------------------------------------- утилиты

_TRANSLIT = dict(zip(
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t",
     "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"],
))


def slug(text: str) -> str:
    text = "".join(_TRANSLIT.get(ch, ch) for ch in text.lower())
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:80] or "page"


def to_markdown(node) -> str:
    md = markdownify(str(node), heading_style="ATX", bullets="-", strip=["a"])
    md = md.replace("\xa0", " ")
    md = re.sub(r"^:[ ]{1,3}", "- ", md, flags=re.M)  # списки определений (примеры в Википедии)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def text_len(md: str) -> int:
    return len(re.sub(r"[\s|\-#*_>]+", " ", md))


def write_md(rel: str, meta: dict, body: str) -> str:
    path = WEB / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False, width=1000).strip()
    path.write_text(f"---\n{fm}\n---\n\n# {meta['title']}\n\n{body.strip()}\n", "utf-8")
    return rel


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- Wikimedia

WM_DROP = (
    ".mw-editsection, .navbox, .vertical-navbox, .sidebar, .noprint, .metadata, .ambox, .hatnote, "
    ".toc, #toc, .reflist, .references, .mw-references-wrap, sup.reference, .mw-empty-elt, style, "
    "script, link, meta, figure, .thumb, img, .sistersitebox, .side-box, .infobox, .printfooter, "
    ".catlinks, .authority-control, .shortdescription, .portalbox, .mw-cite-backlink, .gallery, "
    "audio, .ext-phonos, .navigation-not-searchable, .mw-kartographer-map, .ts-Родственные_проекты, "
    ".wikitable.navigation, .dablink, .rellink, .stub, .ambox-content"
)
WM_TAIL = {
    "see also", "references", "external links", "notes", "further reading", "bibliography",
    "footnotes", "citations", "sources", "notes and references", "works cited",
    "примечания", "литература", "ссылки", "см. также", "источники",
}


def _heading_level(el) -> int | None:
    if not getattr(el, "name", None):
        return None
    if re.fullmatch(r"h[1-6]", el.name):
        return int(el.name[1])
    if el.name == "div" and "mw-heading" in (el.get("class") or []):
        h = el.find(re.compile(r"^h[1-6]$"))
        return int(h.name[1]) if h else None
    return None


def _drop_sections(content, names: set[str]) -> None:
    for h in content.find_all(re.compile(r"^h[2-4]$")):
        if h.decomposed or h.get_text(" ", strip=True).lower() not in names:
            continue
        node = h.parent if "mw-heading" in (h.parent.get("class") or []) else h
        lvl = int(h.name[1])
        for sib in list(node.find_next_siblings()):
            got = _heading_level(sib)
            if got is not None and got <= lvl:
                break
            sib.decompose()
        node.decompose()


def wm_url(host: str, title: str) -> str:
    return f"https://{host}/wiki/{quote(title.replace(' ', '_'), safe='/:()_,')}"


def wm_page(host: str, title: str):
    """Возвращает (заголовок, markdown, BeautifulSoup контента до очистки)."""
    html = F.get(wm_url(host, title))
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if content is None:
        raise ValueError("нет блока контента")
    links = [a.get("href", "") for a in content.select("a[href]") if "new" not in (a.get("class") or [])]
    h1 = soup.select_one("#firstHeading")
    page_title = h1.get_text(" ", strip=True) if h1 else title
    for el in content.select(WM_DROP):
        el.decompose()
    for pre in content.find_all("pre"):
        lines = [ln.strip() for ln in pre.get_text().splitlines()]
        lines = [ln for ln in lines if ln and not re.fullmatch(r"[*\-=_~ ]+", ln)]
        para = soup.new_tag("p")
        for i, ln in enumerate(lines):
            if i:
                para.append(soup.new_tag("br"))
            para.append(ln)
        pre.replace_with(para)
    _drop_sections(content, WM_TAIL)
    return page_title, to_markdown(content), links


def wm_meta(host: str, title: str, page_title: str, source: str, lang: str) -> dict:
    name = title.replace(" ", "_")
    project = "Wikipedia" if "wikipedia" in host else "Wikibooks"
    who = {"en": f"{project} contributors", "ru": f"Авторы проекта {project}"}[lang]
    return {
        "title": page_title,
        "source": source,
        "source_url": f"https://{host}/wiki/{name}",
        "author": who,
        "history_url": f"https://{host}/w/index.php?title={name}&action=history",
        "license": "CC BY-SA 4.0",
        "license_url": CC_URLS["CC BY-SA 4.0"],
        "retrieved": TODAY,
        "language": lang,
    }


def fetch_wikibook(host: str, book: str, source: str, lang: str, out_dir: str,
                   skip: str = r"$^", depth: int = 2, index: str | None = None) -> list[str]:
    index = index or book
    prefix = "/wiki/" + quote(book.replace(" ", "_"), safe="/:()_,") + "/"
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [(index, 0)]
    written = []
    while queue:
        title, d = queue.pop(0)
        if title in seen:
            continue
        seen.add(title)
        try:
            page_title, md, links = wm_page(host, title)
        except Exception as e:  # noqa: BLE001
            log(f"  ! {title}: {e}")
            continue
        if d < depth:
            for href in links:
                href = href.split("#")[0]
                if href.startswith(prefix) and "action=" not in href:
                    child = unquote(href[len("/wiki/"):]).replace("_", " ")
                    if child not in seen and not re.search(skip, child, re.I):
                        queue.append((child, d + 1))
        if title == index:
            continue  # оглавление книги не сохраняем
        if text_len(md) < 300:
            log(f"  - пропуск (заготовка, {text_len(md)} симв.): {title}")
            continue
        short = title[len(book) + 1:]
        rel = f"{out_dir}/{slug(short.replace('/', ' -- '))}.md"
        meta = wm_meta(host, title, page_title.replace("/", " / "), source, lang)
        written.append(write_md(rel, meta, md))
        log(f"  + {rel}")
    return written


def fetch_wikipedia(host: str, titles: list[str], lang: str, out_dir: str) -> list[str]:
    written = []
    for title in titles:
        try:
            page_title, md, _ = wm_page(host, title)
        except Exception as e:  # noqa: BLE001
            log(f"  ! {title}: {e}")
            continue
        source = "Wikipedia" if lang == "en" else "Википедия"
        rel = f"{out_dir}/{slug(title)}.md"
        written.append(write_md(rel, wm_meta(host, title, page_title, source, lang), md))
        log(f"  + {rel}")
    return written


# ---------------------------------------------------------------- LibreTexts

LT_DROP = (
    "script, style, noscript, img, figure, iframe, nav, footer, .mt-listing-detailed, .mt-listings-simple, "
    ".autoattribution, .autoattribution-divider, .mt-content-footer, .mt-related-listings-container, "
    ".MathJax_Preview, .mt-guide-tabs-container, .mt-video-widget, .elm-skip-link, #pageTagsHolder, "
    ".mt-guide-content, .mt-category-container, .printHide, .box-attribution, .glossarizer-tooltip, "
    "header.mt-content-header, .mt-author-container, .mt-icon-article-topic"
)
LT_SKIP = re.compile(r"Front_Matter|Back_Matter|Answer_Key|Detailed_Licensing|Index$|Glossary$", re.I)


def lt_page(url: str):
    html = F.get(url)
    soup = BeautifulSoup(html, "lxml")
    children = []
    for a in soup.select("a[href]"):
        href = a["href"].split("#")[0].split("?")[0]
        if href.startswith(url + "/") and href not in children:
            children.append(href)
    h1 = soup.select_one("h1#title") or soup.select_one("h1")
    title = h1.get_text(" ", strip=True) if h1 else unquote(url.rsplit("/", 1)[-1]).replace("_", " ")
    attr = soup.select_one(".autoattribution")
    attr_text = attr.get_text(" ", strip=True) if attr else ""
    content = soup.select_one("section.mt-content-container")
    md = ""
    if content is not None:
        for el in content.select(LT_DROP):
            el.decompose()
        md = to_markdown(content)
    return title, md, children, attr_text


def fetch_libretexts(start_urls: list[str], source: str, author: str, license_: str,
                     out_dir: str, level_dirs: bool = False, chapters: str | None = None) -> list[str]:
    written = []
    seen: set[str] = set()
    if chapters:  # start_urls[0] — корень книги, берём только главы, подходящие под шаблон
        root = start_urls[0]
        start_urls = [c for c in lt_page(root)[2] if re.search(chapters, c[len(root):]) and c.count("/") == root.count("/") + 1]
        log(f"  главы: {[unquote(c.rsplit('/', 1)[-1]) for c in start_urls]}")
    queue = list(start_urls)
    while queue:
        url = queue.pop(0)
        if url in seen or LT_SKIP.search(url):
            continue
        seen.add(url)
        try:
            title, md, children, attr_text = lt_page(url)
        except Exception as e:  # noqa: BLE001
            log(f"  ! {url}: {e}")
            continue
        queue = [c for c in children if c not in seen] + queue  # обход в глубину, порядок книги
        if text_len(md) < 300:
            continue
        lic = license_
        m = re.search(r"shared under a\s+(CC [A-Z\-]+ \d\.\d)", attr_text)
        if m:
            lic = m.group(1)
        parts = [unquote(p) for p in urlsplit(url).path.split("/")]
        sub = ""
        if level_dirs:
            book = next((p for p in parts if re.match(r"\d+:_Book_\d", p)), "")
            sub = slug(book.split(":_")[-1]) + "/" if book else ""
        rel = f"{out_dir}/{sub}{slug(title)}.md"
        meta = {
            "title": title,
            "source": source,
            "source_url": url,
            "author": author,
            "license": lic,
            "license_url": CC_URLS.get(lic, ""),
            "attribution": attr_text or None,
            "retrieved": TODAY,
            "language": "en",
        }
        written.append(write_md(rel, {k: v for k, v in meta.items() if v}, md))
        log(f"  + {rel}")
    return written


# ---------------------------------------------------------------- CEFR-J


def fetch_cefrj() -> list[str]:
    base = "https://raw.githubusercontent.com/openlanguageprofiles/olp-en-cefrj/master/"
    out = WEB / "cefr-j" / "data"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("cefrj-vocabulary-profile-1.5.csv", "cefrj-grammar-profile-20180315.csv", "README.md"):
        (out / name).write_bytes(F.get(base + name, binary=True))
        files.append(f"cefr-j/data/{name}")
        log(f"  + cefr-j/data/{name}")
    return files + cefrj_markdown(out)


CEFRJ_TOPIC_FIX = {"Work and jobs": "Work and Jobs", "Ways of traveling": "Ways of travelling", "Art": "Arts"}
CEFRJ_LEVELS = ("A1", "A2", "B1")


def cefrj_markdown(data_dir: Path) -> list[str]:
    """Переводит таблицы CEFR-J в Markdown: слова по темам, слова по уровням, грамматический профиль."""
    import csv
    from collections import defaultdict

    meta = {
        "source": "CEFR-J Vocabulary Profile 1.5 / Grammar Profile 20180315",
        "source_url": "https://github.com/openlanguageprofiles/olp-en-cefrj",
        "author": "Yukio Tono (Tono Laboratory, Tokyo University of Foreign Studies)",
        "license": "CEFR-J terms of use: free for research and commercial use with citation",
        "license_url": "https://github.com/openlanguageprofiles/olp-en-cefrj#terms-of-use",
        "citation": "The CEFR-J Wordlist Version 1.5. Compiled by Yukio Tono, Tokyo University of Foreign "
                    "Studies. Retrieved from http://www.cefr-j.org/download.html",
        "retrieved": TODAY,
        "language": "en",
        "kind": "wordlist",
    }
    with (data_dir / "cefrj-vocabulary-profile-1.5.csv").open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["CEFR"] in CEFRJ_LEVELS]
    written = []

    topics: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for col in ("CoreInventory 1", "CoreInventory 2", "Threshold"):
            t = CEFRJ_TOPIC_FIX.get(r[col].strip(), r[col].strip())
            entry = f"{r['headword']} ({r['pos']})"
            if t and entry not in topics[t][r["CEFR"]]:
                topics[t][r["CEFR"]].append(entry)
    parts = []
    for t in sorted(topics):
        if sum(len(v) for v in topics[t].values()) < 10:
            continue
        parts.append(f"## {t}\n")
        for lvl in CEFRJ_LEVELS:
            if topics[t][lvl]:
                parts.append(f"### {t}: {lvl}\n\n" + ", ".join(sorted(topics[t][lvl], key=str.lower)) + "\n")
    written.append(write_md("cefr-j/vocabulary-by-topic.md",
                            {"title": "CEFR-J: vocabulary by topic (A1–B1)", **meta, "level": "A1–B1"}, "\n".join(parts)))

    for lvl in CEFRJ_LEVELS:
        by_pos: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            if r["CEFR"] == lvl:
                by_pos[r["pos"]].append(r["headword"])
        parts = [f"## {lvl}: {pos}\n\n" + ", ".join(sorted(set(words), key=str.lower)) + "\n"
                 for pos, words in sorted(by_pos.items())]
        written.append(write_md(f"cefr-j/vocabulary-{lvl.lower()}.md",
                                {"title": f"CEFR-J: vocabulary list, level {lvl}", **meta, "level": lvl}, "\n".join(parts)))

    with (data_dir / "cefrj-grammar-profile-20180315.csv").open(encoding="utf-8-sig") as f:
        grows = [r for r in csv.DictReader(f) if r["CEFR-J Level"][:2] in CEFRJ_LEVELS]
    grows.sort(key=lambda r: r["CEFR-J Level"])
    lines = ["| Grammatical item | Sentence type | CEFR-J level |", "|---|---|---|"]
    lines += [f"| {r['Grammatical Item']} | {r['Sentence Type'] or '—'} | {r['CEFR-J Level']} |" for r in grows]
    gmeta = {**meta, "kind": "profile",
             "citation": "The CEFR-J Grammar Profile Version 20180315. Retrieved from http://www.cefr-j.org/download.html"}
    written.append(write_md("cefr-j/grammar-profile.md",
                            {"title": "CEFR-J: grammar profile (items at A1–B1)", **gmeta, "level": "A1–B1"}, "\n".join(lines)))
    for rel in written:
        log(f"  + {rel}")
    return written


# ---------------------------------------------------------------- источники

LT_ESL = "https://human.libretexts.org/Bookshelves/Languages/English_as_a_Second_Language/"
LT_BISS = LT_ESL + "ESL_Grammar_The_Way_You_Like_It_(Bissonnette)/"
LT_WFS = "https://human.libretexts.org/Bookshelves/Composition/Introductory_Composition/Writing_for_Success_(McLean)/"

WIKIPEDIA_EN = [
    "English grammar", "English verbs", "Uses of English verb forms", "English modal auxiliary verbs",
    "English auxiliary verbs", "English conditional sentences", "English articles", "English passive voice",
    "English plurals", "English possessive", "English personal pronouns", "English relative clauses",
    "English phrasal verbs", "English irregular verbs", "Going-to future", "Capitalization in English",
    "English nouns", "English adjectives", "English adverbs", "English prepositions", "English determiners",
    "English interrogative words", "Do-support", "Present perfect", "Simple past", "Simple present",
    "Present continuous", "Comparison (grammar)", "Count noun", "Mass noun", "Tag question",
    "Indirect speech", "Shall and will", "Gerund", "English numerals", "English clause syntax",
    "English collocations", "English prefix",
]
WIKIPEDIA_RU = [
    "Грамматика английского языка", "Present simple", "Present continuous", "Present perfect",
    "Past simple", "Future simple",
]

SOURCES = {
    "wikibooks_ru": lambda: fetch_wikibook(
        "ru.wikibooks.org", "Самоучитель английского языка", "Викиучебник: Английский язык", "ru",
        "wikibooks-ru-english", index="Английский язык"),
    "wikibooks_in_use": lambda: fetch_wikibook(
        "en.wikibooks.org", "English in Use", "Wikibooks: English in Use", "en", "wikibooks-en-english-in-use",
        skip=r"About$|External Resources|Restructure|Print version|Q&A|Hard C"),
    "wikibooks_grammar": lambda: fetch_wikibook(
        "en.wikibooks.org", "English Grammar", "Wikibooks: English Grammar", "en", "wikibooks-en-english-grammar",
        skip=r"Appendix|Test$|Review$"),
    "wikipedia_en": lambda: fetch_wikipedia("en.wikipedia.org", WIKIPEDIA_EN, "en", "wikipedia-en"),
    "wikipedia_ru": lambda: fetch_wikipedia("ru.wikipedia.org", WIKIPEDIA_RU, "ru", "wikipedia-ru"),
    "bissonnette": lambda: fetch_libretexts(
        [LT_BISS + f"0{n}%3A_Book_{n}" for n in (1, 2, 3, 4)],
        "ESL Grammar: The Way You Like It (LibreTexts)", "Don Bissonnette", "CC BY-NC 4.0",
        "libretexts-bissonnette-esl-grammar", level_dirs=True),
    "explorations": lambda: fetch_libretexts(
        [LT_ESL + "Explorations_1%3A_Grammar_for_the_Experienced_Beginner"],
        "Explorations 1: Grammar for the Experienced Beginner (LibreTexts)",
        "Susan Caisse, Kristee Emens-Hesslink, Jennifer Rueda", "CC BY-NC-SA 4.0", "libretexts-explorations-1"),
    "writing_for_success": lambda: fetch_libretexts(
        [LT_WFS.rstrip("/")],
        "Writing for Success (LibreTexts)", "Scott McLean; publisher: University of Minnesota Libraries",
        "CC BY-NC-SA 3.0", "libretexts-writing-for-success", chapters=r"^/0?[2356](%3A|:)"),
    "cefrj": fetch_cefrj,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="список источников через запятую")
    args = ap.parse_args()
    names = [n for n in args.only.split(",") if n] or list(SOURCES)
    total = 0
    for name in names:
        log(f"== {name}")
        try:
            files = SOURCES[name]()
        except Exception as e:  # noqa: BLE001
            log(f"  !! источник пропущен: {e}")
            continue
        total += len(files)
        log(f"   файлов: {len(files)}")
    log(f"Готово: {total} файлов, сетевых запросов: {F.network_requests}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
