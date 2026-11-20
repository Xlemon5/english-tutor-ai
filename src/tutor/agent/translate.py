"""Сценарий «перевод»: слово, фраза или короткое предложение — с русского на английский и обратно.

Перевод делает модель: словаря с русскими переводами в материалах нет, а отказ «в материалах этого нет»
на «как будет кошка по-английски» студенту бесполезен. Из библиотеки берётся то, что в ней есть:
уровень английского слова по спискам CEFR-J — он показывается под ответом.
"""
from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, Field

from tutor.agent.theory import TheoryAgent
from tutor.lexicon import word_levels
from tutor.llm.structured import ContextBudgetError, estimate_tokens
from tutor.prompts import load_prompt

TRANSLATE_RULE = "Объяснения пиши по-русски, английские слова и примеры — по-английски. Китайский язык и иероглифы не используй никогда."
MAX_WORDS = 4  # уровень показываем для первых вариантов перевода, а не для всех слов из примеров
POS_RU = {"noun": "существительное", "verb": "глагол", "adjective": "прилагательное", "adverb": "наречие",
          "preposition": "предлог", "pronoun": "местоимение", "conjunction": "союз", "determiner": "определитель",
          "interjection": "междометие", "number": "числительное", "modal auxiliary": "модальный глагол"}
MODE_WORD = "Это слово или короткая фраза: переведи по правилам 1–3, с примерами."
MODE_SENTENCE = ("Это предложение: дай только перевод одной строкой (английский текст — **жирным**) и, если нужно, "
                 "одну строку пояснения по-русски. Часть речи и примеры не пиши.")
# служебные слова просьбы: что остаётся после них — то, что нужно перевести
_SERVICE = re.compile(
    r"\b(переведи\w*|перевед\w*|перевести|перевод\w*|translate\w*|как|будет|сказать|говорят|называется|назвать|"
    r"пишется|произносится|что|значит|означает|обозначает|слово|слова|фраза|фразу|выражение|пожалуйста|мне|"
    r"по[- ]?английск\w*|по[- ]?русск\w*|на|английск\w*|русск\w*|in|english|russian|what|does|do|you|I|say|mean|is|"
    r"how|to|please)\b", re.IGNORECASE)
_BOLD = re.compile(r"\*\*([A-Za-z][A-Za-z' -]*?)\*\*")
_MARK = re.compile(r"[ \t]*\[\d+\]")


class WordLevel(BaseModel):
    word: str
    level: str
    pos: str = ""


class TranslateAnswer(BaseModel):
    status: Literal["answered"] = "answered"
    answer: str
    levels: list[WordLevel] = Field(default_factory=list)  # уровни английских слов по CEFR-J из библиотеки


class TranslateEvent(BaseModel):
    type: Literal["token", "final"]
    text: str = ""
    result: TranslateAnswer | None = None


_HEAD_LINE = re.compile(r"^(\*\*([A-Za-z][A-Za-z' -]*?)\*\*\s*[—–-].*?)[ \t]*\(([^()\n]*)\)[ \t]*$", re.MULTILINE)


def fix_parts_of_speech(answer: str, materials_dir: str) -> str:
    """Часть речи в строке перевода берётся из CEFR-J, если там у слова она одна: модель называла however
    союзом, а под ответом по списку стояло «наречие» — ответ противоречил сам себе."""
    def repl(m: re.Match) -> str:
        parts = {pos for _, pos in word_levels(m.group(2), materials_dir)}
        if len(parts) != 1:
            return m.group(0)
        return f"{m.group(1)} ({POS_RU.get(parts.pop(), m.group(3))})"
    return _HEAD_LINE.sub(repl, answer)


def is_sentence(request: str) -> bool:
    """Просят перевести предложение (три слова и больше), а не слово: тогда без части речи и примеров.
    Текст после двоеточия или в кавычках — это и есть то, что переводят."""
    if ":" in request:
        payload = request.split(":", 1)[1]
    elif quoted := re.search(r"[«\"“]([^»\"”]+)[»\"”]", request):
        payload = quoted.group(1)
    else:
        payload = _SERVICE.sub(" ", request)
    return len(re.findall(r"[A-Za-zА-Яа-яЁё']+", payload)) >= 3


def cefr_levels(answer: str, request: str, materials_dir: str) -> list[WordLevel]:
    """Уровни английских слов из перевода (выделены жирным) и из самой просьбы, если она на английском."""
    candidates = [m.group(1).strip() for m in _BOLD.finditer(answer)]
    candidates += re.findall(r"[«\"'“]([A-Za-z][A-Za-z' -]*)[»\"'”]", request)
    levels: list[WordLevel] = []
    seen: set[str] = set()
    for word in candidates:
        if word.lower() in seen or len(word.split()) > 3:
            continue
        seen.add(word.lower())
        if found := word_levels(word, materials_dir):
            level, pos = found[0]
            levels.append(WordLevel(word=word, level=level, pos=POS_RU.get(pos, pos)))
        if len(levels) >= MAX_WORDS:
            break
    return levels


class TranslateAgent(TheoryAgent):
    """Берёт у TheoryAgent потоковую генерацию с защитой от перехода на китайский; промпт свой."""
    user_prompt = "translate_user"
    resume_rule = TRANSLATE_RULE

    def stream(self, request: str) -> Iterator[TranslateEvent]:
        s = self.retriever.settings
        system, template = load_prompt("translate_system"), load_prompt(self.user_prompt)
        if estimate_tokens(system, template, "", request) + s.llm_max_tokens > s.llm_num_ctx:
            raise ContextBudgetError("Текст для перевода не помещается в окно модели — сократите его.")
        text = ""
        sentence = is_sentence(request)
        inputs = {"system": system, "question": request, "mode": MODE_SENTENCE if sentence else MODE_WORD}
        for piece in self._pieces(inputs):
            text += piece
            yield TranslateEvent(type="token", text=piece)
        answer = re.sub(r"\n{3,}", "\n\n", _MARK.sub("", text)).strip()
        if not sentence:
            answer = fix_parts_of_speech(answer, str(s.materials_dir))
        # уровень по CEFR-J есть у слов, а не у предложений
        levels = [] if sentence else cefr_levels(answer, request, str(s.materials_dir))
        yield TranslateEvent(type="final", result=TranslateAnswer(answer=answer, levels=levels))

    def translate(self, request: str) -> TranslateAnswer:
        final = None
        for event in self.stream(request):
            if event.type == "final":
                final = event.result
        assert final is not None
        return final
