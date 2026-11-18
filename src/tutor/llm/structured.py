"""Структурированный вывод LLM: JSON по pydantic-схеме, валидация и повтор при невалидном ответе."""
from __future__ import annotations

import itertools
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Модель не смогла вернуть валидный JSON за отведённое число попыток."""


# Бюджет окна. Если промпт плюс максимальный ответ не помещаются в num_ctx, Ollama не падает: во время генерации
# она сдвигает контекст и выбрасывает его начало — системный промпт (n_keep = 4). Ответ придёт, но уже без
# инструкций. Поэтому длину проверяем до вызова. Точно посчитать токены заранее нельзя (в Ollama 0.40 нет
# /api/tokenize), оценка — по символам с запасом: на промптах проекта ~3,7 символа на токен (5674 → 1541 по
# счётчику Ollama), берём 3,0, то есть оценка завышена примерно на 20 %.
CHARS_PER_TOKEN = 3.0
MESSAGE_OVERHEAD_TOKENS = 8  # разметка шаблона чата: <|im_start|>роль … <|im_end|>


class ContextBudgetError(StructuredOutputError):
    """Промпт вместе с максимальной длиной ответа не помещается в окно модели."""


def estimate_tokens(*texts: str) -> int:
    return sum(int(len(t) / CHARS_PER_TOKEN) + MESSAGE_OVERHEAD_TOKENS for t in texts)


def check_budget(prompt_tokens: int, num_predict: int | None, num_ctx: int | None) -> None:
    if num_ctx and num_predict and prompt_tokens + num_predict > num_ctx:
        raise ContextBudgetError(
            f"Запрос не помещается в окно модели: ~{prompt_tokens} токенов промпта и до {num_predict} на ответ "
            f"при окне {num_ctx}.")


def extract_json(text: str) -> str:
    """Достаёт JSON-объект из ответа: модели иногда оборачивают его в ```json или добавляют пояснения."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


_seq = itertools.count(1)


def dump_llm_call(directory: Path, llm, messages: list, response, seconds: float) -> None:
    """Отладка (DEBUG_LLM): запрос ровно в том виде, в каком он уходит в Ollama (/api/chat, роли, options,
    format), сырой ответ до разбора и счётчики токенов из ответа. Один вызов — один JSON-файл."""
    try:
        if hasattr(llm, "_chat_params"):  # ChatOllama: тот же метод собирает тело запроса к /api/chat
            request = llm._chat_params(messages)
            request["messages"] = [dict(m) for m in request["messages"]]
            endpoint = f"{(getattr(llm, 'base_url', None) or 'http://localhost:11434').rstrip('/')}/api/chat"
        else:  # другие провайдеры: хотя бы сообщения с ролями
            request = {"messages": [{"role": m.type, "content": m.content} for m in messages]}
            endpoint = type(llm).__name__
        meta = dict(getattr(response, "response_metadata", None) or {})
        record = {"endpoint": endpoint, "request": request, "raw_response": getattr(response, "content", ""),
                  "response_metadata": meta, "seconds": round(seconds, 2)}
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{next(_seq):04d}.json"
        (directory / name).write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), "utf-8")
    except Exception as e:  # noqa: BLE001 — отладочная запись не должна ломать ответ
        logger.warning("DEBUG_LLM: не удалось записать вызов: {}", e)


class StructuredCaller:
    """make_llm(schema) возвращает чат-модель, настроенную на JSON-схему (или обычную, если провайдер не умеет).
    debug_dir — каталог для записи каждого вызова (включается настройкой DEBUG_LLM)."""

    def __init__(self, make_llm: Callable[[dict], object], retries: int = 2, debug_dir: Path | None = None) -> None:
        self.make_llm = make_llm
        self.retries = retries
        self.debug_dir = debug_dir

    def __call__(self, schema: type[T], system: str, user: str, **options) -> T:
        """options — параметры этого вызова для фабрики модели, например seed."""
        # у модели может быть своя, более строгая схема для декодирования (например, с разрешёнными символами);
        # проверка ответа всё равно идёт по pydantic-модели
        fmt = schema.llm_schema() if hasattr(schema, "llm_schema") else schema.model_json_schema()
        llm = self.make_llm(fmt, **options)
        messages = [SystemMessage(system), HumanMessage(user)]
        last_error = ""
        num_predict, num_ctx = getattr(llm, "num_predict", None), getattr(llm, "num_ctx", None)
        for attempt in range(self.retries + 1):
            if attempt == 0:  # исходный запрос обязан помещаться — иначе понятная ошибка
                check_budget(estimate_tokens(system, user), num_predict, num_ctx)
            elif num_ctx and num_predict and estimate_tokens(*(str(m.content) for m in messages)) + num_predict > num_ctx:
                # повтор не должен превращаться в отказ из-за собственной истории: оставляем только причину отказа
                messages = [messages[0], messages[1], messages[-1]]
            start = time.time()
            response = llm.invoke(messages)  # type: ignore[attr-defined]
            if self.debug_dir is not None:
                dump_llm_call(self.debug_dir, llm, messages, response, time.time() - start)
            raw = response.content
            meta = getattr(response, "response_metadata", None) or {}
            # ответ оборван лимитом длины — это ошибка, не ответ (done_reason у Ollama, finish_reason у OpenAI)
            if (meta.get("done_reason") or meta.get("finish_reason")) == "length":
                last_error = f"ответ оборвался на лимите длины ({meta.get('eval_count')} токенов)"
                logger.warning("LLM: {} (попытка {})", last_error, attempt + 1)
                messages = [*messages, HumanMessage(
                    "Ответ оборвался на лимите длины. Верни тот же JSON короче: объяснения — одно предложение.")]
                continue
            try:
                return schema.model_validate(json.loads(extract_json(raw)))
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = str(e)
                logger.warning("невалидный ответ LLM (попытка {}): {}", attempt + 1, last_error[:300])
                messages += [
                    AIMessage(raw),
                    HumanMessage("Ответ не прошёл проверку схемы:\n" + last_error[:800]
                                 + "\nВерни только исправленный JSON, без пояснений."),
                ]
        raise StructuredOutputError(f"LLM не вернула валидный JSON: {last_error[:300]}")
