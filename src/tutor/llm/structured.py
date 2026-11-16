"""Структурированный вывод LLM: JSON по pydantic-схеме, валидация и повтор при невалидном ответе."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Модель не смогла вернуть валидный JSON за отведённое число попыток."""


def extract_json(text: str) -> str:
    """Достаёт JSON-объект из ответа: модели иногда оборачивают его в ```json или добавляют пояснения."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


class StructuredCaller:
    """make_llm(schema) возвращает чат-модель, настроенную на JSON-схему (или обычную, если провайдер не умеет)."""

    def __init__(self, make_llm: Callable[[dict], object], retries: int = 2) -> None:
        self.make_llm = make_llm
        self.retries = retries

    def __call__(self, schema: type[T], system: str, user: str) -> T:
        llm = self.make_llm(schema.model_json_schema())
        messages = [SystemMessage(system), HumanMessage(user)]
        last_error = ""
        for attempt in range(self.retries + 1):
            raw = llm.invoke(messages).content  # type: ignore[attr-defined]
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
