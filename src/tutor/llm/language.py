"""Защита от перехода модели на китайский.

Qwen обучена в основном на китайских и английских текстах. Когда модель не уверена в следующем
русском слове, самым вероятным продолжением иногда оказывается китайский токен, и дальше она
продолжает по-китайски. Чаще это бывает при температуре выше 0, но и при 0 не исключено.
"""
from __future__ import annotations

import re

# иероглифы, кана, хангыль, китайская пунктуация и полноширинные знаки; кириллицу и латиницу не задевает
FOREIGN_SCRIPT = re.compile(r"[⺀-鿿ꥠ-꥿가-힯豈-﫿︰-﹏＀-￯]")
LANGUAGE_RULE = ("Пиши только по-русски, английские примеры и термины — по-английски. "
                 "Китайский язык и иероглифы не используй никогда.")
FOREIGN_SCRIPT_ERROR = "в тексте есть китайские иероглифы — пиши только по-русски и по-английски"


def has_foreign_script(value: object) -> bool:
    """Есть ли иероглифы в строке или в любой строке внутри словаря или списка."""
    if isinstance(value, str):
        return bool(FOREIGN_SCRIPT.search(value))
    if isinstance(value, dict):
        return any(has_foreign_script(v) for v in value.values())
    if isinstance(value, list | tuple):
        return any(has_foreign_script(v) for v in value)
    return False
