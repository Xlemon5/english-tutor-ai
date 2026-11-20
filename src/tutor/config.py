from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    materials_dir: Path = Path("materials")
    data_dir: Path = Path("data")

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "http://localhost:8080/v1"  # для LLM_PROVIDER=openai (llama.cpp, vLLM, LM Studio)
    openai_api_key: str = "not-needed"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096
    llm_max_tokens: int = 700
    llm_keep_alive: str = "30m"
    llm_timeout: float = 300.0  # секунд на один вызов модели: зависшая Ollama не держит запрос вечно
    # генерация тестов: сэмплинг вместо жадного выбора. При T=0 Qwen на части тем стабильно пишет объяснения
    # по-китайски и в каждом раунде повторяет тот же брак (воспроизведено на Present Simple / A1).
    # Отправная точка — рекомендованные параметры Qwen2.5.
    quiz_temperature: float = 0.7
    quiz_top_p: float = 0.8
    quiz_top_k: int = 20
    quiz_repeat_penalty: float = 1.05
    quiz_max_tokens: int = 1500
    # пример текста по просьбе студента: при T=0 на одну и ту же просьбу был бы всегда один и тот же текст
    compose_temperature: float = 0.7  # предохранитель: обычный ответ на 4 вопроса — 250–450 токенов
    # отладка: каждый вызов LLM в JSON-сценариях (запрос, сырой ответ, токены) пишется в debug_llm_dir
    debug_llm: bool = False
    debug_llm_dir: Path = Path("data/llm_debug")

    embeddings_provider: str = "ollama"
    embeddings_model: str = "bge-m3"

    chroma_host: str = ""
    chroma_port: int = 8001
    chroma_collection: str = "materials"

    chunk_size: int = 1200
    chunk_overlap: int = 150
    retrieval_top_k: int = 4
    # сколько символов каждого фрагмента уходит в промпт: меньше — быстрее первый токен
    context_fragment_chars: int = 650
    retrieval_candidates: int = 20
    mmr_lambda: float = 0.7
    # Вес лексической выдачи (BM25) при объединении рангов; семантическая выдача имеет вес 1
    lexical_weight: float = 0.5
    relevance_threshold: float = 0.55
    # тесты: True — фрагменты материалов идут в промпт, False — модель составляет предложения сама
    # по умолчанию — без фрагментов: качество вопросов то же (~48% без замечаний на 82 размеченных вопросах),
    # но тест в 2,2 раза быстрее и чаще полный (6 из 9 тем против 4); поиск всё равно проверяет тему и даёт ссылку
    quiz_use_fragments: bool = False

    # Авторизация: при AUTH_ENABLED=false все запросы идут от пользователя «local»
    auth_enabled: bool = True
    allow_registration: bool = True
    admin_username: str = ""
    admin_password: str = ""

    api_url: str = "http://localhost:8000"  # адрес API для веб-интерфейса

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tutor.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()
