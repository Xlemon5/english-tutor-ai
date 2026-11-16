.PHONY: install manifest ingest ingest-full test lint ask api web eval eval-fast up down

install:        ## установить зависимости в .venv
	uv sync

manifest:       ## пересобрать materials/sources.yaml и отчёт о покрытии
	uv run tutor manifest

ingest:         ## проиндексировать materials/ (только изменившиеся файлы)
	uv run tutor ingest

ingest-full:    ## переиндексировать всё с нуля
	uv run tutor ingest --full

test:           ## pytest (модели не нужны)
	uv run pytest

lint:
	uv run ruff check src tests web eval

ask:            ## make ask Q="Когда используется Present Perfect?"
	uv run tutor ask "$(Q)"

api:            ## API на http://localhost:8000
	uv run uvicorn tutor.api.app:app --host 0.0.0.0 --port 8000

web:            ## веб-интерфейс на http://localhost:8501
	uv run streamlit run web/app.py --server.port 8501 --browser.gatherUsageStats false

eval:           ## оценка RAG: поиск + ответы LLM
	uv run python eval/run_eval.py

eval-fast:      ## оценка RAG без LLM (только поиск и порог)
	uv run python eval/run_eval.py --no-llm

up:             ## поднять проект в Docker (Ollama должна работать на хосте)
	docker compose up --build

down:
	docker compose down
