# Один образ для API и веб-интерфейса: команда запуска задаётся в docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Зависимости ставятся отдельным слоем: он пересобирается, только когда меняется uv.lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY web ./web
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501
CMD ["uvicorn", "tutor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
