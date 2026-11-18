"""FastAPI поверх ядра агента. Ядро про HTTP ничего не знает — здесь только транспорт и история."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator

import httpx
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from tutor.agent.plan import build_plan, plan_summary
from tutor.agent.quiz import TYPE_RU, Quiz
from tutor.agent.router import route
from tutor.llm.structured import ContextBudgetError, StructuredOutputError
from tutor.models import LEVELS
from tutor.services import Services, get_services
from tutor.storage import Storage

SMALLTALK = ("Здравствуйте! Я репетитор по английскому. Задайте вопрос по грамматике или лексике, "
             "пришлите текст на проверку, попросите тест по теме или программу обучения до нужного уровня.")


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[\w.-]+$")
    password: str = Field(min_length=6, max_length=200)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    level: str | None = None


class ReviewRequest(BaseModel):
    text: str = Field(min_length=1)
    task: str = ""


class QuizRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=200)
    level: str = "A2"
    count: int = Field(5, ge=1, le=15)
    types: list[str] = Field(default_factory=lambda: list(TYPE_RU))
    allow_without_materials: bool = False


class QuizAnswers(BaseModel):
    answers: dict[int, str]


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


MODEL_DOWN = "Модель недоступна: не удаётся связаться с Ollama. Проверьте, что она запущена (ollama serve)."
MODEL_TIMEOUT = "Модель не ответила вовремя. Попробуйте ещё раз: после простоя первый запрос загружает модель."
MODEL_FAILED = "Модель не смогла дать корректный ответ. Попробуйте ещё раз или переформулируйте запрос."


def user_message(e: Exception) -> str:
    """Текст ошибки для пользователя: технические подробности (разбор JSON, схемы) остаются в журнале."""
    if isinstance(e, ConnectionError | httpx.ConnectError):
        return MODEL_DOWN
    if isinstance(e, httpx.TimeoutException):
        return MODEL_TIMEOUT
    if isinstance(e, StructuredOutputError) and not isinstance(e, ContextBudgetError) \
            and str(e).startswith("LLM не вернула"):
        logger.warning("ответ модели не прошёл проверку: {}", e)
        return MODEL_FAILED
    return str(e)


def public_quiz(quiz: Quiz) -> dict:
    """Тест для ученика: без правильных ответов и объяснений — они приходят только после проверки."""
    return quiz.model_dump(exclude={"questions": {"__all__": {"answer", "explanation"}}})


def create_app(services: Services | None = None, storage: Storage | None = None) -> FastAPI:
    app = FastAPI(title="English Tutor API", version="0.1.0")

    @app.exception_handler(ConnectionError)
    @app.exception_handler(httpx.ConnectError)
    def model_down(request: Request, e: Exception) -> JSONResponse:
        logger.error("нет связи с моделью: {}", e)
        return JSONResponse(status_code=503, content={"detail": MODEL_DOWN})

    @app.exception_handler(httpx.TimeoutException)
    def model_timeout(request: Request, e: Exception) -> JSONResponse:
        logger.error("модель не ответила вовремя: {}", e)
        return JSONResponse(status_code=504, content={"detail": MODEL_TIMEOUT})
    state: dict = {"services": services, "storage": storage}

    def svc() -> Services:
        if state["services"] is None:
            state["services"] = get_services()
        return state["services"]

    def db() -> Storage:
        if state["storage"] is None:
            s = svc().settings
            state["storage"] = Storage(s.db_path)
            if s.admin_username and s.admin_password:
                state["storage"].create_user(s.admin_username, s.admin_password)
        return state["storage"]

    def current_user(authorization: str | None = Header(None)) -> str:
        if not svc().settings.auth_enabled:
            return "local"
        token = (authorization or "").removeprefix("Bearer ").strip()
        user = db().session_user(token) if token else None
        if not user:
            raise HTTPException(401, "Требуется вход")
        return user

    def check_level(level: str | None) -> str | None:
        if level and level not in LEVELS:
            raise HTTPException(422, f"Уровень должен быть одним из: {', '.join(LEVELS)}")
        return level or None

    # ------------------------------------------------------------ служебное
    @app.get("/api/health")
    def health() -> dict:
        s = svc()
        return {"status": "ok", "chunks": s.store.count(), "llm": s.settings.llm_model,
                "embeddings": s.settings.embeddings_model, "auth": s.settings.auth_enabled,
                "registration": s.settings.allow_registration}

    # ------------------------------------------------------------ авторизация
    @app.post("/api/auth/register")
    def register(body: Credentials) -> dict:
        if not svc().settings.allow_registration:
            raise HTTPException(403, "Регистрация отключена")
        if not db().create_user(body.username, body.password):
            raise HTTPException(409, "Такой пользователь уже есть")
        return {"token": db().create_session(body.username), "username": body.username}

    @app.post("/api/auth/login")
    def login(body: Credentials) -> dict:
        if not db().verify_user(body.username, body.password):
            raise HTTPException(401, "Неверный логин или пароль")
        return {"token": db().create_session(body.username), "username": body.username}

    @app.post("/api/auth/logout")
    def logout(authorization: str | None = Header(None)) -> dict:
        db().delete_session((authorization or "").removeprefix("Bearer ").strip())
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(user: str = Depends(current_user)) -> dict:
        return {"username": user}

    # ------------------------------------------------------------ чат (SSE)
    @app.post("/api/chat")
    def chat(body: ChatRequest, user: str = Depends(current_user)) -> StreamingResponse:
        level = check_level(body.level)
        r = route(body.message)
        db().add_message(user, "user", body.message)

        def events() -> Iterator[str]:
            yield sse("intent", {"intent": r.intent})
            try:
                if r.intent == "smalltalk":
                    payload = {"kind": "smalltalk", "answer": SMALLTALK}
                elif r.intent == "plan":
                    plan = build_plan(svc().settings.materials_dir, r.level or level, r.start_level)
                    payload = {"kind": "plan", "answer": plan_summary(plan), **plan.model_dump()}
                elif r.intent == "review":
                    result = svc().review.review(r.text).model_dump()
                    db().add_review(user, result)
                    payload = {"kind": "review", **result}
                elif r.intent == "quiz":
                    quiz = svc().quiz.generate(r.topic or body.message, r.level or level or "A2", r.count or 5)
                    if quiz.status == "ready":
                        db().add_quiz(user, quiz.model_dump())
                    payload = {"kind": "quiz", **public_quiz(quiz)}
                else:
                    payload = {}
                    for event in svc().theory.stream(body.message, level):
                        if event.type == "token":
                            yield sse("token", {"text": event.text})
                        else:
                            payload = {"kind": "theory", **event.result.model_dump()}
            except (StructuredOutputError, ValueError) as e:
                yield sse("error", {"message": user_message(e)})
                return
            except Exception as e:  # noqa: BLE001
                logger.exception("ошибка чата")
                down = isinstance(e, ConnectionError | httpx.ConnectError | httpx.TimeoutException)
                yield sse("error", {"message": user_message(e) if down else f"Не удалось получить ответ модели: {e}"})
                return
            db().add_message(user, "assistant", payload.get("answer") or payload.get("summary") or
                             payload.get("message") or "", payload)
            yield sse("final", payload)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/chat/history")
    def chat_history(user: str = Depends(current_user)) -> list[dict]:
        return db().messages(user)

    @app.delete("/api/chat/history")
    def clear_chat(user: str = Depends(current_user)) -> dict:
        db().clear_messages(user)
        return {"ok": True}

    # ------------------------------------------------------------ теория без стрима (CLI, eval, тесты)
    @app.post("/api/theory/ask")
    def ask(body: ChatRequest, user: str = Depends(current_user)) -> dict:
        try:
            return svc().theory.answer(body.message, check_level(body.level)).model_dump()
        except StructuredOutputError as e:
            raise HTTPException(502, user_message(e)) from e

    # ------------------------------------------------------------ проверка работ
    @app.post("/api/review")
    def review(body: ReviewRequest, user: str = Depends(current_user)) -> dict:
        try:
            result = svc().review.review(body.text, body.task).model_dump()
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        except StructuredOutputError as e:
            raise HTTPException(502, user_message(e)) from e
        result["id"] = db().add_review(user, result)
        return result

    # ------------------------------------------------------------ тесты
    @app.post("/api/quiz/generate")
    def quiz_generate(body: QuizRequest, user: str = Depends(current_user)) -> dict:
        level = check_level(body.level) or "A2"
        unknown = [t for t in body.types if t not in TYPE_RU]
        if unknown or not body.types:
            raise HTTPException(422, f"Типы заданий: {', '.join(TYPE_RU)}")
        try:
            quiz = svc().quiz.generate(body.topic, level, body.count, body.types, body.allow_without_materials)
        except StructuredOutputError as e:
            raise HTTPException(502, user_message(e)) from e
        if quiz.status == "ready":
            db().add_quiz(user, quiz.model_dump())  # в базе — целиком, ответы нужны для проверки
        return public_quiz(quiz)

    @app.post("/api/quiz/{quiz_id}/check")
    def quiz_check(quiz_id: str, body: QuizAnswers, user: str = Depends(current_user)) -> dict:
        stored = db().get_quiz(user, quiz_id)
        if not stored:
            raise HTTPException(404, "Тест не найден")
        result = svc().quiz.check(Quiz.model_validate(stored), body.answers).model_dump()
        result["id"] = db().add_attempt(user, result)
        return result

    # ------------------------------------------------------------ программа обучения
    @app.get("/api/plan")
    def plan(target: str = "B1", start: str = "A1", user: str = Depends(current_user)) -> dict:
        try:
            result = build_plan(svc().settings.materials_dir, target, start)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return {"answer": plan_summary(result), **result.model_dump()}

    # ------------------------------------------------------------ история
    @app.get("/api/history")
    def history(user: str = Depends(current_user)) -> dict:
        return {"reviews": db().reviews(user), "attempts": db().attempts(user), "topics": db().topic_stats(user)}

    # ------------------------------------------------------------ библиотека
    @app.get("/api/library")
    def library(user: str = Depends(current_user)) -> dict:
        s = svc().settings
        path = s.materials_dir / "sources.yaml"
        manifest = yaml.safe_load(path.read_text("utf-8")) if path.exists() else {}
        chunks: dict[str, int] = {}
        if s.db_path.exists():
            conn = sqlite3.connect(s.db_path)
            try:
                chunks = dict(conn.execute("SELECT file, chunks FROM indexed_files"))
            except sqlite3.OperationalError:
                pass
            finally:
                conn.close()
        files = [{**f, "chunks": chunks.get(f["file"], 0), "indexed": f["file"] in chunks}
                 for f in manifest.get("files", [])]
        return {"generated": manifest.get("generated"), "collections": manifest.get("collections", []),
                "files": files, "indexed_files": len(chunks), "chunks": sum(chunks.values())}

    @app.get("/api/chunks/{chunk_id}")
    def chunk(chunk_id: str, user: str = Depends(current_user)) -> dict:
        found = svc().store.get([chunk_id])
        if not found:
            raise HTTPException(404, "Фрагмент не найден")
        return found[0][0].model_dump()

    return app


app = create_app()
