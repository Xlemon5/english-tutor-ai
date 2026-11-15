"""
REST API для English Tutor AI
Соответствует требованию F4 из презентации преподавателя.

Запуск: python api.py
Документация Swagger: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from rag_agent import RAGAgent

# Создаём приложение
app = FastAPI(
    title="English Tutor AI - REST API",
    description="API для интеллектуального агента по изучению английского языка",
    version="1.0.0"
)

# Инициализируем агента (он сам прочитает все PDF из папок)
agent = RAGAgent()

# Если база пустая — индексируем документы при старте
if agent.stats()["documents"] == 0:
    print("📥 Индексирую документы при старте API...")
    agent.index_all_documents()


# === Модели запросов (Pydantic) ===

class QuestionRequest(BaseModel):
    question: str
    username: Optional[str] = "Ученик"


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class QuizRequest(BaseModel):
    topic: str


# === API Endpoints ===

@app.get("/")
async def root():
    """Проверка работоспособности API"""
    return {
        "status": "ok",
        "message": "English Tutor AI API работает",
        "documents_indexed": agent.stats()["documents"]
    }


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    F4: Приём вопроса и получение ответа от RAG-агента

    Пример запроса:
    {
        "question": "Что такое Present Simple?",
        "username": "Student"
    }
    """
    try:
        answer = agent.ask_tutor(request.question, request.username)
        return {
            "question": request.question,
            "answer": answer,
            "username": request.username
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: SearchRequest):
    """
    F5: Семантический + лексический поиск фрагментов

    Пример запроса:
    {
        "query": "irregular verbs",
        "top_k": 5
    }
    """
    try:
        result = agent.hybrid_search(request.query)
        return {
            "query": request.query,
            "results": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/quiz")
async def generate_quiz(request: QuizRequest):
    """
    Генерация теста по теме

    Пример запроса:
    {
        "topic": "Grammar"
    }
    """
    try:
        quiz = agent.generate_quiz(request.topic)
        return {
            "topic": request.topic,
            "quiz": quiz
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
async def reindex_all():
    """
    F1-F3: Переиндексация всех PDF из папок data/documents/*
    """
    try:
        info = agent.index_all_documents()
        return {
            "status": "success",
            "files": info["files"],
            "chunks": info["chunks"],
            "report": info["report"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === Запуск сервера ===

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🚀 Запуск REST API English Tutor AI")
    print("=" * 60)
    print("📡 API: http://localhost:8000")
    print("📚 Swagger документация: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8000)