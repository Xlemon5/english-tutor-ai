from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tutor.agent.quiz import QuizAgent
from tutor.agent.review import ReviewAgent
from tutor.agent.router import route
from tutor.agent.theory import TheoryAgent
from tutor.api.app import create_app
from tutor.ingestion.indexer import ingest
from tutor.llm.structured import StructuredCaller
from tutor.retrieval.hybrid import Retriever
from tutor.services import Services
from tutor.storage import Storage

REVIEW = {
    "errors": [{"fragment": "brother work", "correction": "brother works", "type": "grammar",
                "explanation": "Нужен -s.", "rule": "third person singular ending -s"}],
    "summary": "Хороший текст, одна ошибка.", "cefr_level": "A1", "review_topics": ["Present Simple"],
}
QUIZ = {"questions": [{"type": "multiple_choice", "prompt": "She ____ in a shop.",
                       "options": ["work", "works", "working"], "answer": "works",
                       "explanation": "После she нужен -s.", "fragment": 1}]}
TOPIC = "present simple third person singular ending -s"


def make_client(settings, embedder, store, tmp_path, *, theory="Окончание -s нужно после he и she [1].",
                structured=()) -> TestClient:
    ingest(settings, embedder, store)
    retriever = Retriever(store, embedder, settings)
    llm = FakeListChatModel(responses=[json.dumps(r) for r in structured] or ["{}"])
    caller = StructuredCaller(lambda schema, **_: llm, retries=0)
    services = Services(settings=settings, store=store, retriever=retriever,
                        theory=TheoryAgent(retriever, FakeListChatModel(responses=[theory])),
                        review=ReviewAgent(retriever, caller), quiz=QuizAgent(retriever, caller))
    return TestClient(create_app(services, Storage(tmp_path / "api.db")))


@pytest.fixture
def auth() -> dict:
    return {}


def login(client: TestClient, name: str = "student") -> dict:
    token = client.post("/api/auth/register", json={"username": name, "password": "secret123"}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def sse_events(response) -> list[tuple[str, dict]]:
    events = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def test_auth_is_required_and_works(settings, embedder, store, tmp_path):
    client = make_client(settings, embedder, store, tmp_path)
    assert client.get("/api/health").json()["chunks"] > 0
    assert client.get("/api/history").status_code == 401
    headers = login(client)
    assert client.get("/api/auth/me", headers=headers).json() == {"username": "student"}
    assert client.post("/api/auth/register", json={"username": "student", "password": "secret123"}).status_code == 409
    assert client.post("/api/auth/login", json={"username": "student", "password": "wrong-pass"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "student", "password": "secret123"}).status_code == 200
    client.post("/api/auth/logout", headers=headers)
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_auth_can_be_disabled(settings, embedder, store, tmp_path):
    settings.auth_enabled = False
    client = make_client(settings, embedder, store, tmp_path)
    assert client.get("/api/auth/me").json() == {"username": "local"}


def test_chat_streams_answer_with_citations(settings, embedder, store, tmp_path):
    client = make_client(settings, embedder, store, tmp_path)
    headers = login(client)
    response = client.post("/api/chat", json={"message": TOPIC}, headers=headers)
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert events[0] == ("intent", {"intent": "theory"})
    assert "".join(d["text"] for name, d in events if name == "token").startswith("Окончание -s")
    name, final = events[-1]
    assert name == "final" and final["status"] == "answered"
    citation = final["citations"][0]
    assert citation["file"] == "web/book/present-simple.md" and citation["text"]
    assert client.get(f"/api/chunks/{citation['chunk_id']}", headers=headers).json()["file"] == citation["file"]
    history = client.get("/api/chat/history", headers=headers).json()
    assert [m["role"] for m in history] == ["user", "assistant"]


def test_chat_reports_missing_material(settings, embedder, store, tmp_path):
    client = make_client(settings, embedder, store, tmp_path)
    events = sse_events(client.post("/api/chat", json={"message": "квантовая хромодинамика кварков"},
                                    headers=login(client)))
    assert [name for name, _ in events] == ["intent", "final"]
    assert events[-1][1]["status"] == "no_material" and events[-1][1]["citations"] == []
    assert events[-1][1]["answer"] == "В материалах этого нет."


def test_review_endpoint_and_history(settings, embedder, store, tmp_path):
    client = make_client(settings, embedder, store, tmp_path, structured=[REVIEW])
    headers = login(client)
    body = client.post("/api/review", json={"text": "My brother work in a bank.", "task": "About family"},
                       headers=headers).json()
    assert body["errors"][0]["start"] == 3 and body["errors"][0]["citation"]["file"] == "web/book/present-simple.md"
    assert body["grade"] and body["cefr_level"] == "A1"
    assert client.post("/api/review", json={"text": "   "}, headers=headers).status_code == 422
    assert len(client.get("/api/history", headers=headers).json()["reviews"]) == 1
    assert client.get("/api/history", headers=login(client, "other")).json()["reviews"] == []  # чужое не видно


def test_quiz_generate_check_and_topic_stats(settings, embedder, store, tmp_path):
    client = make_client(settings, embedder, store, tmp_path, structured=[QUIZ])
    headers = login(client)
    quiz = client.post("/api/quiz/generate", headers=headers, json={
        "topic": TOPIC, "level": "A1", "count": 1, "types": ["multiple_choice"]}).json()
    assert quiz["status"] == "ready" and quiz["questions"][0]["citation"]["file"] == "web/book/present-simple.md"
    result = client.post(f"/api/quiz/{quiz['id']}/check", json={"answers": {"1": "works"}}, headers=headers).json()
    assert (result["score"], result["total"]) == (1, 1)
    history = client.get("/api/history", headers=headers).json()
    assert history["attempts"][0]["score"] == 1 and history["topics"][0]["percent"] == 100
    assert client.post("/api/quiz/nope/check", json={"answers": {}}, headers=headers).status_code == 404
    assert client.post("/api/quiz/generate", headers=headers, json={"topic": TOPIC, "level": "C2"}).status_code == 422
    no_material = client.post("/api/quiz/generate", headers=headers, json={
        "topic": "квантовая хромодинамика кварков", "level": "B1", "count": 1}).json()
    assert no_material["status"] == "no_material" and no_material["questions"] == []


def test_library_lists_files_with_licenses(settings, embedder, store, tmp_path):
    client = make_client(settings, embedder, store, tmp_path)
    lib = client.get("/api/library", headers=login(client)).json()
    assert lib["indexed_files"] == 2 and lib["chunks"] == store.count()
    entry = next(f for f in lib["files"] if f["file"] == "web/book/present-simple.md")
    assert entry["license"] == "CC BY-SA 4.0" and entry["indexed"] and entry["topics"] == ["a1-present-simple"]


def test_router_intents():
    assert route("Привет!").intent == "smalltalk"
    assert route("Когда используется Present Perfect?").intent == "theory"
    assert route("What is the difference between a and an?").intent == "theory"
    r = route("Проверь, пожалуйста: Yesterday I go to the park with my frend.")
    assert r.intent == "review" and r.text == "Yesterday I go to the park with my frend."
    essay = "My name is Anna and I live in a small town near the sea. " * 3
    assert route(essay).intent == "review"
    q = route("Сделай тест из 7 вопросов по теме Past Simple уровня A2")
    assert (q.intent, q.level, q.count) == ("quiz", "A2", 7) and q.topic == "Past Simple"


def test_study_plan_endpoint_and_chat(settings, embedder, store, tmp_path):
    client = make_client(settings, embedder, store, tmp_path)
    headers = login(client)
    plan = client.get("/api/plan?target=B1", headers=headers).json()
    assert [s["level"] for s in plan["stages"]] == ["A1", "B1"]  # в тестовой карте тем уровня A2 нет
    topics = {t["id"]: t for s in plan["stages"] for t in s["topics"]}
    ps = topics["a1-present-simple"]
    assert ps["materials"][0]["file"] == "web/book/present-simple.md" and ps["materials"][0]["license"] == "CC BY-SA 4.0"
    assert topics["b1-passive"]["materials"] == [] and plan["gaps"] == ["B1 · Грамматика · Пассивный залог"]
    assert (plan["total_topics"], plan["covered_topics"]) == (3, 2)
    assert client.get("/api/plan?target=A1", headers=headers).json()["gaps"] == []
    assert client.get("/api/plan?target=C2", headers=headers).status_code == 422

    events = sse_events(client.post("/api/chat", json={"message": "составь программу обучения до a1"}, headers=headers))
    assert events[0] == ("intent", {"intent": "plan"})
    final = events[-1][1]
    assert final["kind"] == "plan" and final["target_level"] == "A1" and len(final["stages"]) == 1
    assert final["answer"] == "Программа обучения, уровень A1. Тем: 2, с материалами в библиотеке: 2."


def test_router_plan_intent():
    r = route("составь программу обучения до a2")
    assert (r.intent, r.level, r.start_level) == ("plan", "A2", "")
    r = route("Составь план обучения с A2 до B1")
    assert (r.intent, r.level, r.start_level) == ("plan", "B1", "A2")
    assert route("С чего мне начать изучение английского?").intent == "plan"
    assert route("Сделай тест по Past Simple, A2").intent == "quiz"
    assert route("Когда используется Present Perfect?").intent == "theory"


def test_model_down_and_bad_json_give_clear_messages(settings, embedder, store, tmp_path, monkeypatch):
    from tutor.api.app import MODEL_DOWN, MODEL_FAILED

    client = make_client(settings, embedder, store, tmp_path, structured=[{"questions": "не список"}])
    headers = login(client)
    r = client.post("/api/quiz/generate", headers=headers, json={"topic": TOPIC, "level": "A1", "count": 1})
    assert r.status_code == 502 and r.json()["detail"] in (MODEL_FAILED, "Не удалось составить ни одного корректного вопроса.")

    def down(*args, **kwargs):
        raise ConnectionError("Failed to connect to Ollama")

    monkeypatch.setattr(Retriever, "search", down)
    r = client.post("/api/quiz/generate", headers=headers, json={"topic": TOPIC, "level": "A1"})
    assert r.status_code == 503 and r.json()["detail"] == MODEL_DOWN


def test_generated_quiz_does_not_reveal_answers(settings, embedder, store, tmp_path):
    client = make_client(settings, embedder, store, tmp_path, structured=[QUIZ])
    headers = login(client)
    quiz = client.post("/api/quiz/generate", headers=headers, json={
        "topic": TOPIC, "level": "A1", "count": 1, "types": ["multiple_choice"]}).json()
    assert "answer" not in quiz["questions"][0] and "explanation" not in quiz["questions"][0]
    result = client.post(f"/api/quiz/{quiz['id']}/check", json={"answers": {"1": "works"}}, headers=headers).json()
    assert result["score"] == 1 and result["results"][0]["question"]["answer"] == "works"  # после проверки — есть
