"""SQLite: пользователи, сессии, история чата, проверок и тестов."""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, salt TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, username TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, payload TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL, grade INTEGER, cefr_level TEXT,
    errors INTEGER, result TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS quizzes (
    id TEXT PRIMARY KEY, username TEXT NOT NULL, topic TEXT NOT NULL, level TEXT NOT NULL,
    quiz TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, quiz_id TEXT NOT NULL, username TEXT NOT NULL, topic TEXT NOT NULL,
    level TEXT NOT NULL, score INTEGER NOT NULL, total INTEGER NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _hash(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


class Storage:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- пользователи и сессии
    def create_user(self, username: str, password: str) -> bool:
        salt = secrets.token_hex(16)
        try:
            with self._conn() as c:
                c.execute("INSERT INTO users VALUES (?, ?, ?, ?)", (username, _hash(password, salt), salt, _now()))
            return True
        except sqlite3.IntegrityError:
            return False

    def verify_user(self, username: str, password: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,)).fetchone()
        return bool(row) and hmac.compare_digest(row["password_hash"], _hash(password, row["salt"]))

    def create_session(self, username: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._conn() as c:
            c.execute("INSERT INTO sessions VALUES (?, ?, ?)", (token, username, _now()))
        return token

    def session_user(self, token: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT username FROM sessions WHERE token = ?", (token,)).fetchone()
        return row["username"] if row else None

    def delete_session(self, token: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM sessions WHERE token = ?", (token,))

    # --- чат
    def add_message(self, username: str, role: str, content: str, payload: dict | None = None) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO chat_messages (username, role, content, payload, created_at) VALUES (?,?,?,?,?)",
                      (username, role, content, json.dumps(payload, ensure_ascii=False) if payload else None, _now()))

    def messages(self, username: str, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT role, content, payload, created_at FROM chat_messages WHERE username = ? "
                             "ORDER BY id DESC LIMIT ?", (username, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"],
                 "payload": json.loads(r["payload"]) if r["payload"] else None} for r in reversed(rows)]

    def clear_messages(self, username: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM chat_messages WHERE username = ?", (username,))

    # --- проверки работ
    def add_review(self, username: str, result: dict) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO reviews (username, grade, cefr_level, errors, result, created_at) VALUES (?,?,?,?,?,?)",
                (username, result["grade"], result["cefr_level"], len(result["errors"]),
                 json.dumps(result, ensure_ascii=False), _now()))
            return int(cur.lastrowid)

    def reviews(self, username: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, grade, cefr_level, errors, result, created_at FROM reviews "
                             "WHERE username = ? ORDER BY id DESC", (username,)).fetchall()
        return [{"id": r["id"], "grade": r["grade"], "cefr_level": r["cefr_level"], "errors": r["errors"],
                 "created_at": r["created_at"], "result": json.loads(r["result"])} for r in rows]

    # --- тесты
    def add_quiz(self, username: str, quiz: dict) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO quizzes VALUES (?,?,?,?,?,?)",
                      (quiz["id"], username, quiz["topic"], quiz["level"], json.dumps(quiz, ensure_ascii=False), _now()))

    def get_quiz(self, username: str, quiz_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT quiz FROM quizzes WHERE id = ? AND username = ?", (quiz_id, username)).fetchone()
        return json.loads(row["quiz"]) if row else None

    def add_attempt(self, username: str, result: dict) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO quiz_attempts (quiz_id, username, topic, level, score, total, result, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (result["quiz_id"], username, result["topic"], result["level"], result["score"], result["total"],
                 json.dumps(result, ensure_ascii=False), _now()))
            return int(cur.lastrowid)

    def attempts(self, username: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT id, quiz_id, topic, level, score, total, result, created_at FROM quiz_attempts "
                             "WHERE username = ? ORDER BY id DESC", (username,)).fetchall()
        return [{**{k: r[k] for k in ("id", "quiz_id", "topic", "level", "score", "total", "created_at")},
                 "result": json.loads(r["result"])} for r in rows]

    def topic_stats(self, username: str) -> list[dict]:
        """Результаты тестов по темам: сколько попыток и средний процент верных ответов."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT topic, level, COUNT(*) AS attempts, SUM(score) AS score, SUM(total) AS total, "
                "MAX(created_at) AS last FROM quiz_attempts WHERE username = ? GROUP BY lower(topic), level "
                "ORDER BY last DESC", (username,)).fetchall()
        return [{"topic": r["topic"], "level": r["level"], "attempts": r["attempts"], "last": r["last"],
                 "percent": round(100 * r["score"] / r["total"]) if r["total"] else 0} for r in rows]
