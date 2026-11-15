import sqlite3
import os
from datetime import datetime

DB_NAME = "users.db"


def init_db():
    """Создает таблицы, если их нет"""
    os.makedirs(os.path.dirname(os.path.abspath(DB_NAME)), exist_ok=True)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Таблица пользователей
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            total_score INTEGER DEFAULT 0,
            tests_completed INTEGER DEFAULT 0
        )
    ''')

    # Таблица результатов тестов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            topic TEXT,
            score INTEGER,
            total_questions INTEGER,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')

    conn.commit()
    conn.close()


def add_user(username: str):
    """Добавляет или обновляет пользователя"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))
    conn.commit()
    conn.close()


def get_user_stats(username: str):
    """Получает статистику пользователя"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT total_score, tests_completed FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {"total_score": result[0], "tests_completed": result[1]}
    return {"total_score": 0, "tests_completed": 0}


def save_test_result(username: str, topic: str, score: int, total_questions: int):
    """Сохраняет результат теста и обновляет общую статистику"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Сохраняем результат теста
    cursor.execute(
        "INSERT INTO test_results (username, topic, score, total_questions) VALUES (?, ?, ?, ?)",
        (username, topic, score, total_questions)
    )

    # Обновляем общую статистику пользователя
    cursor.execute(
        "UPDATE users SET total_score = total_score + ?, tests_completed = tests_completed + 1 WHERE username = ?",
        (score, username)
    )

    conn.commit()
    conn.close()


def get_user_history(username: str, limit: int = 10):
    """Получает историю тестов пользователя"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT topic, score, total_questions, completed_at
           FROM test_results
           WHERE username = ?
           ORDER BY completed_at DESC
           LIMIT ?""",
        (username, limit)
    )
    results = cursor.fetchall()
    conn.close()
    return [
        {
            "topic": r[0],
            "score": r[1],
            "total": r[2],
            "date": r[3]
        }
        for r in results
    ]


# Инициализируем БД при импорте
init_db()
