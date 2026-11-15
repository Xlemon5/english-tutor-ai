import os
# Подавляем надоедливые ошибки телеметрии ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import requests
from rag_agent import RAGAgent

print("1️⃣ Инициализация агента...")
agent = RAGAgent()
print("✅ Агент инициализирован.\n")

print("2️⃣ Проверка доступности Ollama...")
try:
    r = requests.get("http://localhost:11434/api/tags", timeout=5)
    print(f"✅ Ollama отвечает. Статус: {r.status_code}\n")
except Exception as e:
    print(f"❌ Ollama НЕ отвечает: {e}")
    print("💡 Решение: выполни в терминале 'docker restart ollama'")
    exit()

question = "Что такое Present Simple? Объясни кратко."
print(f"3️⃣ Начинаем обработку вопроса: '{question}'\n")

print("  3.1. Лексический поиск (SQLite)...")
lex = agent.lexical_search(question, 3)
print("  ✅ Лексический поиск завершён.\n")

print("  3.2. Семантический поиск (Эмбеддинг + ChromaDB)...")
sem = agent.semantic_search(question, 3)
print("  ✅ Семантический поиск завершён.\n")

print("  3.3. Отправка запроса к LLM (Qwen 2.5)...")
print("  ⏳ (Это может занять 10-30 секунд, если модель загружается в память)...")
try:
    answer = agent.ask_tutor(question, "Тестер")
    print("  ✅ Ответ от LLM успешно получен!\n")
    print("=" * 60)
    print(answer)
    print("=" * 60)
except Exception as e:
    print(f"  ❌ КРИТИЧЕСКАЯ ОШИБКА при запросе к LLM: {type(e).__name__}: {e}")