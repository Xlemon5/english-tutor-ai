"""
Скрипт для ручной переиндексации документов.
Использовать, если добавили новые PDF в папки data/documents/
"""

from rag_agent import RAGAgent

print("🔄 Запуск переиндексации документов...")
agent = RAGAgent()
info = agent.index_all_documents(clear_first=True)

print(f"\nГотово!")
print(f"   Файлов обработано: {info['files']}")
print(f"   Чанков создано: {info['chunks']}")
print("\nОтчёт:")
for line in info["report"]:
    print(f"   {line}")