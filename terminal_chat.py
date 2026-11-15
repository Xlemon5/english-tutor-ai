import sys
import os
from rag_agent import RAGAgent


def clear_screen():
    """Очистка экрана терминала"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Красивая шапка"""
    print("ENGLISH TUTOR AI - Терминальный режим")
    print("Команды:")
    print("  /help    - показать помощь")
    print("  /clear   - очистить экран")
    print("  /exit    - выйти из программы")
    print("  /upload  - загрузить PDF файл")


def main():
    """Главная функция терминального чата"""
    print("Загрузка агента...")
    agent = RAGAgent()

    clear_screen()
    print_header()

    # Запрос имени пользователя
    username = input("\nВведите ваше имя: ").strip()
    if not username:
        username = "Ученик"

    print(f"\nПривет, {username}! Задавайте вопросы по английскому языку.")
    print("   (или введите /help для списка команд)\n")

    # Основной цикл чата
    while True:
        try:
            # Получаем вопрос от пользователя
            question = input(f"{username}: ").strip()

            # Обработка команд
            if question.lower() == '/exit':
                print("\nДо свидания! Хорошего дня!")
                break

            elif question.lower() == '/help':
                print("\nДоступные команды:")
                print("  /help    - показать эту помощь")
                print("  /clear   - очистить экран")
                print("  /exit    - выйти из программы")
                print("  /upload  - загрузить PDF файл в базу знаний")
                print("  /quiz    - сгенерировать тест по теме")
                print("  /stats   - показать статистику базы знаний\n")
                continue

            elif question.lower() == '/clear':
                clear_screen()
                print_header()
                print(f"\nПривет, {username}! Задавайте вопросы.\n")
                continue

            elif question.lower() == '/upload':
                file_path = input("Введите путь к PDF файлу: ").strip()
                if os.path.exists(file_path) and file_path.endswith('.pdf'):
                    print("\nОбработка файла...")
                    result = agent.process_and_upload_pdf(file_path)
                    print(f"{result}\n")
                else:
                    print("Файл не найден или не является PDF.\n")
                continue

            elif question.lower() == '/quiz':
                topic = input("Введите тему для теста: ").strip()
                if topic:
                    print("\nГенерация теста...")
                    quiz = agent.generate_quiz(topic)
                    print(f"\n{quiz}\n")
                continue

            elif question.lower() == '/stats':
                # Показываем статистику базы знаний
                try:
                    count = agent.collection.count()
                    print(f"\nСтатистика базы знаний:")
                    print(f"   Фрагментов в базе: {count}")
                    print(f"   Лексических записей: ", end="")
                    import sqlite3
                    conn = sqlite3.connect("words.db")
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM words_lexical")
                    print(f"{cursor.fetchone()[0]}")
                    conn.close()
                    print()
                except Exception as e:
                    print(f"Ошибка получения статистики: {e}\n")
                continue

            # Если это не команда, обрабатываем как вопрос
            if not question:
                continue

            print(f"\nAI-Tutor думает...")
            answer = agent.ask_tutor(question, username)
            print(f"\nОтвет:\n{answer}\n")
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nПрограмма прервана. До свидания!")
            break
        except Exception as e:
            print(f"\nОшибка: {e}\n")


if __name__ == "__main__":
    main()