import gradio as gr
import database as db
from rag_agent import RAGAgent
import re

# --- ИНИЦИАЛИЗАЦИЯ ---
agent = RAGAgent()
current_user = None
current_quiz = None

# Авто-индексация при запуске
if agent.stats()["documents"] == 0:
    print("База пуста — запускаю индексацию...")
    info = agent.index_all_documents()
    print(f"Готово: {info['files']} файлов, {info['chunks']} чанков")
else:
    print(f"База загружена ({agent.stats()['documents']} чанков)")


# --- ФУНКЦИИ ПРОФИЛЯ И ЧАТА ---

def login(username):
    global current_user
    if not username.strip():
        return "❌ Введите имя", 0, 0
    username = username.strip().lower()
    db.add_user(username)
    current_user = username
    stats = db.get_user_stats(username)
    return f"✅ Привет, {username}!", stats["total_score"], stats["tests_completed"]


def respond(message, history):
    global current_user
    if not current_user:
        history.append((message, "⚠️ Сначала войдите в систему."))
        return "", history
    answer = agent.ask_tutor(message, current_user)
    history.append((message, answer))
    return "", history


# --- ЛОГИКА ТЕСТА С RAG-РЕКОМЕНДАЦИЯМИ ---

def start_quiz(topic, num_questions):
    global current_user, current_quiz
    if not current_user:
        return "⚠️ Сначала войдите в систему.", "", "", 0, gr.update(interactive=False)
    n = int(num_questions)
    if n < 1 or n > 10:
        return "Введите от 1 до 10 вопросов.", "", "", 0, gr.update(interactive=False)

    print(f"Генерация теста: {n} вопросов по теме '{topic}'...")
    questions = agent.generate_quiz_with_sources(topic, n)
    if not questions:
        return "❌ Не удалось создать тест, прошедший проверку источников и ответов. Попробуй уточнить тему или выбрать другую.", "", "", 0, gr.update(interactive=False)

    current_quiz = {"questions": questions, "index": 0, "score": 0,
                    "answered": False, "topic": topic}
    return show_question()


def show_question():
    global current_quiz
    q = current_quiz['questions'][current_quiz['index']]
    num = current_quiz['index'] + 1
    total = len(current_quiz['questions'])

    q_text = f"**Вопрос {num} из {total}:**\n\n{q['question']}"

    opts = []
    for letter in ['A', 'B', 'C', 'D']:
        if letter in q['options']:
            opts.append(f"**{letter})** {q['options'][letter]}")

    opt_text = "\n\n".join(opts)
    return q_text, opt_text, "", current_quiz['score'], gr.update(interactive=False)


def check_answer(user_answer):
    global current_quiz
    if not current_quiz or current_quiz["answered"]:
        return "⚠️ Сначала начните тест.", "", "", current_quiz["score"], gr.update(interactive=False)
    if not user_answer or user_answer.strip().upper() not in ["A", "B", "C", "D"]:
        return "⚠️ Введите букву A, B, C или D.", "", "", current_quiz["score"], gr.update(interactive=False)

    q = current_quiz["questions"][current_quiz["index"]]
    selected = user_answer.strip().upper()
    correct = q["correct"]

    if selected == correct:
        current_quiz["score"] += 1
        feedback = "✅ **Правильно!** (+1 балл)"
    else:
        feedback = (f"❌ **Неправильно.**\nПравильный ответ: "
                    f"**{correct}) {q['options'].get(correct, '')}**")

    # Источник, из которого был составлен этот вопрос — всегда точный
    if q.get("source"):
        feedback += (f"\n\n📖 **Источник:** {q['source']}, стр. {q['page']} "
                     f"(раздел «{current_quiz['topic']}»)")
    else:
        feedback += "\n\n📖 *Источник не указан.*"

    current_quiz["answered"] = True
    q_text, opt_text, _, score, _ = show_question()
    return q_text, opt_text, feedback, score, gr.update(interactive=True)


def next_question():
    global current_quiz

    if not current_quiz or not current_quiz['answered']:
        return "⚠️ Сначала проверьте ответ.", "", "", current_quiz['score'], gr.update(interactive=False)

    current_quiz['index'] += 1
    current_quiz['answered'] = False

    if current_quiz['index'] >= len(current_quiz['questions']):
        return finish_quiz()

    return show_question()


def finish_quiz():
    global current_quiz, current_user

    score = current_quiz['score']
    total = len(current_quiz['questions'])
    topic = current_quiz['topic']

    if current_user:
        db.save_test_result(current_user, topic, score, total)

    percent = (score / total) * 100
    result = f"🎉 **Тест по теме '{topic}' завершён!**\n\nРезультат: **{score} из {total}** ({percent:.0f}%)\n"

    if percent == 100:
        result += "🌟 Идеально!"
    elif percent >= 70:
        result += "👍 Хороший результат!"
    else:
        result += "📚 Обратите внимание на рекомендации в пройденных вопросах, чтобы подтянуть знания."

    current_quiz = None
    stats = db.get_user_stats(current_user)

    return result, "", "", stats["total_score"], gr.update(interactive=False)


def get_history_display():
    """ИСПРАВЛЕНО: теперь возвращает строку, а не список, чтобы gr.Markdown не ломался"""
    global current_user
    if not current_user:
        return "⚠️ Сначала войдите в систему."

    history = db.get_user_history(current_user, limit=10)
    if not history:
        return "📋 У вас пока нет пройденных тестов."

    text = "📋 **История ваших тестов:**\n\n"
    for h in history:
        text += f"• **{h['topic']}**: {h['score']}/{h['total']} ({(h['score'] / h['total'] * 100):.0f}%) — {h['date']}\n"

    return text


# --- ИНТЕРФЕЙС ---

with gr.Blocks(title="English Tutor AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 English Tutor AI\nRAG-агент для изучения английского по вашим документам")

    with gr.Tab("👤 Профиль"):
        u = gr.Textbox(label="Ваше имя", placeholder="Введите имя...")
        b = gr.Button("Войти", variant="primary")
        out = gr.Markdown("### Статус: Не авторизован")
        sc = gr.Number(label="Всего баллов", value=0)
        tc = gr.Number(label="Пройдено тестов", value=0)
        b.click(login, u, [out, sc, tc])

    with gr.Tab("💬 Чат"):
        chat = gr.Chatbot(height=400)
        msg = gr.Textbox(label="Ваш вопрос")
        msg.submit(respond, [msg, chat], [msg, chat])

    with gr.Tab("📝 Тест"):
        gr.Markdown(
            "### Тестирование\nВыберите категорию. Вопросы будут составлены строго по материалам из соответствующих PDF-файлов.")

        with gr.Row():
            with gr.Column():
                # Выпадающий список вместо свободного ввода
                topic_in = gr.Dropdown(
                    label="Категория материалов",
                    choices=["Grammar", "Vocabulary", "Communication"],
                    value="Grammar"
                )
                num_in = gr.Number(label="Кол-во вопросов", value=3, minimum=1, maximum=10)
                start_btn = gr.Button("🚀 Начать тест", variant="primary")

            with gr.Column():
                q_md = gr.Markdown("### Вопрос появится здесь")
                opt_md = gr.Markdown("### Варианты ответов")

                ans_in = gr.Textbox(label="Ваш ответ (A, B, C или D)", placeholder="A")

                with gr.Row():
                    check_btn = gr.Button("✅ Проверить", variant="secondary")
                    next_btn = gr.Button("➡️ Дальше", variant="primary", interactive=False)

                fb_md = gr.Markdown("### Здесь будет результат проверки и рекомендация")
                score_md = gr.Number(label="Текущий счёт", value=0)

        start_btn.click(start_quiz, [topic_in, num_in], [q_md, opt_md, fb_md, score_md, next_btn])
        check_btn.click(check_answer, ans_in, [q_md, opt_md, fb_md, score_md, next_btn])
        next_btn.click(next_question, None, [q_md, opt_md, fb_md, score_md, next_btn])

    with gr.Tab("📊 История"):
        h_btn = gr.Button("Показать историю")
        h_md = gr.Markdown("Нажмите кнопку, чтобы увидеть историю.")
        # ИСПРАВЛЕНО: вызываем функцию get_history_display, которая возвращает строку
        h_btn.click(get_history_display, None, h_md)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
