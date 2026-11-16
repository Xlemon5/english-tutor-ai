"""Веб-интерфейс репетитора (Streamlit). Запуск: uv run streamlit run web/app.py"""
from __future__ import annotations

import streamlit as st
import views
from client import Api, ApiError

st.set_page_config(page_title="Репетитор по английскому", page_icon="📘", layout="centered")


def login_screen(registration: bool) -> None:
    st.title("📘 Репетитор по английскому")
    st.caption("Уровни A1–B1. Ответы строятся только на материалах библиотеки.")
    tabs = st.tabs(["Вход", "Регистрация"] if registration else ["Вход"])
    for tab, path, label in zip(tabs, ("/api/auth/login", "/api/auth/register"), ("Войти", "Зарегистрироваться")):
        with tab, st.form(path):
            username = st.text_input("Логин")
            password = st.text_input("Пароль", type="password", help="Не короче 6 символов")
            if st.form_submit_button(label, type="primary"):
                try:
                    data = Api().post(path, {"username": username.strip(), "password": password})
                except ApiError as e:
                    st.error(e.message)
                else:
                    st.session_state["token"] = data["token"]
                    st.session_state["username"] = data["username"]
                    st.rerun()


def main() -> None:
    try:
        health = Api().get("/api/health")
    except ApiError as e:
        st.error(f"Не удаётся подключиться к API. {e.message}")
        st.stop()
    if health["auth"] and not st.session_state.get("token"):
        login_screen(health["registration"])
        return
    pages = {
        "chat": st.Page(views.page_chat, title="Чат", icon="💬", url_path="chat", default=True),
        "review": st.Page(views.page_review, title="Проверка работ", icon="📝", url_path="review"),
        "quiz": st.Page(views.page_quiz, title="Тесты", icon="✅", url_path="quiz"),
        "plan": st.Page(views.page_plan, title="Программа", icon="🗺️", url_path="plan"),
        "history": st.Page(views.page_history, title="История", icon="📈", url_path="history"),
        "library": st.Page(views.page_library, title="Библиотека", icon="📚", url_path="library"),
    }
    st.session_state["pages"] = pages
    with st.sidebar:
        st.caption(f"Модель: {health['llm']} · фрагментов в индексе: {health['chunks']}")
        if health["auth"]:
            st.caption(f"Вы вошли как **{st.session_state.get('username', '')}**")
            if st.button("Выйти", width="stretch"):
                Api(st.session_state["token"]).post("/api/auth/logout")
                st.session_state.clear()
                st.rerun()
        if health["chunks"] == 0:
            st.warning("Индекс пуст. Выполните `make ingest`.")
    st.navigation(list(pages.values())).run()


main()
