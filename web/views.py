"""Страницы веб-интерфейса: чат, проверка работ, тесты, история, библиотека."""
from __future__ import annotations

import html
from datetime import datetime

import streamlit as st
from client import Api, ApiError

LEVELS = ["A1", "A2", "B1"]
TYPES = {"multiple_choice": "Выбор ответа", "fill_gap": "Заполнение пропусков",
         "error_correction": "Исправление ошибок", "translation": "Перевод предложений"}
ERROR_COLORS = {"grammar": "#ffd6d6", "vocabulary": "#ffe7c2", "spelling": "#d9e8ff",
                "punctuation": "#e3dcff", "style": "#d8f3dc"}
NO_MATERIAL = "В материалах этого нет"


def when(iso: str) -> str:
    """Время из истории (хранится в UTC) в местном часовом поясе."""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return iso[:16]


def api() -> Api:
    return Api(st.session_state.get("token", ""))


# ------------------------------------------------------------------ общие элементы


def citation_popover(c: dict, key: str) -> None:
    """Ссылка на источник: по клику открывается найденный фрагмент."""
    name = " › ".join(p for p in (c.get("title"), c.get("section")) if p)
    page = f", с. {c['page']}" if c.get("page") else ""
    with st.popover(f"[{c['n']}] {name[:70]}{page}", key=key, width="stretch"):
        st.markdown(f"**{name}{page}**")
        st.caption(f"Файл: `{c['file']}`")
        if c.get("url"):
            st.markdown(f"[Открыть источник]({c['url']})")
        licence = " · ".join(p for p in (c.get("author"), c.get("license")) if p)
        if licence:
            st.caption(licence)
        st.text_area("Найденный фрагмент", c["text"], height=220, disabled=True, key=key + "-text")


def show_citations(citations: list[dict], key: str) -> None:
    if citations:
        st.caption("Источники — нажмите, чтобы увидеть фрагмент:")
        for c in citations:
            citation_popover(c, f"{key}-c{c['n']}")


def show_no_material(payload: dict) -> None:
    st.warning(f"**{NO_MATERIAL}.** Я отвечаю только по учебным материалам из библиотеки и не подменяю их "
               "собственными знаниями.", icon="⚠️")
    if payload.get("suggestions"):
        st.caption("Близкие темы, которые в материалах есть:")
        for s in payload["suggestions"]:
            st.markdown(f"- {s}")


def highlighted_text(text: str, errors: list[dict]) -> str:
    """Исходный текст с подсветкой ошибок; по клику на фрагмент открывается карточка с разбором."""
    out, pos = [], 0
    located = sorted((e for e in errors if e.get("start") is not None), key=lambda e: e["start"])
    for n, e in enumerate(located, 1):
        if e["start"] < pos:
            continue
        out.append(html.escape(text[pos:e["start"]]))
        theory = e.get("citation")
        if theory:
            name = html.escape(" › ".join(p for p in (theory["title"], theory["section"]) if p))
            link = f' · <a href="{html.escape(theory["url"])}" target="_blank">источник</a>' if theory.get("url") else ""
            theory_html = (f'<div class="th"><b>Теория:</b> {name}{link}<br>'
                           f'<i>{html.escape(theory["text"][:350])}…</i></div>')
        else:
            theory_html = f'<div class="th"><b>Теория:</b> {html.escape(e.get("theory_note") or "")}</div>'
        out.append(
            f'<span class="err" tabindex="0" style="background:{ERROR_COLORS.get(e["type"], "#eee")}">'
            f'{html.escape(text[e["start"]:e["end"]])}<sup>{n}</sup>'
            f'<span class="card"><b>{html.escape(e["fragment"])}</b> → <b class="fix">{html.escape(e["correction"])}</b>'
            f'<br><span class="tp">{html.escape(e["type_ru"])}</span><br>{html.escape(e["explanation"])}'
            f'{theory_html}</span></span>')
        pos = e["end"]
    out.append(html.escape(text[pos:]))
    style = """<style>
.rv{line-height:2.1;font-size:1.05rem;white-space:pre-wrap;color:#1f2430;background:#fff;border:1px solid #ddd;
border-radius:8px;padding:14px 16px}
.rv .err{position:relative;border-radius:4px;padding:2px 3px;cursor:pointer;outline:none}
.rv .err:focus{box-shadow:0 0 0 2px #e25555}
.rv .card{display:none;position:absolute;left:0;top:1.9em;z-index:50;width:min(340px,80vw);background:#fff;
color:#1f2430;border:1px solid #bbb;border-radius:8px;padding:10px 12px;font-size:.9rem;line-height:1.45;
box-shadow:0 6px 20px rgba(0,0,0,.18);white-space:normal;cursor:default}
.rv .err:focus .card,.rv .err:focus-within .card{display:block}
.rv .fix{color:#18794e}.rv .tp{color:#666;font-size:.8rem;text-transform:uppercase}
.rv .th{margin-top:6px;padding-top:6px;border-top:1px solid #eee;font-size:.85rem}
</style>"""
    return style + f'<div class="rv">{"".join(out)}</div>'


def show_review(result: dict, key: str) -> None:
    st.markdown("**Текст с отмеченными ошибками** — нажмите на выделенный фрагмент:")
    st.html(highlighted_text(result["text"], result["errors"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("Оценка", f"{result['grade']} / 5")
    c2.metric("Примерный уровень", result["cefr_level"])
    c3.metric("Ошибок", len(result["errors"]))
    st.info(result["summary"])
    if result["review_topics"]:
        st.markdown("**Стоит повторить:** " + "; ".join(result["review_topics"]))
    if result["errors"]:
        st.markdown("**Разбор ошибок**")
    for n, e in enumerate(result["errors"], 1):
        with st.expander(f"{n}. «{e['fragment']}» → «{e['correction']}» · {e['type_ru']}"):
            st.write(e["explanation"])
            if e.get("citation"):
                citation_popover(e["citation"], f"{key}-e{n}")
            else:
                st.caption(f"Ссылка на теорию: {e.get('theory_note') or 'нет'}")


def show_quiz_result(result: dict, key: str) -> None:
    st.subheader(f"Результат: {result['score']} из {result['total']}")
    st.progress(result["score"] / max(result["total"], 1))
    for r in result["results"]:
        q = r["question"]
        mark = "✅" if r["correct"] else "❌"
        with st.expander(f"{mark} {q['id']}. {q['prompt']}", expanded=not r["correct"]):
            st.markdown(f"Ваш ответ: **{r['given'] or '—'}**")
            st.markdown(f"Правильный ответ: **{q['answer']}**")
            if r.get("comment"):
                st.caption(r["comment"])
            st.write(q["explanation"])
            if q.get("citation"):
                citation_popover(q["citation"], f"{key}-q{q['id']}")
            else:
                st.caption("Ссылка на теорию: в материалах не найдено")


def show_plan(plan: dict, key: str, passed: dict[str, int] | None = None) -> None:
    """Программа обучения: темы по уровням, к каждой — материалы библиотеки и переход к тесту."""
    passed = passed or {}
    for stage in plan["stages"]:
        st.subheader(f"Уровень {stage['level']}")
        area = ""
        for t in stage["topics"]:
            if t["area_ru"] != area:
                area = t["area_ru"]
                st.markdown(f"**{area}**")
            percent = passed.get(t["title"].lower())
            icon = "✅" if percent is not None and percent >= 70 else "📘" if t["materials"] else "⚪"
            note = f" · тест: {percent}%" if percent is not None else "" if t["materials"] else " · материалов нет"
            with st.expander(f"{icon} {t['title']}{note}"):
                if not t["materials"]:
                    st.caption("В библиотеке нет материалов по этой теме. Агент по ней не отвечает.")
                    continue
                for m in t["materials"]:
                    name = f"[{m['title']}]({m['url']})" if m.get("url") else m["title"]
                    credit = ", ".join(p for p in (m.get("author"), m.get("license")) if p)
                    st.markdown(f"- {m['kind_ru']} · {name}  \n  :gray[{credit} · `{m['file']}`]")
                if t["materials_total"] > len(t["materials"]):
                    st.caption(f"Всего материалов по теме в библиотеке: {t['materials_total']}.")
                if st.button("Пройти тест по теме", key=f"{key}-{t['id']}"):
                    st.session_state["quiz_prefill"] = {"topic": t["title"], "level": t["level"]}
                    for name in ("quiz", "quiz_result", "quiz_pending"):
                        st.session_state.pop(name, None)
                    st.switch_page(st.session_state["pages"]["quiz"])
    if plan["gaps"]:
        st.caption("Темы без материалов: " + "; ".join(plan["gaps"]))


# ------------------------------------------------------------------ чат


def render_assistant(payload: dict, key: str) -> None:
    kind = payload.get("kind")
    if kind == "theory":
        if payload["status"] == "no_material":
            show_no_material(payload)
        else:
            st.markdown(payload["answer"])
            show_citations(payload["citations"], key)
    elif kind == "review":
        show_review(payload, key)
    elif kind == "plan":
        st.markdown(payload["answer"])
        show_plan(payload, key)
    elif kind == "quiz":
        if payload["status"] == "no_material":
            st.warning(payload["message"], icon="⚠️")
        else:
            st.markdown(f"Тест по теме «{payload['topic']}» ({payload['level']}) готов: "
                        f"{len(payload['questions'])} вопросов.")
            if st.button("Перейти к тесту", key=key + "-go"):
                st.session_state["quiz"] = payload
                st.session_state.pop("quiz_result", None)
                st.switch_page(st.session_state["pages"]["quiz"])
    else:
        st.markdown(payload.get("answer", ""))


def page_chat() -> None:
    st.title("Чат с репетитором")
    st.caption("Вопросы по теории — с ответами по материалам и ссылками. Можно прислать текст на проверку, "
               "попросить тест («Сделай тест по Past Simple, A2») или программу обучения («Программа до A2»).")
    if "chat" not in st.session_state:
        try:
            st.session_state["chat"] = api().get("/api/chat/history")
        except ApiError:
            st.session_state["chat"] = []
    with st.sidebar:
        level = st.selectbox("Уровень материалов", ["Любой", *LEVELS], key="chat_level")
        if st.button("Очистить чат", width="stretch"):
            api().delete("/api/chat/history")
            st.session_state["chat"] = []
            st.rerun()
    for i, m in enumerate(st.session_state["chat"]):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_assistant(m.get("payload") or {"answer": m["content"]}, f"m{i}")
    message = st.chat_input("Спросите о правиле, пришлите текст или попросите тест")
    if not message:
        return
    st.session_state["chat"].append({"role": "user", "content": message})
    with st.chat_message("user"):
        st.markdown(message)
    with st.chat_message("assistant"):
        final: dict = {}
        error = ""
        placeholder = st.empty()

        def tokens():
            nonlocal final, error
            for event, data in api().chat(message, None if level == "Любой" else level):
                if event == "token":
                    yield data["text"]
                elif event == "final":
                    final = data
                elif event == "error":
                    error = data["message"]

        try:
            with st.spinner("Ищу в материалах…"), placeholder.container():
                st.write_stream(tokens())
        except ApiError as e:
            error = e.message
        placeholder.empty()
        if error or not final:
            st.error(error or "Ответ не получен.")
            return
        st.session_state["chat"].append({"role": "assistant", "content": final.get("answer", ""), "payload": final})
        render_assistant(final, f"m{len(st.session_state['chat']) - 1}")


# ------------------------------------------------------------------ проверка работ


def page_review() -> None:
    st.title("Проверка работы")
    st.caption("Вставьте текст на английском: эссе, перевод или ответы на упражнение. Требования — не выше B1.")
    with st.form("review_form"):
        task = st.text_input("Задание (необязательно)", placeholder="Write about your last weekend.")
        text = st.text_area("Текст студента", height=220, max_chars=3000,
                            placeholder="Yesterday I go to the park with my frend…")
        submitted = st.form_submit_button("Проверить", type="primary")
    if submitted:
        if not text.strip():
            st.error("Введите текст работы.")
        else:
            try:
                with st.spinner("Проверяю работу — обычно это занимает до минуты…"):
                    st.session_state["review"] = api().post("/api/review", {"text": text, "task": task})
            except ApiError as e:
                st.error(e.message)
    if st.session_state.get("review"):
        st.divider()
        show_review(st.session_state["review"], "rv")


# ------------------------------------------------------------------ тесты


def page_quiz() -> None:
    st.title("Тесты")
    prefill = st.session_state.get("quiz_prefill") or {}
    if prefill:
        st.info(f"Тема из программы обучения: «{prefill['topic']}». Нажмите «Составить тест».")
    with st.expander("Параметры нового теста", expanded="quiz" not in st.session_state), st.form("quiz_form"):
        topic = st.text_input("Тема", value=prefill.get("topic", ""),
                              placeholder="Past Simple, articles, comparative adjectives…")
        c1, c2 = st.columns(2)
        level = c1.selectbox("Уровень", LEVELS, index=LEVELS.index(prefill.get("level", "A2")))
        count = c2.slider("Количество вопросов", 1, 15, 5)
        types = st.multiselect("Типы заданий", list(TYPES), default=list(TYPES), format_func=TYPES.get)
        create = st.form_submit_button("Составить тест", type="primary")
    request = None
    if create:
        if len(topic.strip()) < 2 or not types:
            st.error("Укажите тему и хотя бы один тип заданий.")
        else:
            request = {"topic": topic, "level": level, "count": count, "types": types}
    pending = st.session_state.get("quiz_pending")
    if pending:
        st.warning(pending["message"], icon="⚠️")
        for s in pending.get("suggestions", []):
            st.markdown(f"- близкая тема в материалах: {s}")
        if st.button("Сгенерировать без опоры на материалы"):
            request = {**pending["request"], "allow_without_materials": True}
    if request:
        try:
            with st.spinner("Составляю тест — это может занять до пары минут…"):
                quiz = api().post("/api/quiz/generate", request)
        except ApiError as e:
            st.error(e.message)
            return
        st.session_state.pop("quiz_result", None)
        st.session_state.pop("quiz_prefill", None)
        if quiz["status"] == "no_material":
            st.session_state["quiz_pending"] = {**quiz, "request": request}
            st.session_state.pop("quiz", None)
        else:
            st.session_state["quiz"] = quiz
            st.session_state.pop("quiz_pending", None)
        st.rerun()

    quiz = st.session_state.get("quiz")
    if not quiz:
        return
    st.divider()
    st.subheader(f"{quiz['topic']} · {quiz['level']}")
    if not quiz["grounded"]:
        st.warning("Тест составлен без опоры на материалы: ссылок на теорию в нём нет.", icon="⚠️")
    if st.session_state.get("quiz_result"):
        show_quiz_result(st.session_state["quiz_result"], "qr")
        if st.button("Пройти ещё раз"):
            st.session_state.pop("quiz_result")
            st.rerun()
        return
    with st.form("quiz_take"):
        answers: dict[str, str] = {}
        for q in quiz["questions"]:
            st.markdown(f"**{q['id']}. {q['prompt']}**  \n:gray[{q['type_ru']}]")
            key = f"a-{quiz['id']}-{q['id']}"
            if q["type"] == "multiple_choice":
                answers[str(q["id"])] = st.radio("Ответ", q["options"], index=None, key=key,
                                                 label_visibility="collapsed") or ""
            else:
                answers[str(q["id"])] = st.text_input("Ответ", key=key, label_visibility="collapsed",
                                                      placeholder="Ваш ответ")
        done = st.form_submit_button("Проверить ответы", type="primary")
    if done:
        try:
            with st.spinner("Проверяю ответы…"):
                st.session_state["quiz_result"] = api().post(f"/api/quiz/{quiz['id']}/check", {"answers": answers})
        except ApiError as e:
            st.error(e.message)
            return
        st.rerun()


# ------------------------------------------------------------------ программа обучения


def page_plan() -> None:
    st.title("Программа обучения")
    st.caption("Темы по уровням CEFR и материалы библиотеки к каждой. Программа собирается из карты тем и "
               "списка материалов — в ней нет ничего, чего нет в библиотеке.")
    c1, c2 = st.columns(2)
    start = c1.selectbox("С уровня", LEVELS, index=0)
    target = c2.selectbox("До уровня", LEVELS, index=len(LEVELS) - 1)
    try:
        plan = api().get(f"/api/plan?start={start}&target={target}")
        topics = api().get("/api/history")["topics"]
    except ApiError as e:
        st.error(e.message)
        return
    passed: dict[str, int] = {}
    for t in topics:
        passed[t["topic"].lower()] = max(t["percent"], passed.get(t["topic"].lower(), 0))
    in_plan = {t["title"].lower() for stage in plan["stages"] for t in stage["topics"]}
    done = sum(1 for name, percent in passed.items() if name in in_plan and percent >= 70)
    m1, m2, m3 = st.columns(3)
    m1.metric("Тем в программе", plan["total_topics"])
    m2.metric("С материалами", plan["covered_topics"])
    m3.metric("Пройдено тестов (от 70%)", done)
    st.caption("📘 — материалы есть · ✅ — тест по теме пройден · ⚪ — материалов нет")
    show_plan(plan, "plan", passed)


# ------------------------------------------------------------------ история


def page_history() -> None:
    st.title("История")
    try:
        data = api().get("/api/history")
    except ApiError as e:
        st.error(e.message)
        return
    tab_topics, tab_quiz, tab_review = st.tabs(["Результаты по темам", "Тесты", "Проверки работ"])
    with tab_topics:
        if not data["topics"]:
            st.info("Пройдите первый тест — здесь появится статистика по темам.")
        for t in data["topics"]:
            st.markdown(f"**{t['topic']}** · {t['level']} · попыток: {t['attempts']}")
            st.progress(t["percent"] / 100, text=f"{t['percent']}% верных ответов")
    with tab_quiz:
        if not data["attempts"]:
            st.info("Пройденных тестов пока нет.")
        for a in data["attempts"]:
            title = f"{when(a['created_at'])} · {a['topic']} ({a['level']}) — {a['score']}/{a['total']}"
            with st.expander(title):
                show_quiz_result(a["result"], f"h-a{a['id']}")
    with tab_review:
        if not data["reviews"]:
            st.info("Проверенных работ пока нет.")
        for r in data["reviews"]:
            title = (f"{when(r['created_at'])} · оценка {r['grade']}/5 · уровень {r['cefr_level']} "
                     f"· ошибок: {r['errors']}")
            with st.expander(title):
                show_review(r["result"], f"h-r{r['id']}")


# ------------------------------------------------------------------ библиотека


def page_library() -> None:
    st.title("Библиотека материалов")
    try:
        lib = api().get("/api/library")
    except ApiError as e:
        st.error(e.message)
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Файлов", len(lib["files"]))
    c2.metric("Проиндексировано", lib["indexed_files"])
    c3.metric("Фрагментов в индексе", lib["chunks"])
    st.subheader("Источники")
    st.dataframe([{"Источник": c["source"], "Лицензия": c["license"], "Файлов": c["files"]}
                  for c in lib["collections"]], width="stretch", hide_index=True)
    st.subheader("Файлы")
    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Поиск по названию или теме", "")
    level = f2.selectbox("Уровень", ["Все", *LEVELS])
    licences = sorted({f.get("license", "") for f in lib["files"]})
    licence = f3.selectbox("Лицензия", ["Все", *licences])
    rows = []
    for f in lib["files"]:
        haystack = (f.get("title", "") + " " + " ".join(f.get("topics", []))).lower()
        if query and query.lower() not in haystack:
            continue
        if level != "Все" and level not in f.get("level", ""):
            continue
        if licence != "Все" and f.get("license") != licence:
            continue
        rows.append({"Название": f.get("title"), "Уровень": f.get("level"), "Темы": ", ".join(f.get("topics", [])),
                     "Лицензия": f.get("license"), "Автор": f.get("author"), "Фрагментов": f.get("chunks"),
                     "Источник": f.get("url"), "Файл": f.get("file")})
    st.caption(f"Показано файлов: {len(rows)}")
    st.dataframe(rows, width="stretch", hide_index=True, height=520,
                 column_config={"Источник": st.column_config.LinkColumn("Источник", display_text="открыть")})
