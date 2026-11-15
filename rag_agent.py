import json
import requests
import sqlite3
import fitz  # PyMuPDF для чтения PDF
import os
import re

# Отключаем надоедливую телеметрию ChromaDB
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import chromadb

from config import (
    OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL,
    CHROMA_COLLECTION, LEXICAL_DB, DOCUMENTS_ROOT, DOC_CATEGORIES,
)
from verification import VerificationLayer


class RAGAgent:
    def __init__(self):
        # Локальная векторная база
        persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.verifier = VerificationLayer()
        self._init_lexical_db()

    # ========== SQLITE: лексический поиск ==========

    def _init_lexical_db(self):
        os.makedirs(os.path.dirname(LEXICAL_DB), exist_ok=True)
        conn = sqlite3.connect(LEXICAL_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source TEXT, page INTEGER, category TEXT, content TEXT
            )
        """)
        conn.commit()
        conn.close()

    # ========== ЭМБЕДДИНГИ ==========

    def _embed(self, text: str):
        r = requests.post(f"{OLLAMA_BASE_URL}/api/embed",
                          json={"model": EMBEDDING_MODEL, "input": text}, timeout=60)
        if r.status_code != 200:
            raise Exception(f"Ollama embed error: {r.text}")
        d = r.json()
        return d.get("embeddings", [d.get("embedding")])[0]

    # ========== ИНДЕКСАЦИЯ PDF ==========

    def index_all_documents(self, clear_first=True):
        """Сканирует data/documents/{communication,grammar,reading,vocabulary}/*.pdf"""
        if clear_first:
            try:
                self.client.delete_collection(CHROMA_COLLECTION)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
            conn = sqlite3.connect(LEXICAL_DB)
            conn.execute("DELETE FROM chunks")
            conn.commit()
            conn.close()

        total_files, total_chunks = 0, 0
        report = []
        for category in DOC_CATEGORIES:
            folder = os.path.join(DOCUMENTS_ROOT, category)
            if not os.path.isdir(folder):
                report.append(f"Папка не найдена: {folder}")
                continue
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith(".pdf"):
                    continue
                path = os.path.join(folder, fname)
                try:
                    n = self._index_pdf(path, fname, category)
                    total_files += 1
                    total_chunks += n
                    report.append(f"{category}/{fname}: {n} чанков")
                except Exception as e:
                    report.append(f"{category}/{fname}: {e}")
        return {"files": total_files, "chunks": total_chunks, "report": report}

    def _index_pdf(self, path, fname, category):
        doc = fitz.open(path)
        added = 0
        for page_num in range(len(doc)):
            text = doc[page_num].get_text()
            for i, para in enumerate(text.split("\n")):
                clean = para.strip()
                if len(clean) < 40:  # отбрасываем короткие обрывки
                    continue
                chunk_id = f"{category}__{fname}__p{page_num+1}__c{i}"
                emb = self._embed(clean)
                meta = {"source": fname, "page": page_num + 1,
                        "category": category, "content": clean}
                self.collection.upsert(ids=[chunk_id], embeddings=[emb],
                                       metadatas=[meta], documents=[clean])
                conn = sqlite3.connect(LEXICAL_DB)
                conn.execute(
                    "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?)",
                    (chunk_id, fname, page_num + 1, category, clean))
                conn.commit()
                conn.close()
                added += 1
        doc.close()
        return added

    # ========== ПОИСК ==========

    def lexical_search(self, query, top_k=5):
        """Точное вхождение в текст чанков (LIKE)"""
        conn = sqlite3.connect(LEXICAL_DB)
        rows = conn.execute(
            "SELECT source, page, category, content FROM chunks WHERE content LIKE ? LIMIT ?",
            (f"%{query}%", top_k)).fetchall()
        conn.close()
        out = "Лексический поиск (точное вхождение):\n"
        if not rows:
            return out + "Ничего не найдено.\n"
        for src, pg, cat, content in rows:
            out += f"- [{cat} | {src}, стр.{pg}] {content[:200]}\n"
        return out

    def semantic_search(self, query, top_k=5):
        """Смысловой поиск по эмбеддингам"""
        try:
            emb = self._embed(query)
            res = self.collection.query(query_embeddings=[emb], n_results=top_k,
                                        include=["metadatas", "distances"])
            out = "Смысловой поиск (по смыслу):\n"
            metas = res["metadatas"][0]
            dists = res["distances"][0]
            if not metas:
                return out + "Ничего не найдено.\n"
            for m, d in zip(metas, dists):
                out += (f"- [{m['category']} | {m['source']}, стр.{m['page']}] "
                        f"{m['content'][:200]} (сходство: {1-d:.2f})\n")
            return out
        except Exception as e:
            return f"Ошибка: {e}"

    def hybrid_search(self, query):
        """Гибридный поиск = лексический + смысловой"""
        return self.lexical_search(query) + "\n" + self.semantic_search(query)

    # ========== ЧАТ С РЕПЕТИТОРОМ ==========

    @staticmethod
    def _normalized_text(text):
        return " ".join(str(text or "").casefold().split())

    @staticmethod
    def _json_payload(text):
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            return json.loads(text)
        except (TypeError, json.JSONDecodeError):
            match = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None

    def _search_context_chunks(self, query, top_k=8, min_similarity=0.30):
        count = self.collection.count()
        if count == 0:
            return []

        embedding = self._embed(query)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["metadatas", "distances"],
        )
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        chunks = []
        for chunk_id, metadata, distance in zip(ids, metadatas, distances):
            similarity = 1 - distance
            if similarity < min_similarity:
                continue
            chunks.append({**metadata, "id": chunk_id, "similarity": similarity})
        expanded = []
        seen_pages = set()
        for chunk in chunks:
            page_key = (chunk.get("category"), chunk.get("source"), chunk.get("page"))
            if page_key in seen_pages:
                continue
            seen_pages.add(page_key)
            chunk["content"] = self._page_context(chunk)
            expanded.append(chunk)
        return expanded

    def ask_tutor(self, question, user_name="Ученик"):
        fallback = (
            "Не удалось найти подтверждённую информацию по этому вопросу."
        )
        if not question or not question.strip():
            return "Введите вопрос по учебным материалам."

        try:
            chunks = self._search_context_chunks(question, top_k=4)
            if not chunks:
                return fallback

            retrieved = {chunk["id"]: chunk for chunk in chunks}
            context_items = []
            for chunk in chunks:
                context_items.append(
                    f"SOURCE_ID: {chunk['id']}\n"
                    f"File: {chunk['source']}\nPage: {chunk['page']}\n"
                    f"Text: {chunk['content'][:1500]}"
                )
            context = "\n\n".join(context_items)

            prompt = f"""Ты — репетитор английского языка. Отвечай ученику по-русски.
Ученик: {user_name}.

Используй только приведённые фрагменты. Разбей ответ на короткие атомарные утверждения: одно проверяемое утверждение на элемент claims.
Каждое утверждение должно быть коротким и атомарным. Верни text как точный краткий перевод claim_source на русский, без новых деталей;
claim_source — полный проверяемый факт на языке цитаты, сохрани ключевые термины и смысл источника; claim_english — тот же смысл простым английским.
Также укажи source_id и дословную quote, содержащую достаточно контекста для подтверждения.
Если фрагменты не отвечают на вопрос, верни abstain=true и пустой claims. Иначе abstain=false.
Не добавляй фразу об отсутствии информации к содержательному ответу. Не добавляй источники от себя.

Фрагменты:
{context}

Вопрос ученика: {question}

Верни только JSON: {{"abstain":false,"claims":[{{"text":"...","claim_source":"...","claim_english":"...","source_id":"точный SOURCE_ID","quote":"дословная цитата"}}]}}."""

            candidates = []
            temperatures = (0.0, 0.35, 0.7)
            strict_prompt = prompt + """

СТРОГАЯ ПРОВЕРЯЕМАЯ ПОВТОРНАЯ ПОПЫТКА:
- В каждом claim ровно один факт.
- claim_source и text должны сообщать один и тот же факт без дополнительных деталей.
- quote должен целиком подтверждать этот факт; если цитата не подтверждает деталь, не включай ее.
- Для русскоязычной цитаты формулируй claim_source по-русски, для англоязычной — по-английски.
- Если ни одного подтверждаемого факта нет, верни abstain=true и пустой claims.
"""
            for temperature in temperatures:
                sample = None
                for attempt in range(2):
                    try:
                        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
                            "model": LLM_MODEL,
                            "prompt": prompt if attempt == 0 else strict_prompt,
                            "format": "json",
                            "stream": False,
                            "options": {"temperature": temperature if attempt == 0 else 0.0, "top_p": 0.9},
                        }, timeout=300)
                        response.raise_for_status()
                    except requests.RequestException as error:
                        print(f"Ошибка генерации ответа: {error}")
                        return fallback

                    payload = self._json_payload(response.json().get("response", ""))
                    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
                        self.verifier.record_failure("unsupported_claim", "chat", reason="invalid_claim_schema")
                        continue
                    if payload.get("abstain") is True:
                        sample = {"abstain": True, "claims": []}
                        break
                    claims = payload["claims"]
                    if not claims or any(not isinstance(claim, dict) for claim in claims):
                        self.verifier.record_failure("unsupported_claim", "chat", reason="empty_or_invalid_claims")
                        continue
                    ok, _score = self.verifier.verify_claims(claims, retrieved)
                    if ok:
                        sample = {"abstain": False, "claims": claims}
                        break
                if sample is None:
                    return fallback
                candidates.append(sample)

            consistent, agreed_claims = self.verifier.verify_consistency(candidates)
            if not consistent or not agreed_claims:
                return fallback

            citations = []
            rendered_claims = []
            for claim in agreed_claims:
                text = claim.get("_verified_text", claim["text"]).strip()
                if text not in rendered_claims:
                    rendered_claims.append(text)
                source = retrieved[claim["source_id"]]
                citation = f"[Файл: {source['source']}, стр. {source['page']}]"
                if citation not in citations:
                    citations.append(citation)
            return " ".join(rendered_claims) + "\n\nИсточники: " + ", ".join(citations)

        except Exception as e:
            print(f"Ошибка в ask_tutor: {e}")
            return fallback


    # ========== RAG-ТЕСТ (новый метод, заменяет старый generate_quiz) ==========

    def _relevant_chunks(self, topic, n=8):
        """Ищет содержательные фрагменты по нескольким формулировкам темы."""
        topic_l = (topic or "").lower()
        category = None
        for c in ("grammar", "vocabulary", "reading", "communication"):
            if c in topic_l:
                category = c
                break
        try:
            search_phrases = [
                topic,
                f"English {topic} rules explanations examples and correct usage",
                f"How {topic} is explained and used in the lesson",
                f"English {topic} examples and correct forms",
            ]
            candidates = {}
            candidate_count = min(max(n * 8, 40), 100)
            for phrase in search_phrases:
                embedding = self._embed(phrase)
                result = self.collection.query(
                    query_embeddings=[embedding],
                    n_results=min(candidate_count, self.collection.count()),
                    where={"category": category} if category else None,
                    include=["metadatas", "distances"],
                )
                ids = result.get("ids", [[]])[0]
                metadatas = result.get("metadatas", [[]])[0]
                distances = result.get("distances", [[]])[0]
                for chunk_id, metadata, distance in zip(ids, metadatas, distances):
                    similarity = 1 - distance
                    content = (metadata.get("content") or "").strip()
                    words = re.findall(r"[A-Za-zА-Яа-яЁё]+", content)
                    if similarity < 0.30 or len(content) < 45 or len(words) < 6:
                        continue
                    if re.match(r"^(?:A1|A2|B1|B2)\s*\|", content, re.IGNORECASE):
                        continue
                    if re.match(r"^(?:Topic:|Read the conversation)", content, re.IGNORECASE):
                        continue
                    if re.match(r"^\d+\.\s*GRAMMAR\s*:", content, re.IGNORECASE):
                        continue
                    if re.match(r"^(?:teacher note|answer key|instructions?)\s*:", content, re.IGNORECASE):
                        continue
                    if re.match(
                        r"^(?:use|write|complete|choose|select|match|fill in|read|answer|underline|circle)\b",
                        content,
                        re.IGNORECASE,
                    ):
                        continue
                    if re.match(r"^(?:module|language focus)\s*[:#]", content, re.IGNORECASE):
                        continue

                    key = chunk_id
                    source_name = (metadata.get("source") or "").casefold()
                    quality_score = similarity + (0.12 if "теория" in source_name or "theory" in source_name else 0.0)
                    if key not in candidates or quality_score > candidates[key]["rank_score"]:
                        candidates[key] = {
                            **metadata,
                            "id": chunk_id,
                            "similarity": similarity,
                            "rank_score": quality_score,
                        }

            ranked = sorted(candidates.values(), key=lambda item: item["rank_score"], reverse=True)
            # PDF indexing stores one line per Chroma chunk. Expand a selected
            # result to its page so the generator sees the rule and its nearby
            # examples instead of an isolated, often unanswerable line.
            expanded = []
            seen_pages = set()
            for item in ranked:
                page_key = (item.get("category"), item.get("source"), item.get("page"))
                if page_key in seen_pages:
                    continue
                seen_pages.add(page_key)
                item["content"] = self._page_context(item)
                expanded.append(item)
                if len(expanded) >= n:
                    break
            return expanded
        except Exception:
            return []

    def _page_context(self, chunk):
        conn = sqlite3.connect(LEXICAL_DB)
        rows = conn.execute(
            "SELECT id, content FROM chunks WHERE source = ? AND page = ? AND category = ?",
            (chunk.get("source"), chunk.get("page"), chunk.get("category")),
        ).fetchall()
        conn.close()
        if not rows:
            return chunk.get("content", "")
        def order(row):
            match = re.search(r"__c(\d+)$", row[0])
            return int(match.group(1)) if match else 0
        return "\n".join(content for _chunk_id, content in sorted(rows, key=order))

    def generate_quiz_with_sources(self, topic, num_questions=3):
        try:
            num_questions = int(num_questions)
        except (TypeError, ValueError):
            return []
        if not 1 <= num_questions <= 10:
            return []

        chunks = self._relevant_chunks(topic, n=num_questions)
        if not chunks:
            return []

        retrieved = {chunk["id"]: chunk for chunk in chunks}
        context_items = []
        for m in chunks:
            context_items.append(
                f"SOURCE_ID: {m['id']}\n"
                f"File: {m['source']}\nPage: {m['page']}\n"
                f"Text: {m['content'][:2200]}"
            )
        context = "\n\n".join(context_items)

        # Специфика каждой категории
        category_instructions = {
            "grammar": (
                "Ты тестируешь знание грамматики. "
                "По возможности чередуй типы вопросов, но не создавай больше вопросов, чем запрошено. "
                "Возможные типы: "
                "1. ФОРМА: 'Как образуется...?' / 'Какая форма глагола правильная в...?' "
                "2. ИСПОЛЬЗОВАНИЕ: 'В каком предложении верно употреблено...?' "
                "3. КОНТРАСТ: 'Чем отличается X от Y?' / 'В каком случае используется X, а не Y?' "
                "4. ПОИСК ОШИБКИ: 'В каком предложении допущена грамматическая ошибка?' "
                "5. ПОРЯДОК СЛОВ: 'Какое предложение построено правильно?' "
                "6. АРТИКЛИ/ПРЕДЛОГИ: 'Какой артикль/предлог нужен в...?' "
                "ЗАПРЕЩЕНО задавать три вопроса подряд в форме 'Какое время используется для...?'"
            ),
            "vocabulary": (
                "Ты тестируешь знание лексики. "
                "Чередуй типы вопросов: "
                "1. Значение слова ('Что означает слово...?') "
                "2. Синонимы/антонимы ('Какой синоним к...?') "
                "3. Употребление в контексте ('Какое слово подходит в пропуск...?') "
                "4. Устойчивые выражения ('Что означает фраза...?') "
                "5. Словообразование ('Какая форма слова правильная...?')"
            ),
            "communication": (
                "Ты тестируешь практические навыки общения (Functional Language). "
                "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО: "
                "1. Задавать вопросы по грамматике (никаких вопросов про времена, артикли, спряжения). "
                "2. Задавать абстрактные вопросы ('Какая стратегия рекомендуется...', 'Что нужно делать, чтобы...'). "
                "Вместо этого задавай вопросы о КОНКРЕТНЫХ фразах или ситуациях из текста. "
                "Примеры ХОРОШИХ вопросов: "
                "- 'Какую фразу из текста лучше всего использовать, чтобы вежливо отказаться?' "
                "- 'Что означает выражение [вставь конкретную фразу из текста] в данном диалоге?' "
                "- 'Согласно тексту, как следует реагировать, если собеседник не расслышал вопрос?'"
            ),
        }

        topic_lower = (topic or "").lower()
        category_hint = ""
        for cat, instruction in category_instructions.items():
            if cat in topic_lower:
                category_hint = f"\nСПЕЦИФИКА КАТЕГОРИИ '{topic.upper()}':\n{instruction}\n"
                break

        prompt = f"""Ты — составитель тестов по английскому языку.
Сгенерируй РОВНО {num_questions} вопросов с четырьмя вариантами A, B, C, D.{category_hint}
Используй только фрагменты ниже. У каждого вопроса должен быть ровно один правильный вариант.
Для каждого вопроса укажи точный source_id из фрагмента, дословную цитату evidence, краткое утверждение question_claim
и option_claims для всех A/B/C/D. Все эти утверждения пиши на том же языке, что и цитата evidence.
Утверждение варианта должно быть полным фактом, который можно проверить по цитате.
Если не можешь сделать ровно {num_questions} подтверждаемых вопросов, верни {{"questions":[]}}.

ФРАГМЕНТЫ:
{context}

Формат JSON:
{{"questions":[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"correct":"A","source_id":"точный SOURCE_ID","evidence":"дословная цитата","question_claim":"...","option_claims":{{"A":"...","B":"...","C":"...","D":"..."}}}}]}}"""

        for attempt in range(2):
            current_prompt = prompt
            if attempt:
                current_prompt += (
                    f"\nПОВТОРНАЯ ПОПЫТКА: верни РОВНО {num_questions} валидных вопросов. "
                    "Не меняй SOURCE_ID и цитаты. Если точное число невозможно, верни пустой questions."
                )
            try:
                r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
                    "model": LLM_MODEL,
                    "prompt": current_prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0 if attempt == 0 else 0.2, "top_p": 0.9},
                }, timeout=300)
                r.raise_for_status()
            except requests.RequestException as e:
                print(f"Ошибка генерации теста: {e}")
                return []

            candidates = self._parse_quiz_with_sources(
                r.json().get("response", ""), retrieved, num_questions=num_questions
            )
            if candidates is not None and self.verifier.verify_quiz_items(
                candidates, retrieved, requested_count=num_questions
            ):
                return [
                    {
                        "question": item["question"],
                        "options": item["options"],
                        "correct": item["correct"],
                        "source": retrieved[item["source_id"]]["source"],
                        "page": retrieved[item["source_id"]]["page"],
                    }
                    for item in candidates
                ]
        return []

    def _parse_quiz_with_sources(self, text, retrieved, num_questions=10):
        """Strictly parses exact-count quiz JSON and maps verified IDs to retrieved chunks."""
        payload = self._json_payload(text)
        if isinstance(payload, dict):
            candidates = payload.get("questions", [])
        else:
            self.verifier.record_failure("wrong_count", "quiz", reason="invalid_json_schema")
            return None
        if not isinstance(candidates, list) or len(candidates) != num_questions:
            self.verifier.record_failure(
                "wrong_count", "quiz", requested=num_questions,
                received=len(candidates) if isinstance(candidates, list) else None,
            )
            return None

        questions = []
        seen = set()
        for item in candidates:
            if not isinstance(item, dict):
                self.verifier.record_failure("unsupported_claim", "quiz", reason="question_not_object")
                return None

            question = item.get("question")
            raw_options = item.get("options")
            correct = item.get("correct")
            source_id = item.get("source_id")
            evidence = item.get("evidence")
            if not isinstance(question, str) or not question.strip() or not isinstance(raw_options, dict) or not isinstance(correct, str):
                self.verifier.record_failure("unsupported_claim", "quiz", reason="invalid_question_schema")
                return None
            options = {str(key).strip().upper(): value for key, value in raw_options.items()}
            if set(options) != {"A", "B", "C", "D"}:
                self.verifier.record_failure("unsupported_claim", "quiz", reason="invalid_option_set")
                return None
            if any(not isinstance(options[key], str) or not options[key].strip() for key in options):
                self.verifier.record_failure("unsupported_claim", "quiz", reason="empty_option")
                return None
            correct = correct.strip().upper()
            if correct not in options:
                self.verifier.record_failure("unsupported_claim", "quiz", reason="invalid_correct_option")
                return None
            if not isinstance(source_id, str) or source_id not in retrieved:
                self.verifier.record_failure("bad_citation", "quiz", reason="source_id_not_retrieved")
                return None
            if not isinstance(evidence, str) or not isinstance(item.get("question_claim"), str):
                self.verifier.record_failure("bad_citation", "quiz", reason="missing_evidence_or_claim")
                return None
            raw_option_claims = item.get("option_claims")
            if not isinstance(raw_option_claims, dict):
                self.verifier.record_failure("unsupported_claim", "quiz", reason="missing_option_claims")
                return None
            option_claims = {str(key).strip().upper(): value for key, value in raw_option_claims.items()}
            if set(option_claims) != {"A", "B", "C", "D"}:
                self.verifier.record_failure("unsupported_claim", "quiz", reason="invalid_option_claim_set")
                return None
            source_chunk = retrieved[source_id]
            normalized_evidence = self._normalized_text(evidence)
            normalized_content = self._normalized_text(source_chunk.get("content", ""))
            if len(normalized_evidence) < 12 or normalized_evidence not in normalized_content:
                self.verifier.record_failure("citation_drift", "quiz", reason="quote_not_in_source")
                return None

            question_key = self._normalized_text(question)
            if question_key in seen:
                self.verifier.record_failure("unsupported_claim", "quiz", reason="duplicate_question")
                return None
            seen.add(question_key)
            questions.append({
                "question": question.strip(),
                "options": {key: options[key].strip() for key in ("A", "B", "C", "D")},
                "correct": correct,
                "source_id": source_id,
                "evidence": evidence,
                "question_claim": item["question_claim"].strip(),
                "option_claims": option_claims,
            })
        return questions if len(questions) == num_questions else None

    # ========== СТАТИСТИКА ==========

    def stats(self):
        return {"documents": self.collection.count()}
