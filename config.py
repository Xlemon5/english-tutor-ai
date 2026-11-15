import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_ROOT = DATA_DIR / "documents"

# Категории = имена папок с PDF
DOC_CATEGORIES = ["communication", "grammar", "reading", "vocabulary"]

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct-q4_K_M")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

CHROMA_COLLECTION = "english_docs"
LEXICAL_DB = str(DATA_DIR / "lexical.db")

GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))