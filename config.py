import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

APP_TITLE = "Northstar Realty AI Assistant"
APP_SUBTITLE = "A production-style real estate copilot grounded in your internal knowledge base."

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "gemini-3.5-flash"
TOP_K_RESULTS = 4
MAX_MEMORY_TURNS = 8
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
# Never hardcode secrets; only read from environment variables.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

LOGIN_USERNAME = os.getenv("LOGIN_USERNAME", "northstar")
LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD", "realty2026")

