import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

# ── Load .env ────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Pre-compute paths ───────────────────────────────────
_UPLOAD_DIR = BASE_DIR / "uploads"
_INDEX_DIR = BASE_DIR / "indices"
_DB_PATH = BASE_DIR / "documents.db"

_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_INDEX_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):

    # ── Paths ────────────────────────────────────────────
    BASE_DIR: Path = BASE_DIR
    UPLOAD_DIR: Path = _UPLOAD_DIR
    INDEX_DIR: Path = _INDEX_DIR
    DB_PATH: Path = _DB_PATH

    # ── Ollama ───────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    CHAT_MODEL: str = "llama2"

    # ── Chunking ─────────────────────────────────────────
    CHUNK_SIZE: int = Field(default=500, ge=100, le=2000)
    CHUNK_OVERLAP: int = Field(default=100, ge=0, le=500)

    # ── Retrieval ────────────────────────────────────────
    TOP_K_CHUNKS: int = Field(default=5, ge=1, le=20)
    SIMILARITY_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)

    # ── Confidence ───────────────────────────────────────
    CONFIDENCE_LOW_THRESHOLD: float = 0.4
    CONFIDENCE_MEDIUM_THRESHOLD: float = 0.7

    # ── API Settings ─────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 20
    MAX_QUESTION_LENGTH: int = 1000
    MAX_RETRIES: int = 3
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt"}

    # ── Logging ──────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def overlap_must_be_less_than_chunk_size(cls, v, info):
        chunk_size = info.data.get("CHUNK_SIZE", 500)
        if v >= chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({v}) must be less than CHUNK_SIZE ({chunk_size})"
            )
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# ── Instantiate ──────────────────────────────────────────
settings = Settings()


# ── Logging ──────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    log = logging.getLogger("ultra_doc_intel")
    log.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))

    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)

    return log


logger = setup_logging()