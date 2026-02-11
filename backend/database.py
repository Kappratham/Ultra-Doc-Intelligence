

import sqlite3
import json
import threading
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional

from backend.config import settings, logger


class DocumentDatabase:

    def __init__(self, db_path: str = None):
        self.db_path = str(db_path or settings.DB_PATH)
        self._local = threading.local()
        self._initialise_schema()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection

    @contextmanager
    def _cursor(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def _initialise_schema(self):
        with self._cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    file_size_bytes INTEGER,
                    created_at TEXT NOT NULL,
                    status TEXT DEFAULT 'active'
                )
            """)
        logger.info("Document database initialised")

    def save_document(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        full_text: str,
        chunk_count: int,
        file_size_bytes: int = 0,
    ) -> None:

        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents 
                (document_id, filename, file_path, full_text, 
                 chunk_count, file_size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    filename,
                    file_path,
                    full_text,
                    chunk_count,
                    file_size_bytes,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        logger.info(f"Document saved: {document_id} ({filename})")

    def get_document(self, document_id: str) -> Optional[dict]:

        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM documents WHERE document_id = ? AND status = 'active'",
                (document_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return dict(row)

    def document_exists(self, document_id: str) -> bool:

        return self.get_document(document_id) is not None

    def list_documents(self) -> list[dict]:

        with self._cursor() as cursor:
            cursor.execute(
                "SELECT document_id, filename, chunk_count, created_at "
                "FROM documents WHERE status = 'active' "
                "ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def delete_document(self, document_id: str) -> bool:

        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE documents SET status = 'deleted' WHERE document_id = ?",
                (document_id,),
            )
            return cursor.rowcount > 0

    def get_document_count(self) -> int:

        with self._cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM documents WHERE status = 'active'"
            )
            return cursor.fetchone()["count"]


# ── Singleton instance ───────────────────────────────────
document_db = DocumentDatabase()