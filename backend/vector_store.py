
import json
import threading
import numpy as np
import faiss
from pathlib import Path

from backend.config import settings, logger

# Thread lock for FAISS operations — FAISS is not thread-safe by default
_index_lock = threading.Lock()


def save_index(
    document_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:

    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    index_path = settings.INDEX_DIR / f"{document_id}.faiss"
    chunks_path = settings.INDEX_DIR / f"{document_id}.json"

    with _index_lock:
        faiss.write_index(index, str(index_path))
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Index saved: {document_id} ({len(chunks)} chunks, dim={dimension})")


def search_index(
    document_id: str,
    query_embedding: list[float],
    top_k: int = None,
) -> list[dict]:

    if top_k is None:
        top_k = settings.TOP_K_CHUNKS

    index_path = settings.INDEX_DIR / f"{document_id}.faiss"
    chunks_path = settings.INDEX_DIR / f"{document_id}.json"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"No index found for document: {document_id}")

    with _index_lock:
        index = faiss.read_index(str(index_path))
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

    query_vector = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": chunks[idx]["text"],
            "index": chunks[idx]["index"],
            "similarity": float(score),
        })

    logger.debug(f"Search results: {len(results)} chunks for document {document_id}")
    return results


def index_exists(document_id: str) -> bool:

    return (settings.INDEX_DIR / f"{document_id}.faiss").exists()