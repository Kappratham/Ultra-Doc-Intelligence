import time

from backend.config import settings, logger
from backend.vector_store import search_index
from backend.llm_client import embed_single, chat
from backend.guardrails import (
    apply_retrieval_guardrail,
    compute_confidence_score,
    format_guardrail_response,
)

SYSTEM_PROMPT = (
    "You are a precise logistics document assistant inside a Transportation Management System.\n\n"
    "STRICT RULES:\n"
    "1. ONLY answer using the provided document context. Never use outside knowledge.\n"
    "2. If the answer is not in the context, respond exactly: \"Not found in document.\"\n"
    "3. Keep answers concise and factual.\n"
    "4. Mention which section supports your answer.\n"
    "5. Do not guess or infer beyond what the text explicitly states.\n"
    "6. For numbers, quote them exactly as they appear."
)


def ask_question(document_id, question):
    logger.info(f"Question for {document_id}: {question[:100]}...")

    query_embedding = embed_single(question)

    retrieved_chunks = search_index(document_id, query_embedding)
    if retrieved_chunks:
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks, top sim: {retrieved_chunks[0]['similarity']:.3f}")

    passed, filtered_chunks = apply_retrieval_guardrail(retrieved_chunks)

    if not passed:
        logger.warning("Retrieval guardrail triggered")
        return {
            "answer": "Not found in document. No relevant sections found for your question.",
            "confidence": 0.0,
            "sources": [],
            "guardrail_triggered": True,
            "guardrail_reason": "No chunks passed similarity threshold",
        }

    context = _build_context(filtered_chunks)
    user_message = (
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using ONLY the context above. If not found, say \"Not found in document.\""
    )

    answer = chat(SYSTEM_PROMPT, user_message, temperature=0.1, max_tokens=500)
    logger.info(f"Answer: {answer[:100]}...")

    confidence = compute_confidence_score(filtered_chunks, answer)
    if "not found in document" in answer.lower():
        confidence = min(confidence, 0.2)
    logger.info(f"Confidence: {confidence:.3f}")

    sources = [
        {
            "chunk_index": chunk["index"],
            "text": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
            "similarity": round(chunk["similarity"], 3),
        }
        for chunk in filtered_chunks
    ]

    return format_guardrail_response(confidence, answer, sources)


def _build_context(chunks):
    parts = [f"[Section {c['index']}]:\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)