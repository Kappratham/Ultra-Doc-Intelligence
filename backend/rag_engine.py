import time
import ollama

from backend.config import settings, logger
from backend.vector_store import search_index
from backend.guardrails import (
    apply_retrieval_guardrail,
    compute_confidence_score,
    format_guardrail_response,
)

SYSTEM_PROMPT = """You are a precise logistics document assistant inside a Transportation Management System (TMS).

STRICT RULES:
1. ONLY answer using the provided document context below. Never use outside knowledge.
2. If the answer is not in the context, respond exactly: "Not found in document."
3. Keep answers concise and factual.
4. Mention which section of the context supports your answer.
5. Do not guess, assume, or infer beyond what the text explicitly states.
6. For numbers (rates, weights, dates), quote them exactly as they appear.
"""


def ask_question(document_id: str, question: str) -> dict:
    logger.info(f"Question for {document_id}: {question[:100]}...")

    # Step 1: Embed the question
    query_embedding = _embed_question(question)

    # Step 2: Retrieve relevant chunks
    retrieved_chunks = search_index(document_id, query_embedding)
    if retrieved_chunks:
        logger.info(
            f"Retrieved {len(retrieved_chunks)} chunks, "
            f"top sim: {retrieved_chunks[0]['similarity']:.3f}"
        )

    # Step 3: Apply retrieval guardrail
    passed, filtered_chunks = apply_retrieval_guardrail(retrieved_chunks)

    if not passed:
        logger.warning("Retrieval guardrail triggered — no chunks above threshold")
        return {
            "answer": "Not found in document. No relevant sections were found for your question.",
            "confidence": 0.0,
            "sources": [],
            "guardrail_triggered": True,
            "guardrail_reason": "No chunks passed similarity threshold",
        }

    # Step 4: Build grounded prompt
    context = _build_context(filtered_chunks)
    user_message = (
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using ONLY the context above. "
        f"If not found, say \"Not found in document.\""
    )

    # Step 5: Generate answer
    answer = _call_llm_with_retry(user_message)
    logger.info(f"Answer: {answer[:100]}...")

    # Step 6: Compute confidence
    confidence = compute_confidence_score(filtered_chunks, answer)

    if "not found in document" in answer.lower():
        confidence = min(confidence, 0.2)

    logger.info(f"Confidence: {confidence:.3f}")

    # Step 7: Format sources
    sources = [
        {
            "chunk_index": chunk["index"],
            "text": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
            "similarity": round(chunk["similarity"], 3),
        }
        for chunk in filtered_chunks
    ]

    # Step 8: Apply guardrails
    return format_guardrail_response(confidence, answer, sources)


def _embed_question(question: str) -> list[float]:
    for attempt in range(1, settings.MAX_RETRIES + 1):
        try:
            response = ollama.embed(
                model=settings.EMBEDDING_MODEL,
                input=question,
            )
            return response["embeddings"][0]

        except Exception as e:
            logger.warning(f"Question embedding failed (attempt {attempt}): {e}")
            if attempt == settings.MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to embed question. Is Ollama running? Run: ollama serve"
                )
            time.sleep(2 ** attempt)


def _build_context(chunks: list[dict]) -> str:
    parts = [f"[Section {c['index']}]:\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def _call_llm_with_retry(user_message: str) -> str:
    for attempt in range(1, settings.MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=settings.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                options={
                    "temperature": 0.1,
                    "num_predict": 500,
                },
            )
            return response["message"]["content"].strip()

        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt}): {e}")
            if attempt == settings.MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to get LLM response. Is Ollama running? Run: ollama serve"
                )
            time.sleep(2 ** attempt)