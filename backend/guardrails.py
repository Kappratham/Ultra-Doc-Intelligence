
from backend.config import settings, logger


def apply_retrieval_guardrail(
    retrieved_chunks: list[dict],
) -> tuple[bool, list[dict]]:
    filtered = [
        chunk for chunk in retrieved_chunks
        if chunk["similarity"] >= settings.SIMILARITY_THRESHOLD
    ]

    passed = len(filtered) > 0

    logger.info(
        f"Retrieval guardrail: {len(filtered)}/{len(retrieved_chunks)} chunks "
        f"passed threshold ({settings.SIMILARITY_THRESHOLD})"
    )

    return passed, filtered


def compute_confidence_score(
    filtered_chunks: list[dict],
    answer: str,
) -> float:
    if not filtered_chunks:
        return 0.0

    # 1. Retrieval similarity
    similarities = [c["similarity"] for c in filtered_chunks]
    retrieval_score = min(sum(similarities) / len(similarities), 1.0)

    # 2. Chunk agreement
    if len(similarities) >= 2:
        score_range = max(similarities) - min(similarities)
        agreement_score = 1.0 - min(score_range, 1.0)
    else:
        agreement_score = 0.5

    # 3. Answer coverage
    coverage_score = _compute_answer_coverage(answer, filtered_chunks)

    confidence = (
        0.40 * retrieval_score
        + 0.30 * agreement_score
        + 0.30 * coverage_score
    )

    logger.debug(
        f"Confidence breakdown: retrieval={retrieval_score:.3f}, "
        f"agreement={agreement_score:.3f}, coverage={coverage_score:.3f}, "
        f"final={confidence:.3f}"
    )

    return round(confidence, 3)


def _compute_answer_coverage(answer: str, chunks: list[dict]) -> float:
    source_text = " ".join(c["text"] for c in chunks).lower()
    source_words = set(source_text.split())

    answer_words = [
        word.lower().strip(".,;:!?\"'()[]{}") 
        for word in answer.split()
        if len(word) > 3
    ]

    if not answer_words:
        return 1.0

    grounded = sum(1 for w in answer_words if w in source_words)
    return grounded / len(answer_words)


def format_guardrail_response(
    confidence: float,
    answer: str,
    sources: list[dict],
) -> dict:
    if confidence < settings.CONFIDENCE_LOW_THRESHOLD:
        logger.warning(f"Answer blocked — confidence {confidence:.3f} below threshold")
        return {
            "answer": (
                "Not found in document. The available context does not "
                "contain enough information to answer this question confidently."
            ),
            "confidence": confidence,
            "sources": sources,
            "guardrail_triggered": True,
            "guardrail_reason": "Confidence below minimum threshold",
        }

    response = {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "guardrail_triggered": False,
        "guardrail_reason": None,
    }

    if confidence < settings.CONFIDENCE_MEDIUM_THRESHOLD:
        response["answer"] = f"⚠️ Low confidence answer: {answer}"
        response["guardrail_triggered"] = True
        response["guardrail_reason"] = "Confidence is moderate — answer may be incomplete"
        logger.info(f"Low confidence warning applied: {confidence:.3f}")

    return response