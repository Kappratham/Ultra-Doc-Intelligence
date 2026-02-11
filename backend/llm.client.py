import os
import time
import logging

logger = logging.getLogger("ultra_doc_intel")

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
MAX_RETRIES = 3

_embed_model = None
_groq_client = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        logger.info("Loading fastembed model (first time, may take a moment)...")
        _embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")
        logger.info("Embedding model ready")
    return _embed_model


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Get one free at https://console.groq.com")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def embed_texts(texts):
    """Embed a list of texts. Returns list of float lists."""
    if PROVIDER == "groq":
        return _embed_texts_fastembed(texts)
    return _embed_texts_ollama(texts)


def embed_single(text):
    """Embed a single text. Returns one float list."""
    if PROVIDER == "groq":
        return _embed_single_fastembed(text)
    return _embed_single_ollama(text)


def chat(system_prompt, user_message, temperature=0.1, max_tokens=500):
    """Send a chat message. Returns the response string."""
    if PROVIDER == "groq":
        return _chat_groq(system_prompt, user_message, temperature, max_tokens)
    return _chat_ollama(system_prompt, user_message, temperature, max_tokens)


# -- Ollama implementations --

def _embed_texts_ollama(texts):
    import ollama
    model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    embeddings = []

    for i, text in enumerate(texts):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = ollama.embed(model=model, input=text)
                embeddings.append(response["embeddings"][0])
                if (i + 1) % 5 == 0 or i == len(texts) - 1:
                    logger.info(f"Embedded {i + 1}/{len(texts)} chunks")
                break
            except Exception as e:
                logger.warning(f"Ollama embed failed (attempt {attempt}): {e}")
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"Embedding failed. Is Ollama running? Run: ollama serve")
                time.sleep(2 ** attempt)

    return embeddings


def _embed_single_ollama(text):
    import ollama
    model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.embed(model=model, input=text)
            return response["embeddings"][0]
        except Exception as e:
            logger.warning(f"Ollama embed failed (attempt {attempt}): {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError("Embedding failed. Is Ollama running?")
            time.sleep(2 ** attempt)


def _chat_ollama(system_prompt, user_message, temperature, max_tokens):
    import ollama
    model = os.getenv("CHAT_MODEL", "llama2")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Ollama chat failed (attempt {attempt}): {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError("Chat failed. Is Ollama running?")
            time.sleep(2 ** attempt)


# -- Groq + fastembed implementations --

def _embed_texts_fastembed(texts):
    model = _get_embed_model()
    results = list(model.embed(texts))
    embeddings = [r.tolist() for r in results]
    logger.info(f"Embedded {len(embeddings)} chunks via fastembed")
    return embeddings


def _embed_single_fastembed(text):
    model = _get_embed_model()
    results = list(model.embed([text]))
    return results[0].tolist()


def _chat_groq(system_prompt, user_message, temperature, max_tokens):
    client = _get_groq_client()
    model = os.getenv("CHAT_MODEL", "llama-3.1-8b-instant")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Groq chat failed (attempt {attempt}): {e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Groq API failed: {e}")
            time.sleep(2 ** attempt)