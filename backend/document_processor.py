import os
import time

import fitz
import docx

from backend.config import settings, logger
from backend.llm_client import embed_texts


def parse_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    if file_size > max_size:
        raise ValueError(f"File too large ({file_size / 1024 / 1024:.1f}MB). Max: {settings.MAX_FILE_SIZE_MB}MB")

    extension = os.path.splitext(file_path)[1].lower()
    logger.info(f"Parsing: {file_path} ({extension}, {file_size} bytes)")

    parsers = {
        ".pdf": _parse_pdf,
        ".docx": _parse_docx,
        ".txt": _parse_txt,
    }

    parser = parsers.get(extension)
    if not parser:
        raise ValueError(f"Unsupported file type: {extension}")

    text = parser(file_path)
    logger.info(f"Parsed {len(text)} characters")
    return text


def _parse_pdf(file_path):
    try:
        document = fitz.open(file_path)
    except Exception as e:
        raise ValueError(f"Cannot open PDF: {e}")

    pages = []
    for page_num in range(len(document)):
        text = document[page_num].get_text()
        if text.strip():
            pages.append(text)
    document.close()

    if not pages:
        raise ValueError("PDF has no extractable text. Might be scanned/image-only.")

    return "\n\n".join(pages)


def _parse_docx(file_path):
    try:
        doc = docx.Document(file_path)
    except Exception as e:
        raise ValueError(f"Cannot open DOCX: {e}")

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    if not paragraphs:
        raise ValueError("DOCX has no extractable text.")

    return "\n\n".join(paragraphs)


def _parse_txt(file_path):
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            if content.strip():
                return content
        except UnicodeDecodeError:
            continue
    raise ValueError("Cannot read TXT file with any supported encoding.")


def chunk_text(text):
    text = text.strip()
    if not text:
        raise ValueError("Cannot chunk empty text.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + settings.CHUNK_SIZE, len(text))

        if end < len(text):
            adjusted = _find_sentence_break(text, start, end)
            if adjusted > start:
                end = adjusted

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append({
                "index": len(chunks),
                "text": chunk_content,
                "char_start": start,
                "char_end": end,
            })

        next_start = end - settings.CHUNK_OVERLAP
        start = next_start if next_start > start else end

    logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
    return chunks


def _find_sentence_break(text, start, end):
    search_start = start + int((end - start) * 0.7)
    search_region = text[search_start:end]

    for delimiter in ["\n\n", "\n", ". ", "? ", "! ", "; "]:
        last_break = search_region.rfind(delimiter)
        if last_break != -1:
            return search_start + last_break + len(delimiter)
    return end


def create_embeddings(chunks):
    logger.info(f"Creating embeddings for {len(chunks)} chunks")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)
    logger.info(f"All {len(embeddings)} embeddings created")
    return embeddings


def process_document(file_path):
    logger.info(f"Processing document: {file_path}")
    full_text = parse_file(file_path)
    chunks = chunk_text(full_text)
    embeddings = create_embeddings(chunks)
    logger.info(f"Done: {len(full_text)} chars -> {len(chunks)} chunks -> {len(embeddings)} embeddings")
    return chunks, embeddings, full_text