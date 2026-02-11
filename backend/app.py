
import uuid
import shutil
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import settings, logger
from backend.database import document_db
from backend.document_processor import process_document
from backend.vector_store import save_index
from backend.rag_engine import ask_question
from backend.extractor import extract_structured_data


# ── Response Models ──────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    message: str


class SourceChunk(BaseModel):
    chunk_index: int
    text: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[SourceChunk]
    guardrail_triggered: bool
    guardrail_reason: Optional[str] = None


class ExtractedData(BaseModel):
    shipment_id: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    pickup_datetime: Optional[str] = None
    delivery_datetime: Optional[str] = None
    equipment_type: Optional[str] = None
    mode: Optional[str] = None
    rate: Optional[str] = None
    currency: Optional[str] = None
    weight: Optional[str] = None
    carrier_name: Optional[str] = None


class ExtractResponse(BaseModel):
    document_id: str
    filename: str
    extracted_data: ExtractedData


class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    version: str


# ── Request Models ───────────────────────────────────────────────────────

class AskRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=50)
    question: str = Field(..., min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only.")
        return v.strip()


class ExtractRequest(BaseModel):
    document_id: str = Field(..., min_length=1, max_length=50)


# ── App Setup ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Ultra Doc-Intelligence",
    description="AI assistant for logistics document Q&A and structured extraction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Logging Middleware ───────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} → "
        f"{response.status_code} ({duration:.2f}s)"
    )

    return response


# ── Global Exception Handler ────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again."},
    )


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    # Validate file type
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{file_ext}'. "
                f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            ),
        )

    # Validate file size
    file_content = await file.read()
    file_size = len(file_content)
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    if file_size > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large ({file_size / 1024 / 1024:.1f}MB). "
                f"Maximum: {settings.MAX_FILE_SIZE_MB}MB"
            ),
        )

    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    # Save to disk
    document_id = str(uuid.uuid4())[:8]
    safe_filename = f"{document_id}{file_ext}"
    file_path = settings.UPLOAD_DIR / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file.")

    # Process: parse → chunk → embed
    try:
        chunks, embeddings, full_text = process_document(str(file_path))
    except ValueError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        file_path.unlink(missing_ok=True)
        logger.error(f"Document processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Document processing failed.")

    # Store index and metadata
    save_index(document_id, chunks, embeddings)
    document_db.save_document(
        document_id=document_id,
        filename=file.filename,
        file_path=str(file_path),
        full_text=full_text,
        chunk_count=len(chunks),
        file_size_bytes=file_size,
    )

    logger.info(f"Document uploaded successfully: {document_id} ({file.filename})")

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunks_created=len(chunks),
        message="Document uploaded and indexed successfully.",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not document_db.document_exists(request.document_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.document_id}' not found. Upload it first.",
        )

    try:
        result = ask_question(request.document_id, request.question)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document index not found.")
    except Exception as e:
        logger.error(f"Error during Q&A: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing your question.")

    return result


@app.post("/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest):
    doc = document_db.get_document(request.document_id)

    if doc is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{request.document_id}' not found. Upload it first.",
        )

    try:
        extracted = extract_structured_data(doc["full_text"])
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Extraction failed.")

    return ExtractResponse(
        document_id=request.document_id,
        filename=doc["filename"],
        extracted_data=ExtractedData(**extracted),
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        documents_loaded=document_db.get_document_count(),
        version="1.0.0",
    )