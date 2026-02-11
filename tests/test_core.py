
import pytest
import json
import os
import tempfile
from pathlib import Path

from backend.document_processor import chunk_text, parse_file
from backend.guardrails import (
    apply_retrieval_guardrail,
    compute_confidence_score,
    _compute_answer_coverage,
)
from backend.extractor import _parse_and_validate


# ── Chunking Tests ───────────────────────────────────────────────────────

class TestChunking:

    def test_short_text_single_chunk(self):

        chunks = chunk_text("This is a short document.")
        assert len(chunks) == 1
        assert chunks[0]["index"] == 0

    def test_long_text_multiple_chunks(self):
        long_text = "Word " * 500  # ~2500 characters
        chunks = chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunks_have_required_fields(self):
        chunks = chunk_text("Test document with some content here. " * 50)
        for chunk in chunks:
            assert "index" in chunk
            assert "text" in chunk
            assert "char_start" in chunk
            assert "char_end" in chunk

    def test_chunks_are_sequential(self):
        chunks = chunk_text("Content " * 200)
        indices = [c["index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_text_raises_error(self):
        with pytest.raises(ValueError):
            chunk_text("")

    def test_whitespace_only_raises_error(self):
        with pytest.raises(ValueError):
            chunk_text("   \n\n   ")


# ── File Parsing Tests ───────────────────────────────────────────────────

class TestFileParsing:

    def test_parse_txt_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("This is test content for the logistics document.")
            f.flush()
            text = parse_file(f.name)

        assert "test content" in text
        os.unlink(f.name)

    def test_parse_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            parse_file("/nonexistent/path/file.pdf")

    def test_parse_unsupported_type(self):
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            f.write(b"fake content")
            f.flush()

        with pytest.raises(ValueError, match="Unsupported"):
            parse_file(f.name)

        os.unlink(f.name)


# ── Guardrail Tests ──────────────────────────────────────────────────────

class TestGuardrails:

    def test_retrieval_guardrail_filters_low_similarity(self):
        chunks = [
            {"text": "Relevant", "index": 0, "similarity": 0.8},
            {"text": "Irrelevant", "index": 1, "similarity": 0.1},
            {"text": "Borderline", "index": 2, "similarity": 0.3},
        ]
        passed, filtered = apply_retrieval_guardrail(chunks)
        assert passed is True
        assert len(filtered) == 2  # 0.8 and 0.3 pass, 0.1 does not

    def test_retrieval_guardrail_fails_when_nothing_passes(self):
        chunks = [
            {"text": "Low", "index": 0, "similarity": 0.05},
            {"text": "Also low", "index": 1, "similarity": 0.1},
        ]
        passed, filtered = apply_retrieval_guardrail(chunks)
        assert passed is False
        assert len(filtered) == 0

    def test_confidence_score_range(self):
        chunks = [
            {"text": "The carrier rate is $1500", "index": 0, "similarity": 0.85},
        ]
        answer = "The carrier rate is $1500"
        score = compute_confidence_score(chunks, answer)
        assert 0.0 <= score <= 1.0

    def test_confidence_zero_for_empty_chunks(self):
        score = compute_confidence_score([], "Any answer")
        assert score == 0.0

    def test_high_coverage_for_grounded_answer(self):
        chunks = [{"text": "The shipment weight is 45000 pounds", "index": 0, "similarity": 0.9}]
        answer = "The shipment weight is 45000 pounds"
        coverage = _compute_answer_coverage(answer, chunks)
        assert coverage > 0.8

    def test_low_coverage_for_hallucinated_answer(self):
        chunks = [{"text": "Pickup at warehouse A", "index": 0, "similarity": 0.9}]
        answer = "The quantum flux capacitor requires recalibration immediately"
        coverage = _compute_answer_coverage(answer, chunks)
        assert coverage < 0.3


# ── Extraction Tests ─────────────────────────────────────────────────────

class TestExtraction:

    def test_parse_valid_json(self):
        raw = json.dumps({
            "shipment_id": "SHIP-001",
            "shipper": "Acme Corp",
            "consignee": None,
            "rate": "1500",
        })
        result = _parse_and_validate(raw)
        assert result["shipment_id"] == "SHIP-001"
        assert result["shipper"] == "Acme Corp"
        assert result["consignee"] is None

    def test_parse_json_with_code_fences(self):
        raw = '```json\n{"shipment_id": "123"}\n```'
        result = _parse_and_validate(raw)
        assert result["shipment_id"] == "123"

    def test_parse_invalid_json_returns_nulls(self):
        result = _parse_and_validate("This is not JSON at all")
        assert all(v is None for v in result.values())

    def test_all_expected_fields_present(self):
        result = _parse_and_validate("{}")
        expected = {
            "shipment_id", "shipper", "consignee",
            "pickup_datetime", "delivery_datetime",
            "equipment_type", "mode",
            "rate", "currency", "weight",
            "carrier_name",
        }
        assert set(result.keys()) == expected

    def test_empty_strings_become_null(self):
        raw = json.dumps({"shipper": "", "rate": "  "})
        result = _parse_and_validate(raw)
        assert result["shipper"] is None
        assert result["rate"] is None