import json
import time

from backend.config import settings, logger
from backend.llm_client import chat

EXPECTED_FIELDS = [
    "shipment_id", "shipper", "consignee",
    "pickup_datetime", "delivery_datetime",
    "equipment_type", "mode",
    "rate", "currency", "weight",
    "carrier_name",
]

EXTRACTION_PROMPT = (
    "You are a logistics document data extractor.\n\n"
    "Extract EXACTLY these fields from the document. Use null for missing fields.\n\n"
    "Fields:\n"
    "- shipment_id: Shipment, load, order, or reference number\n"
    "- shipper: Origin company/party shipping the goods\n"
    "- consignee: Destination company/party receiving the goods\n"
    "- pickup_datetime: Pickup date and/or time (original format)\n"
    "- delivery_datetime: Delivery date and/or time (original format)\n"
    "- equipment_type: Trailer type (e.g., Dry Van, Reefer, Flatbed)\n"
    "- mode: Transport mode (e.g., FTL, LTL, Intermodal)\n"
    "- rate: Total rate/cost (number only, no currency symbol)\n"
    "- currency: Currency code (e.g., USD, CAD)\n"
    "- weight: Total weight with unit if mentioned\n"
    "- carrier_name: Carrier/trucking company name\n\n"
    "RULES:\n"
    "1. Return ONLY valid JSON. No markdown, no explanation.\n"
    "2. Use null for fields not found. Do not guess.\n"
    "3. Extract exactly what the document says."
)


def extract_structured_data(full_text):
    logger.info(f"Starting extraction ({len(full_text)} chars)")

    if len(full_text) > 15000:
        logger.warning(f"Truncating from {len(full_text)} to 15000 chars")
    truncated = full_text[:15000]

    user_message = f"DOCUMENT TEXT:\n{truncated}\n\nExtract the structured shipment data as JSON."

    raw_response = chat(EXTRACTION_PROMPT, user_message, temperature=0.0, max_tokens=600)
    extracted = _parse_and_validate(raw_response)

    filled = sum(1 for v in extracted.values() if v is not None)
    logger.info(f"Extraction done. Fields found: {filled}/{len(EXPECTED_FIELDS)}")

    return extracted


def _parse_and_validate(raw_response):
    cleaned = raw_response.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}") + 1

    if json_start != -1 and json_end > json_start:
        cleaned = cleaned[json_start:json_end]

    try:
        extracted = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}")
        extracted = {}

    result = {}
    for field in EXPECTED_FIELDS:
        value = extracted.get(field)
        if isinstance(value, str) and not value.strip():
            value = None
        result[field] = value

    return result