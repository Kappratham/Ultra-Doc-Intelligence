import json
import time
import ollama

from backend.config import settings, logger

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
    "1. Return ONLY valid JSON. No markdown, no explanation, no extra text.\n"
    "2. Use null for fields not found. Do not guess.\n"
    "3. Extract exactly what the document says.\n\n"
    "Example output format:\n"
    '{"shipment_id": "LD-001", "shipper": "Acme Corp", "consignee": null, '
    '"pickup_datetime": null, "delivery_datetime": null, "equipment_type": null, '
    '"mode": "FTL", "rate": "1500", "currency": "USD", "weight": null, "carrier_name": null}'
)


def extract_structured_data(full_text):
    logger.info(f"Starting extraction ({len(full_text)} chars)")

    if len(full_text) > 15000:
        logger.warning(f"Document truncated from {len(full_text)} to 15000 chars")
    truncated = full_text[:15000]

    user_message = f"DOCUMENT TEXT:\n{truncated}\n\nExtract the structured shipment data as JSON."

    raw_response = _call_extraction_llm(user_message)
    extracted = _parse_and_validate(raw_response)

    filled = sum(1 for v in extracted.values() if v is not None)
    logger.info(f"Extraction complete. Fields found: {filled}/{len(EXPECTED_FIELDS)}")

    return extracted


def _call_extraction_llm(user_message):
    for attempt in range(1, settings.MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=settings.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                options={
                    "temperature": 0.0,
                    "num_predict": 600,
                },
            )
            return response["message"]["content"].strip()

        except Exception as e:
            logger.warning(f"Extraction LLM failed (attempt {attempt}): {e}")
            if attempt == settings.MAX_RETRIES:
                raise RuntimeError(
                    f"Extraction failed. Is Ollama running? Run: ollama serve"
                )
            time.sleep(2 ** attempt)


def _parse_and_validate(raw_response):
    cleaned = raw_response.strip()

    # Remove markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    # Try to find JSON in the response
    json_start = cleaned.find("{")
    json_end = cleaned.rfind("}") + 1

    if json_start != -1 and json_end > json_start:
        cleaned = cleaned[json_start:json_end]

    try:
        extracted = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extraction JSON: {e}")
        extracted = {}

    # Ensure all fields present, normalise empty strings to null
    result = {}
    for field in EXPECTED_FIELDS:
        value = extracted.get(field)
        if isinstance(value, str) and not value.strip():
            value = None
        result[field] = value

    return result