
import streamlit as st
import requests
import os

API_BASE_URL = os.environ.get("API_BASE_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "http://localhost:8000"
if "onrender.com" in API_BASE_URL and "frontend" in API_BASE_URL:
    API_BASE_URL = API_BASE_URL.replace("frontend", "backend")
REQUEST_TIMEOUT = 60

# ── Page Setup ───────────────────────────────────────────
st.set_page_config(
    page_title="Ultra Doc-Intelligence",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Ultra Doc-Intelligence")
st.caption("AI-powered logistics document assistant — Upload, Ask, Extract")


# ── Helper Functions ─────────────────────────────────────

def check_backend_health() -> bool:

    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def upload_document(uploaded_file) -> dict:

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    resp = requests.post(
        f"{API_BASE_URL}/upload",
        files=files,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def ask_question(document_id: str, question: str) -> dict:

    resp = requests.post(
        f"{API_BASE_URL}/ask",
        json={"document_id": document_id, "question": question},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def extract_data(document_id: str) -> dict:

    resp = requests.post(
        f"{API_BASE_URL}/extract",
        json={"document_id": document_id},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def render_confidence_badge(confidence: float) -> str:

    if confidence >= 0.7:
        return f"🟢 High ({confidence:.0%})"
    elif confidence >= 0.4:
        return f"🟡 Medium ({confidence:.0%})"
    else:
        return f"🔴 Low ({confidence:.0%})"


# ── Backend Health Check ─────────────────────────────────

if not check_backend_health():
    st.sidebar.error("❌ Cannot connect to backend at localhost:8000")
    st.error(
        "**Backend is not running.** Start it with:\n\n"
        "```bash\nuvicorn backend.app:app --reload --port 8000\n```"
    )
    st.stop()

st.sidebar.success("✅ Backend connected")

# ── Session State ────────────────────────────────────────

if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# ── Section 1: Document Upload ───────────────────────────

st.header("1️⃣ Upload a Document")

uploaded_file = st.file_uploader(
    "Upload a logistics document (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
    help="Rate Confirmations, BOLs, Invoices, Shipment Instructions",
)

if uploaded_file is not None:
    if st.button("📤 Process Document", type="primary"):
        with st.spinner("Parsing, chunking, and embedding..."):
            try:
                result = upload_document(uploaded_file)
                st.session_state.document_id = result["document_id"]
                st.session_state.filename = result["filename"]
                st.session_state.qa_history = []

                st.success(
                    f"✅ **{result['filename']}** processed successfully!\n\n"
                    f"- **Document ID:** `{result['document_id']}`\n"
                    f"- **Chunks created:** {result['chunks_created']}"
                )
            except requests.exceptions.HTTPError as e:
                error = e.response.json().get("detail", str(e))
                st.error(f"❌ Upload failed: {error}")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Try a smaller document.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# ── Active Document Info ─────────────────────────────────

if st.session_state.document_id:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Active Document")
    st.sidebar.markdown(f"**File:** {st.session_state.filename}")
    st.sidebar.markdown(f"**ID:** `{st.session_state.document_id}`")

    if st.sidebar.button("🗑️ Clear Document"):
        st.session_state.document_id = None
        st.session_state.filename = None
        st.session_state.qa_history = []
        st.rerun()

    # ── Section 2: Ask Questions ─────────────────────────

    st.markdown("---")
    st.header("2️⃣ Ask a Question")

    # Quick question buttons for common logistics queries
    st.caption("Quick questions:")
    quick_cols = st.columns(4)
    quick_questions = [
        "What is the carrier rate?",
        "When is pickup scheduled?",
        "Who is the consignee?",
        "What is the shipment weight?",
    ]

    selected_quick = None
    for i, q in enumerate(quick_questions):
        if quick_cols[i].button(q, key=f"quick_{i}"):
            selected_quick = q

    question = st.text_input(
        "Or type your own question:",
        value=selected_quick or "",
        placeholder="e.g., What equipment type is required?",
    )

if st.button("🔍 Ask", type="primary") and question:
        with st.spinner("Searching..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={"document_id": st.session_state.document_id, "question": question},
                    timeout=REQUEST_TIMEOUT,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state.qa_history.append({"question": question, "result": result})
                else:
                    st.error(f"Error: {resp.text}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # show all Q&A history
    for qa in reversed(st.session_state.qa_history):
        r = qa["result"]

        st.markdown(f"**Question:** {qa['question']}")

        # confidence badge
        conf = r.get("confidence", 0)
        if conf >= 0.7:
            badge = f"🟢 High ({conf:.0%})"
        elif conf >= 0.4:
            badge = f"🟡 Medium ({conf:.0%})"
        else:
            badge = f"🔴 Low ({conf:.0%})"

        st.markdown(f"**Confidence:** {badge}")

        # guardrail warning
        if r.get("guardrail_triggered"):
            st.warning(f"Guardrail: {r.get('guardrail_reason', 'N/A')}")

        # THE ANSWER
        st.success(r.get("answer", "No answer returned"))

        # sources
        sources = r.get("sources", [])
        if sources:
            with st.expander(f"View {len(sources)} sources"):
                for j, src in enumerate(sources, 1):
                    st.markdown(f"**Source {j}** — Chunk #{src.get('chunk_index', '?')} (similarity: {src.get('similarity', 0):.3f})")
                    st.text(src.get("text", ""))
                    st.markdown("---")

        st.markdown("---")
    # ── Section 3: Structured Extraction ─────────────────

    st.header("3️⃣ Structured Data Extraction")
    st.caption("Extract key shipment fields automatically")

    if st.button("📊 Extract Structured Data", type="primary"):
        with st.spinner("Extracting shipment data..."):
            try:
                result = extract_data(st.session_state.document_id)
                extracted = result["extracted_data"]

                st.markdown("### 📦 Extracted Shipment Data")

                # Two-column layout
                col1, col2 = st.columns(2)

                left_fields = [
                    ("Shipment ID", "shipment_id"),
                    ("Shipper", "shipper"),
                    ("Consignee", "consignee"),
                    ("Pickup Date/Time", "pickup_datetime"),
                    ("Delivery Date/Time", "delivery_datetime"),
                    ("Carrier Name", "carrier_name"),
                ]
                right_fields = [
                    ("Equipment Type", "equipment_type"),
                    ("Mode", "mode"),
                    ("Rate", "rate"),
                    ("Currency", "currency"),
                    ("Weight", "weight"),
                ]

                with col1:
                    for label, key in left_fields:
                        value = extracted.get(key)
                        display = value if value else "—"
                        st.markdown(f"**{label}:** {display}")

                with col2:
                    for label, key in right_fields:
                        value = extracted.get(key)
                        display = value if value else "—"
                        st.markdown(f"**{label}:** {display}")

                # Completeness indicator
                filled = sum(
                    1 for v in extracted.values() if v is not None
                )
                total = len(extracted)
                st.progress(
                    filled / total,
                    text=f"Fields extracted: {filled}/{total}",
                )

                with st.expander("📋 Raw JSON"):
                    st.json(extracted)

            except requests.exceptions.HTTPError as e:
                error = e.response.json().get("detail", str(e))
                st.error(f"❌ {error}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Upload a document above to get started.")
