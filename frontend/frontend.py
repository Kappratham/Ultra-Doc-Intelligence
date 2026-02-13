import streamlit as st
import requests
import os

if os.environ.get("RENDER"):
    API_BASE_URL = "https://ultra-doc-backend-vvy0.onrender.com/"
else:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

REQUEST_TIMEOUT = 60

st.set_page_config(page_title="Ultra Doc-Intelligence", page_icon="📄", layout="wide")
st.title("📄 Ultra Doc-Intelligence")
st.caption("AI-powered logistics document assistant")


def check_backend():
    try:
        return requests.get(f"{API_BASE_URL}/health", timeout=5).status_code == 200
    except Exception:
        return False


if not check_backend():
    st.sidebar.error("Backend offline")
    st.error(f"Cannot connect to backend at {API_BASE_URL}")
    st.stop()

st.sidebar.success("Backend connected")

if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# ── Upload Section ───────────────────────────────────────

st.header("1️⃣ Upload a Document")

uploaded_file = st.file_uploader(
    "Upload a logistics document (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "txt"],
)

if uploaded_file is not None:
    if st.button("📤 Process Document", type="primary"):
        with st.spinner("Processing document..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }
                resp = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=REQUEST_TIMEOUT)

                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state.document_id = result["document_id"]
                    st.session_state.filename = result["filename"]
                    st.session_state.qa_history = []
                    st.success(
                        f"Document processed! ID: {result['document_id']} | "
                        f"Chunks: {result['chunks_created']}"
                    )
                else:
                    st.error(f"Upload failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# ── Sidebar Info ─────────────────────────────────────────

if st.session_state.document_id:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Document")
    st.sidebar.markdown(f"**File:** {st.session_state.filename}")
    st.sidebar.markdown(f"**ID:** `{st.session_state.document_id}`")

    if st.sidebar.button("Clear Document"):
        st.session_state.document_id = None
        st.session_state.filename = None
        st.session_state.qa_history = []
        st.rerun()

    # ── Ask Section ──────────────────────────────────────

    st.markdown("---")
    st.header("2️⃣ Ask a Question")

    quick_cols = st.columns(4)
    quick_questions = [
        "What is the carrier rate?",
        "When is pickup scheduled?",
        "Who is the consignee?",
        "What is the shipment weight?",
    ]

    selected_quick = None
    for i, q in enumerate(quick_questions):
        if quick_cols[i].button(q, key=f"q_{i}"):
            selected_quick = q

    question = st.text_input(
        "Or type your own question:",
        value=selected_quick or "",
        placeholder="e.g., What equipment type is required?",
    )

    if st.button("🔍 Ask", type="primary") and question:
        with st.spinner("Searching and generating answer..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/ask",
                    json={
                        "document_id": st.session_state.document_id,
                        "question": question,
                    },
                    timeout=REQUEST_TIMEOUT,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    st.session_state.qa_history.append({
                        "question": question,
                        "result": result,
                    })
                else:
                    st.error(f"Error: {resp.text}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    # ── Show Q&A History ─────────────────────────────────

    for qa in reversed(st.session_state.qa_history):
        r = qa["result"]

        st.markdown(f"**❓ {qa['question']}**")

        conf = r.get("confidence", 0)
        if conf >= 0.7:
            badge = f"🟢 High ({conf:.0%})"
        elif conf >= 0.4:
            badge = f"🟡 Medium ({conf:.0%})"
        else:
            badge = f"🔴 Low ({conf:.0%})"

        st.markdown(f"**Confidence:** {badge}")

        if r.get("guardrail_triggered"):
            st.warning(f"Guardrail: {r.get('guardrail_reason', 'N/A')}")

        st.success(r.get("answer", "No answer returned"))

        sources = r.get("sources", [])
        if sources:
            with st.expander(f"📑 View {len(sources)} sources"):
                for j, src in enumerate(sources, 1):
                    chunk_idx = src.get("chunk_index", "?")
                    sim = src.get("similarity", 0)
                    st.markdown(f"**Source {j}** — Chunk #{chunk_idx} (similarity: {sim:.3f})")
                    st.text(src.get("text", ""))
                    st.markdown("---")

        st.markdown("---")

    # ── Extract Section ──────────────────────────────────

    st.header("3️⃣ Structured Extraction")

    if st.button("📊 Extract Data", type="primary"):
        with st.spinner("Extracting structured data..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/extract",
                    json={"document_id": st.session_state.document_id},
                    timeout=REQUEST_TIMEOUT,
                )

                if resp.status_code == 200:
                    result = resp.json()
                    extracted = result["extracted_data"]

                    st.markdown("### Extracted Shipment Data")

                    col1, col2 = st.columns(2)

                    left = [
                        ("Shipment ID", "shipment_id"),
                        ("Shipper", "shipper"),
                        ("Consignee", "consignee"),
                        ("Pickup", "pickup_datetime"),
                        ("Delivery", "delivery_datetime"),
                        ("Carrier", "carrier_name"),
                    ]
                    right = [
                        ("Equipment", "equipment_type"),
                        ("Mode", "mode"),
                        ("Rate", "rate"),
                        ("Currency", "currency"),
                        ("Weight", "weight"),
                    ]

                    with col1:
                        for label, key in left:
                            val = extracted.get(key) or "—"
                            st.markdown(f"**{label}:** {val}")

                    with col2:
                        for label, key in right:
                            val = extracted.get(key) or "—"
                            st.markdown(f"**{label}:** {val}")

                    filled = sum(1 for v in extracted.values() if v is not None)
                    total = len(extracted)
                    st.progress(filled / total, text=f"Fields found: {filled}/{total}")

                    with st.expander("Raw JSON"):
                        st.json(extracted)

                else:
                    st.error(f"Error: {resp.text}")

            except Exception as e:
                st.error(f"Error: {str(e)}")

else:
    st.info("Upload a document above to get started.")
