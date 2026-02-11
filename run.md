# How to Run Ultra Doc-Intelligence

## Prerequisites
- **Python 3.10+**
- **Ollama**: [Download from ollama.com](https://ollama.com/)

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/ultra-doc-intelligence.git
cd ultra-doc-intelligence

# 2. Setup Virtual Environment
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull local models
ollama pull llama2
ollama pull nomic-embed-text

# 5. Start Backend (Terminal 1)
uvicorn backend.app:app --reload --port 8000

# 6. Start Frontend (Terminal 2)
streamlit run frontend/frontend.py
```

## API Testing with curl

```bash
# Health Check
curl http://localhost:8000/health

# Upload Document
curl -X POST http://localhost:8000/upload -F "file=@sample.pdf"

# Ask Question (replace {id} with document_id from upload)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"document_id": "{id}", "question": "What is the total rate?"}'

# Extract Data
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"document_id": "{id}"}'
```

## Running Tests
```bash
pytest tests/ -v
```