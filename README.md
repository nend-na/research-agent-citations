# Research Agent with Citations

A Retrieval-Augmented Generation (RAG) agent that answers research
questions strictly from a set of supplied documents, tags every claim
with a `[Source N]` citation, and openly says so when the evidence isn't
there. Built and tested in VS Code.

My agent takes a research question and produces a citation-backed answer
from the supplied documents using semantic retrieval and optional LLM
synthesis.

This submission is scoped for the 24-hour AI Agent Challenge: it is
built as a runnable agent with a clear install path, sample data, and
honest tradeoff notes. The preferred reviewer flow is:

1. Install requirements
2. Copy `.env.example` to `.env`
3. Run `python run_agent.py` or `streamlit run streamlit_app.py`
4. Ask a research question and inspect citations

---

# Features

- Loads PDF, TXT, and Markdown documents.
- Chunks text (500 tokens, 50-token overlap) and embeds it using `sentence-transformers/all-MiniLM-L6-v2`.
- Semantic search over a FAISS index (cosine similarity, top-k = 5).
- Similarity threshold prevents irrelevant citations.
- Retrieval-Augmented Generation (RAG) with mandatory `[Source N]` citations.
- Supports Anthropic, OpenAI, Groq, and dependency-free Mock mode.
- Returns **"Not enough information"** instead of hallucinating.
- Clear provider error reporting for missing API keys, quota limits, authentication failures, and connectivity issues.
- Beautiful Streamlit interface.
- FastAPI REST API.
- Works entirely offline using Mock mode.

---

# Diagnosing Answer Generation Issues

If answers appear to come from the extractive fallback instead of an LLM, check the **System Status** card (Streamlit) or call:

```
GET /provider-status
```

Possible error types include:

| Error | Meaning | Resolution |
|--------|----------|------------|
| missing_api_key | API key missing | Add the API key to `.env` |
| authentication | Invalid key | Verify your API key |
| permission | Model access denied | Check provider permissions |
| not_found | Model doesn't exist | Verify model name |
| rate_limit | Quota exceeded | Retry later |
| network | Connection failed | Check internet/firewall |

Whenever a provider fails, the application still generates an extractive answer and clearly informs the user that an LLM response was unavailable.

---

# Project Structure

```text
research-agent/
├── app/
│   ├── loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── generator.py
│   ├── citations.py
│   └── main.py
├── data/
│   ├── sample_docs/
│   └── uploads/
├── outputs/
│   └── sample_answers/
├── tests/
├── streamlit_app.py
├── run_agent.py
├── questions.json
├── requirements.txt
├── tradeoffs.md
├── .env.example
└── README.md
```

---

# Setup

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Copy the environment file.

```bash
cp .env.example .env
```

Recommended VS Code extensions:

- Python
- Pylance
- Black
- Ruff
- Jupyter

If `sentence-transformers` or `faiss-cpu` are unavailable, the project automatically falls back to TF-IDF embeddings and NumPy similarity search so the pipeline remains runnable.

---

# Environment Variables

| Variable | Purpose |
|----------|---------|
| LLM_PROVIDER | mock / groq / openai / anthropic |
| GROQ_API_KEY | Required only for Groq |
| OPENAI_API_KEY | Required only for OpenAI |
| ANTHROPIC_API_KEY | Required only for Anthropic |
| CHUNK_SIZE | Chunk size |
| CHUNK_OVERLAP | Chunk overlap |
| TOP_K | Retrieved passages |

The default **Mock** provider requires **no API key**, making the project immediately runnable for reviewers.

---

# Running the Agent

## Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## Command Line

```bash
python run_agent.py
```

---

## FastAPI

```bash
uvicorn app.main:app --reload
```

Swagger documentation:

```
http://localhost:8000/docs
```

Example request:

```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"question":"What causes employee attrition?"}'
```

---

# Sample Data

Included sample documents:

- employee_report.txt
- remote_work_retention.txt
- climate_agriculture.txt

Sample questions:

```
questions.json
```

Generated outputs:

```
outputs/sample_answers/sample_outputs.json
```

One sample intentionally demonstrates the **"Not enough information"** path.

---

# Retrieval Pipeline

- Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector Database: FAISS
- Similarity Metric: Cosine Similarity
- Chunk Size: 500 tokens
- Chunk Overlap: 50 tokens
- Top-K Retrieval: 5

---

# Running Tests

```bash
pytest tests/
```

The tests cover:

- Document loading
- Chunking
- Retrieval
- Citation generation
- No-evidence handling

---

# Tradeoffs

See `tradeoffs.md` for:

- Design decisions
- Model selection
- Retrieval strategy
- Limitations
- Future improvements
- Planned enhancements

---

# Screenshots

 screenshots:

- Home screen
- Document upload
- Generated answer
- Citation panel
- Retrieved passages
