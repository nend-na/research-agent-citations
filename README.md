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

1. install requirements
2. copy `.env.example` to `.env`
3. run `python run_agent.py` or `streamlit run streamlit_app.py`
4. ask a research question and inspect citations

---

## Features

- Loads PDF / TXT / Markdown documents
- Chunks text (500 tokens, 50-token overlap) and embeds it with
  `sentence-transformers/all-MiniLM-L6-v2`
- Semantic search over a FAISS index (cosine similarity, top-k = 5),
  with a similarity threshold so out-of-domain questions correctly get
  "not enough information" instead of citing irrelevant passages
- Answer synthesis with mandatory `[Source N]` citations, and a pluggable
  LLM backend (Anthropic / OpenAI / Groq / dependency-free mock mode)
- Explicit "not enough information" response when no evidence supports
  the question — never fabricates an answer
- **Generation never fails silently.** If the selected provider's API key
  is missing, invalid, out of quota, lacks model access, or unreachable,
  the app reports exactly which of those it was — rather than quietly
  substituting the extractive fallback with no explanation
- Two ways to run it: a clean, card-based Streamlit UI, or a FastAPI
  JSON API
- Runs out of the box, with **zero API keys**, via the extractive mock
  generator (great for a 5-minute reviewer run); swap in a real LLM key
  any time for higher-quality synthesis

---

## Diagnosing Answer Generation Issues

If answers seem to be coming from the extractive fallback instead of a
real LLM, or generation appears "not to work," check the **System
Status** card in the sidebar (Streamlit) or call `GET /provider-status`
(FastAPI) — this reports readiness without making any API call. If a
question was already asked, the answer itself will show a red alert box
naming the exact failure mode:

| error_type | Meaning | Fix |
|---|---|---|
| `missing_api_key` | The relevant `*_API_KEY` env var isn't set | Add it to `.env` and restart the app |
| `authentication` | The key was rejected by the provider | Confirm the key was copied correctly and hasn't been revoked |
| `permission` | The key doesn't have access to the configured model | Check the model name / your account's model access |
| `not_found` | The configured model name doesn't exist for that provider | Check `ANTHROPIC_MODEL` / `OPENAI_MODEL` / `GROQ_MODEL` in `.env` |
| `rate_limit` | Provider rate limit or quota exceeded | Wait and retry, or check your usage/plan |
| `network` | The API couldn't be reached at all | Check your internet connection or firewall/proxy settings |

In every case, the UI still shows an answer (the extractive fallback) so
the session isn't a dead end — it just makes clear that answer did not
come from the LLM you selected.

---

## Project Structure

```
research-agent/
├── app/
│   ├── loader.py          # PDF/TXT/MD loading
│   ├── chunking.py        # 500-token / 50-overlap chunking
│   ├── embeddings.py      # MiniLM embeddings (TF-IDF fallback)
│   ├── retriever.py       # FAISS index + search (numpy fallback)
│   ├── generator.py       # LLM synthesis (mock/anthropic/openai/groq)
│   ├── citations.py       # Citation metadata + cross-checking
│   └── main.py            # FastAPI backend
├── streamlit_app.py       # Polished card-based UI
├── run_agent.py           # Command-line research agent entrypoint
├── data/
│   ├── sample_docs/       # 3 sample research documents
│   └── uploads/           # created at runtime for uploaded files
├── outputs/sample_answers/sample_outputs.json
├── tests/test_pipeline.py
├── questions.json
├── tradeoffs.md
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup (VS Code)

Open the folder in VS Code, then in the integrated terminal:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

Recommended VS Code extensions: Python, Pylance, Jupyter, Ruff/Black.

> **Note:** `sentence-transformers` and `faiss-cpu` are the "real" backends
> for embeddings and retrieval. If either isn't installed (or the model
> download isn't available in your environment), the code automatically
> falls back to a TF-IDF embedding + numpy cosine-similarity search so the
> pipeline still runs end to end.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in only what you need:

| Variable | Purpose | Required? |
|---|---|---|
| `LLM_PROVIDER` | `mock` (default), `anthropic`, `openai`, or `groq` | No — defaults to `mock` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Used if `LLM_PROVIDER=anthropic` | Only for that provider |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | Used if `LLM_PROVIDER=openai` | Only for that provider |
| `GROQ_API_KEY` / `GROQ_MODEL` | Used if `LLM_PROVIDER=groq` | Only for that provider |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | Retrieval tuning | No — sensible defaults |

The `mock` provider needs **no key at all** and is dependency-free — it
extracts the most question-relevant sentence from each retrieved chunk
and tags it with its source. This keeps the project fully runnable and
reproducible for reviewers who don't want to provision an API key.

---

## Running the Agent

### Option A — Streamlit UI (recommended, fastest)

```bash
streamlit run streamlit_app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

### Option B — Command-line agent (simple, reviewer-friendly)

```bash
python run_agent.py
```

This launches a prompt-based agent that reads the sample documents and
answers research questions with citation-aware responses.

**Reviewer tip:** if you do not want to configure an API key, leave
`LLM_PROVIDER=mock` in `.env`; the agent will still answer using the
extractive fallback.

### Option C — FastAPI backend

```bash
uvicorn app.main:app --reload
```

API docs (Swagger UI) will be available at `http://localhost:8000/docs`.

Example request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes employee attrition?"}'
```

Expected response shape:

```json
{
  "question": "What causes employee attrition?",
  "answer": "... [Source 1] ... [Source 2]",
  "citations": [
    {"index": 1, "filename": "employee_report.txt", "chunk_id": 0, "similarity": 0.42, "preview": "..."}
  ],
  "unused_citations": [],
  "retrieval_notes": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 5,
    "similarity_metric": "cosine"
  }
}
```

---

## Sample Data & Expected Output

Three sample documents ship in `data/sample_docs/`:

- `employee_report.txt` — causes of employee attrition
- `remote_work_retention.txt` — remote work and retention
- `climate_agriculture.txt` — climate impacts on agriculture

Sample questions live in `questions.json`, and pre-generated answers
(using the mock generator) are saved in
`outputs/sample_answers/sample_outputs.json` for quick inspection without
running anything.

One of the sample questions ("What challenges arise during AI adoption?")
deliberately has **no matching source document**, to demonstrate the
system's "not enough information" behavior rather than hallucinating.

---

## Retrieval Notes

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Chunking:** 500-token chunks, 50-token overlap
- **Vector index:** FAISS `IndexFlatIP` (exact search, cosine similarity
  via inner product on L2-normalized vectors)
- **Top-k:** 5 passages per query (configurable)

---

## Tests

```bash
pytest tests/
```

Covers chunk overlap correctness, document loading, retrieval relevance,
citation building, and the "no evidence" path.

---

## Tradeoffs & Limitations

See [`tradeoffs.md`](tradeoffs.md) for the full writeup — why FAISS, why
MiniLM, why a pluggable LLM, latency/accuracy tradeoffs, current
limitations, and planned improvements (reranking, hybrid retrieval,
multi-hop reasoning).

---

## Screenshots

See `screenshots/` — add UI screenshots here after running the Streamlit
app locally (e.g. the question card, retrieval results, and citation
panel).
