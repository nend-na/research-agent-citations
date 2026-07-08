"""
streamlit_app.py

A clean, professional, card-based UI for the Research Agent with
Citations. Runs entirely in-process (no separate FastAPI server needed)
by calling the app modules directly.
"""

from __future__ import annotations

import os
import time

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env so the running app picks up keys
load_dotenv()

from app.chunking import chunk_documents
from app.citations import build_citations, unused_citations
from app.embeddings import EmbeddingModel
from app.generator import check_provider_status, generate_answer
from app.loader import load_documents_from_folder
from app.retriever import Retriever

UPLOAD_DIR = "data/uploads"
SAMPLE_DIR = "data/sample_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Research Agent", layout="wide")

# ---------------------------------------------------------------- styling --
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background-color: #FAFAFA; }
    .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1120px; }

    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #0F172A; letter-spacing: -0.01em; }

    /* Base card */
    .card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.5rem 1.75rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.25rem;
    }
    .card-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #0F172A;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .card-subtitle { color: #6B7280; font-size: 0.85rem; margin-bottom: 1rem; }

    /* Status badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 999px;
    }
    .badge-ready { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
    .badge-error { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
    .badge-neutral { background: #F1F5F9; color: #475569; border: 1px solid #E2E8F0; }
    .badge-source { background: #EEF2FF; color: #4338CA; font-weight: 600; }
    .badge-score { background: #F8FAFC; color: #334155; border: 1px solid #E2E8F0; }

    /* Alert boxes */
    .alert {
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        font-size: 0.88rem;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .alert-error { background: #FEF2F2; border: 1px solid #FECACA; color: #7F1D1D; }
    .alert-warning { background: #FFFBEB; border: 1px solid #FDE68A; color: #78350F; }
    .alert-title { font-weight: 600; margin-bottom: 2px; display: block; }

    /* Answer + citation content */
    .answer-body {
        color: #1F2937;
        font-size: 1rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    .citation-row {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    .citation-meta { font-size: 0.82rem; color: #6B7280; margin-top: 2px; }
    .citation-preview {
        font-size: 0.88rem;
        color: #374151;
        margin-top: 6px;
        border-left: 2px solid #E5E7EB;
        padding-left: 10px;
    }

    .doc-chip {
        display: inline-flex;
        align-items: center;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 6px 12px;
        margin: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #334155;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 500;
        border: none;
        background-color: #2563EB;
        color: white;
    }
    div.stButton > button:hover { background-color: #1D4ED8; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state ----
if "retriever" not in st.session_state:
    st.session_state.embedding_model = EmbeddingModel()
    st.session_state.retriever = Retriever(st.session_state.embedding_model)
    st.session_state.history = []


def rebuild_index():
    docs = load_documents_from_folder(SAMPLE_DIR) + load_documents_from_folder(UPLOAD_DIR)
    chunks = chunk_documents(docs)
    st.session_state.retriever.build(chunks)
    st.session_state.loaded_docs = sorted({d.filename for d in docs})


if "loaded_docs" not in st.session_state:
    rebuild_index()

ERROR_COPY = {
    "missing_api_key": "No API key is configured for the selected provider.",
    "authentication": "The API key was rejected. Double-check that it was copied correctly and hasn't expired.",
    "permission": "This API key does not have access to the selected model.",
    "rate_limit": "The provider's rate limit or quota has been exceeded. Wait and try again, or check your plan.",
    "not_found": "The selected model was not found or is unavailable to this key.",
    "network": "Could not reach the provider's API. Check your internet connection or firewall settings.",
    "unknown": "Generation failed for an unexpected reason.",
}

# ---------------------------------------------------------------- header ---
st.title("Research Agent")
st.markdown(
    '<div class="card-subtitle" style="margin-top:-8px;">'
    "Ask a question and get an answer synthesized strictly from your documents, "
    "with every claim traced back to its source.</div>",
    unsafe_allow_html=True,
)

col_main, col_side = st.columns([2, 1], gap="large")

# ---------------------------------------------------------------- sidebar --
with col_side:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Source Documents</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Upload PDF, TXT, or Markdown files to add to the knowledge base.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Upload documents", type=["pdf", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed"
    )
    if uploaded:
        for f in uploaded:
            with open(os.path.join(UPLOAD_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())
        rebuild_index()
        st.success(f"Added {len(uploaded)} file(s). Index rebuilt.")

    for name in st.session_state.loaded_docs:
        st.markdown(f'<span class="doc-chip">{name}</span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Retrieval Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-k passages", 1, 10, 5, label_visibility="visible")
    provider_choice = st.selectbox(
        "LLM provider",
        ["mock (no API key)", "anthropic", "openai", "groq"],
    )
    provider_key = "mock" if provider_choice.startswith("mock") else provider_choice
    os.environ["LLM_PROVIDER"] = provider_key
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">System Status</div>', unsafe_allow_html=True)
    status = check_provider_status(provider_key)
    badge_class = "badge-ready" if status["ready"] else "badge-error"
    badge_text = "Ready" if status["ready"] else "Not configured"
    st.markdown(
        f'<span class="badge {badge_class}">{badge_text}</span>'
        f'<div class="citation-meta" style="margin-top:8px;">{status["detail"]}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Tradeoff Notes</div>', unsafe_allow_html=True)
    st.markdown(
        """<div class="card-subtitle" style="margin-bottom:0;">
        Embeddings: MiniLM (384-dim, fast, low memory)<br>
        Vector store: FAISS, cosine similarity<br>
        Chunking: 500 tokens, 50 overlap<br>
        Default mode: dependency-free extractive fallback<br>
        Future work: reranking, hybrid search, multi-hop reasoning
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- main -----
with col_main:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Ask a Research Question</div>', unsafe_allow_html=True)
    question = st.text_area(
        "Question",
        placeholder="e.g. What are the major causes of employee attrition?",
        label_visibility="collapsed",
        height=90,
    )
    ask_clicked = st.button("Get Answer", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if ask_clicked and question.strip():
        with st.spinner("Retrieving relevant passages..."):
            retrieved = st.session_state.retriever.search(question, top_k=top_k)
            time.sleep(0.15)

        if not retrieved:
            st.markdown(
                '<div class="alert alert-warning">'
                '<span class="alert-title">No relevant evidence found</span>'
                "None of the loaded documents are similar enough to this question to answer it reliably."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Synthesizing answer..."):
                result = generate_answer(question, retrieved)
                citations = build_citations(retrieved)
                unused = unused_citations(result.answer, citations)

            st.session_state.history.insert(
                0, {"question": question, "answer": result.answer, "citations": citations}
            )

            if not result.success:
                friendly = ERROR_COPY.get(result.error_type, ERROR_COPY["unknown"])
                st.markdown(
                    f'<div class="alert alert-error">'
                    f'<span class="alert-title">Generation with "{result.requested_provider}" failed &mdash; showing extractive fallback instead</span>'
                    f"{friendly}"
                    f'<div class="citation-meta" style="margin-top:6px;">Details: {result.error_message}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown('<div class="card">', unsafe_allow_html=True)
            header = "Answer"
            if result.success and result.provider_used != "mock":
                header += f" &middot; {result.provider_used} ({result.model})"
            elif result.provider_used == "mock":
                header += " &middot; extractive fallback"
            st.markdown(f'<div class="card-title">{header}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-body">{result.answer}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Citations</div>', unsafe_allow_html=True)
            for c in citations:
                st.markdown(
                    f'<div class="citation-row">'
                    f'<span class="badge badge-source">Source {c.index}</span> '
                    f'<span class="badge badge-score">similarity {c.similarity:.2f}</span>'
                    f'<div class="citation-meta">{c.filename} &mdash; chunk #{c.chunk_id}</div>'
                    f'<div class="citation-preview">{c.preview}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            if unused:
                st.markdown(
                    f'<div class="card-subtitle">{len(unused)} retrieved passage(s) were not referenced in the final answer.</div>',
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    elif ask_clicked:
        st.markdown(
            '<div class="alert alert-warning">'
            '<span class="alert-title">Missing question</span>'
            "Please enter a question before requesting an answer."
            "</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.history:
        with st.expander(f"Search history ({len(st.session_state.history)})"):
            for item in st.session_state.history:
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(item["answer"])
                st.divider()
