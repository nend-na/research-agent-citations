# Tradeoff Notes

## Why MiniLM (`all-MiniLM-L6-v2`) for embeddings?
It is small (~80MB), fast on CPU, and produces 384-dimensional embeddings
that are good enough for semantic retrieval over small-to-medium document
sets. Larger embedding models (e.g. `bge-large`, `text-embedding-3-large`)
would likely improve recall slightly but cost significantly more in
latency and memory, which isn't justified for a document set of this size.

## Why FAISS?
FAISS is lightweight, has no external service dependency (unlike Pinecone
or Weaviate), and is trivial to embed directly inside a Python process.
`IndexFlatIP` (exact search via inner product on normalized vectors ==
cosine similarity) is used rather than an approximate index (e.g. IVF or
HNSW) because the corpus is small enough that exact search is fast and
guarantees the true top-k neighbors.

## Why a pluggable LLM (mock / Anthropic / OpenAI / Groq)?
The generation step is provider-agnostic by design:
- **mock** (default): a dependency-free extractive fallback that picks the
  most relevant sentence from each retrieved chunk. This keeps the whole
  pipeline runnable and reproducible with zero API keys and zero network
  calls, which matters for quick review.
- **anthropic / openai / groq**: real LLM synthesis for higher-quality,
  more fluent answers. Any of these can be enabled by setting
  `LLM_PROVIDER` and the matching API key in `.env`.

Llama 3.3 70B (via Groq) was the suggested default for a "real" run
because of its strong quality-to-latency ratio, but the system doesn't
lock a reviewer into one paid API.

## Chunking: 500 tokens / 50 overlap
500-word chunks are large enough to preserve context within a claim but
small enough to keep retrieval precise and prompts short. A 50-word
overlap prevents relevant sentences from being cut across a chunk
boundary and disappearing from retrieval entirely.

## Top-k = 5
Five passages balance recall (enough evidence to synthesize a complete
answer) against prompt length (more passages increase LLM cost/latency
and dilute relevance). This is configurable via `TOP_K`/the UI slider.

## Latency tradeoffs
- Embedding + exact FAISS search over a few hundred chunks: sub-second.
- Mock generation: near-instant (no network call).
- Real LLM generation: dominated by the provider's API latency (typically
  1-4 seconds), not by retrieval.

## Accuracy tradeoffs
- Exact FAISS search guarantees correct top-k neighbors but doesn't scale
  to millions of chunks without switching to an approximate index.
- The mock generator is deliberately conservative (extractive) so it never
  hallucinates, but it produces less fluent prose than a real LLM.
- Citation correctness currently depends on the LLM/mock honoring the
  `[Source N]` convention; there is no independent fact-checking step.

## Current limitations
- No reranking step after initial retrieval.
- No hybrid (keyword + semantic) search; purely dense retrieval.
- No multi-hop reasoning across documents that don't share vocabulary.
- Citation matching relies on chunk metadata (filename + chunk id), not
  exact character-offset provenance within the source file.
- PDF text extraction quality depends on how the PDF was produced (scanned
  PDFs without OCR will yield poor or empty text).

## Future improvements
- Add a cross-encoder reranker on top of the initial FAISS retrieval.
- Add hybrid retrieval (BM25 + dense) for better handling of rare terms.
- Add multi-hop / iterative retrieval for questions spanning documents.
- Add OCR fallback for scanned PDFs.
- Add streaming responses in the UI and export-to-PDF for answers.
