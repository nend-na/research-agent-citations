const API = "";

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------- toast ---
function toast(message, type = "info") {
  const container = el("toast-container");
  const node = document.createElement("div");
  node.className = `toast ${type}`;
  node.textContent = message;
  container.appendChild(node);
  setTimeout(() => {
    node.style.opacity = "0";
    node.style.transition = "opacity 0.2s ease";
    setTimeout(() => node.remove(), 200);
  }, 4000);
}

// ---------------------------------------------------------------- theme ---
function initTheme() {
  const saved = localStorage.getItem("ra-theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  el("theme-toggle").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("ra-theme", next);
  });
}

// ---------------------------------------------------------------- status --
async function refreshStatus() {
  try {
    const res = await fetch(`${API}/provider-status`);
    const data = await res.json();
    const pill = el("status-pill");
    pill.textContent = data.ready ? `${data.provider} ready` : `${data.provider} not configured`;
    pill.className = `status-pill ${data.ready ? "status-ready" : "status-error"}`;
    el("modal-status").textContent = data.detail;
  } catch (e) {
    el("status-pill").textContent = "Backend unreachable";
    el("status-pill").className = "status-pill status-error";
  }
}

async function refreshConfig() {
  try {
    const res = await fetch(`${API}/config`);
    const data = await res.json();
    const list = el("config-list");
    const rows = [
      ["Embedding model", data.embedding_model],
      ["Chunk size / overlap", `${data.chunk_size} / ${data.chunk_overlap}`],
      ["Top-k", data.top_k],
      ["Similarity metric", data.similarity_metric],
      ["LLM provider", data.llm_provider],
    ];
    list.innerHTML = rows
      .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`)
      .join("");
    el("provider-select").value = data.llm_provider;
  } catch (e) {
    // non-fatal; tradeoffs panel just stays empty
  }
}

// ---------------------------------------------------------------- modal ---
function initModal() {
  el("settings-button").addEventListener("click", () => el("settings-modal").classList.remove("hidden"));
  el("close-settings").addEventListener("click", () => el("settings-modal").classList.add("hidden"));
  el("settings-modal").addEventListener("click", (e) => {
    if (e.target === el("settings-modal")) el("settings-modal").classList.add("hidden");
  });
  el("provider-select").addEventListener("change", async (e) => {
    const provider = e.target.value;
    try {
      const res = await fetch(`${API}/set-provider`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider }),
      });
      const data = await res.json();
      el("modal-status").textContent = data.detail;
      toast(`Switched to ${provider}`, data.ready ? "success" : "error");
      refreshStatus();
      refreshConfig();
    } catch (err) {
      toast("Could not reach the backend to switch providers.", "error");
    }
  });
}

// ---------------------------------------------------------------- collapse
function initCollapsible() {
  document.querySelectorAll(".collapsible-header").forEach((header) => {
    const body = el(header.dataset.target);
    body.style.maxHeight = body.scrollHeight + "px";
    header.addEventListener("click", () => {
      header.classList.toggle("collapsed");
      if (header.classList.contains("collapsed")) {
        body.style.maxHeight = "0px";
      } else {
        body.style.maxHeight = body.scrollHeight + "px";
      }
    });
  });
}

// ---------------------------------------------------------------- docs ----
async function loadDocuments() {
  try {
    const res = await fetch(`${API}/documents`);
    const data = await res.json();
    renderDocuments(data.documents);
  } catch (e) {
    // backend not reachable yet; ignore
  }
}

function renderDocuments(names) {
  const list = el("document-list");
  if (!names.length) {
    list.innerHTML = `<p class="panel-sub">No documents loaded yet.</p>`;
    return;
  }
  list.innerHTML = names
    .map(
      (name) => `
      <div class="doc-item" data-name="${name}">
        <span class="doc-item-name">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6 2h9l5 5v15H6Z"/><path d="M15 2v5h5"/></svg>
          ${name}
        </span>
        <button class="doc-remove" aria-label="Remove ${name}">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6 6l12 12M18 6 6 18"/></svg>
        </button>
      </div>`
    )
    .join("");

  list.querySelectorAll(".doc-remove").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      const item = e.target.closest(".doc-item");
      const name = item.dataset.name;
      try {
        const res = await fetch(`${API}/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
        if (!res.ok) throw new Error(await res.text());
        toast(`Removed ${name}`, "success");
        loadDocuments();
      } catch (err) {
        toast(`Could not remove ${name}. It may be a bundled sample document.`, "error");
      }
    });
  });
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList).filter((f) => /\.(pdf|txt|md)$/i.test(f.name));
  if (!files.length) {
    toast("Only PDF, TXT, or Markdown files are supported.", "error");
    return;
  }
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));

  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    toast(`Added ${data.uploaded.length} file(s)`, "success");
    loadDocuments();
  } catch (err) {
    toast("Upload failed. Check the file type and try again.", "error");
  }
}

function initUpload() {
  const dropzone = el("dropzone");
  const fileInput = el("file-input");

  dropzone.addEventListener("click", () => fileInput.click());
  el("upload-trigger").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => uploadFiles(e.target.files));

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );
  dropzone.addEventListener("drop", (e) => uploadFiles(e.dataTransfer.files));
}

// ---------------------------------------------------------------- ask -----
const ERROR_COPY = {
  missing_api_key: "No API key is configured for the selected provider.",
  authentication: "The API key was rejected. Double-check that it was copied correctly and hasn't expired.",
  permission: "This API key does not have access to the selected model.",
  rate_limit: "The provider's rate limit or quota has been exceeded.",
  not_found: "The selected model was not found or is unavailable to this key.",
  network: "Could not reach the provider's API. Check your connection or firewall settings.",
  unknown: "Generation failed for an unexpected reason.",
};

const LOADING_STAGES = [
  "Reading documents",
  "Finding relevant passages",
  "Generating response",
  "Verifying citations",
  "Preparing final answer",
];

let _stageTimer = null;

function renderStages() {
  el("retrieval-panel").classList.remove("hidden");
  el("answer-panel").classList.remove("hidden");
  el("citation-panel").classList.add("hidden");
  el("status-banner-slot").innerHTML = "";
  el("answer-source-tag").textContent = "";
  el("retrieval-cards").innerHTML = "";

  const list = document.createElement("div");
  list.className = "stage-list";
  list.id = "stage-list";
  LOADING_STAGES.forEach((label, i) => {
    const item = document.createElement("div");
    item.className = "stage-item" + (i === 0 ? " active" : "");
    item.dataset.index = i;
    item.innerHTML = `<span class="stage-dot"></span><span>${label}&hellip;</span>`;
    list.appendChild(item);
  });
  el("answer-body").innerHTML = "";
  el("answer-body").appendChild(list);
}

function advanceStage(index) {
  const list = el("stage-list");
  if (!list) return;
  list.querySelectorAll(".stage-item").forEach((item) => {
    const i = Number(item.dataset.index);
    item.classList.toggle("done", i < index);
    item.classList.toggle("active", i === index);
  });
}

function startStageTimer() {
  let current = 0;
  _stageTimer = setInterval(() => {
    current = Math.min(current + 1, LOADING_STAGES.length - 1);
    advanceStage(current);
    if (current >= LOADING_STAGES.length - 1) clearInterval(_stageTimer);
  }, 550);
}

function stopStageTimer() {
  if (_stageTimer) clearInterval(_stageTimer);
  _stageTimer = null;
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

function renderAnswer(text) {
  // Turn "[Source N]" markers into clickable badges that scroll to the
  // matching citation card and highlight it briefly.
  const escaped = escapeHtml(text);
  return escaped.replace(/\[Source (\d+)\]/g, (_, n) => `<span class="cite-badge" data-source="${n}">${n}</span>`);
}

function attachCiteHandlers() {
  document.querySelectorAll(".cite-badge").forEach((badge) => {
    badge.addEventListener("click", () => {
      const target = document.querySelector(`.evidence-card[data-source="${badge.dataset.source}"]`);
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.add("highlight");
      setTimeout(() => target.classList.remove("highlight"), 1400);
    });
  });
}

function renderEvidenceCard(c, container) {
  const pct = Math.max(0, Math.min(100, Math.round(c.similarity * 100)));
  const card = document.createElement("div");
  card.className = "evidence-card";
  card.dataset.source = c.index;
  card.innerHTML = `
    <div class="evidence-top">
      <span class="badge-source">Source ${c.index}</span>
      <span class="badge-score">similarity ${c.similarity.toFixed(2)}</span>
    </div>
    <div class="evidence-meta">${escapeHtml(c.filename)} &middot; chunk #${c.chunk_id}</div>
    <div class="similarity-bar"><div class="similarity-bar-fill" style="width:${pct}%"></div></div>
    <div class="evidence-preview">${escapeHtml(c.preview)}</div>
    <button class="evidence-expand">Show more</button>
  `;
  const preview = card.querySelector(".evidence-preview");
  const expandBtn = card.querySelector(".evidence-expand");
  expandBtn.addEventListener("click", () => {
    preview.classList.toggle("expanded");
    expandBtn.textContent = preview.classList.contains("expanded") ? "Show less" : "Show more";
  });
  container.appendChild(card);
}

async function askQuestion() {
  const question = el("question-input").value.trim();
  if (!question) {
    toast("Please enter a question first.", "error");
    return;
  }

  renderStages();
  startStageTimer();
  const askBtn = el("ask-button");
  askBtn.disabled = true;
  askBtn.classList.add("is-loading");

  try {
    const res = await fetch(`${API}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    stopStageTimer();
    advanceStage(LOADING_STAGES.length - 1);
    setTimeout(() => renderResult(data), 220);
  } catch (err) {
    stopStageTimer();
    el("answer-body").innerHTML = "";
    el("status-banner-slot").innerHTML = `
      <div class="status-banner error">
        <span class="status-banner-title">Request failed</span>
        Could not reach the backend, or it returned an error. Check that the server is running.
      </div>`;
    toast("Something went wrong while asking your question.", "error");
  } finally {
    askBtn.disabled = false;
    askBtn.classList.remove("is-loading");
  }
}

function renderResult(data) {
  const retrievalCards = el("retrieval-cards");
  retrievalCards.innerHTML = "";

  if (!data.citations.length) {
    el("retrieval-count").textContent = "0 passages met the relevance threshold";
    retrievalCards.innerHTML = `<p class="panel-sub">No passages in the current document set were similar enough to this question.</p>`;
    el("citation-panel").classList.add("hidden");
  } else {
    el("retrieval-count").textContent = `${data.citations.length} passage(s) retrieved`;
    data.citations.forEach((c) => renderEvidenceCard(c, retrievalCards));
    el("citation-panel").classList.remove("hidden");
    const citationCards = el("citation-cards");
    citationCards.innerHTML = "";
    data.citations.forEach((c) => renderEvidenceCard(c, citationCards));
    el("citation-count").textContent = `${data.citations.length} source(s)`;
  }

  const banner = el("status-banner-slot");
  const gs = data.generation_status;
  if (gs && gs.success === false) {
    const friendly = ERROR_COPY[gs.error_type] || ERROR_COPY.unknown;
    banner.innerHTML = `
      <div class="status-banner error">
        <span class="status-banner-title">Generation with "${escapeHtml(gs.requested_provider || "")}" failed &mdash; showing extractive fallback instead</span>
        ${friendly}
        <span class="status-banner-detail">${escapeHtml(gs.error_message || "")}</span>
      </div>`;
  } else {
    banner.innerHTML = "";
  }

  const tag = el("answer-source-tag");
  if (gs && gs.provider_used && gs.provider_used !== "mock" && gs.success) {
    tag.textContent = `${gs.provider_used}${gs.model ? " · " + gs.model : ""}`;
  } else if (gs && gs.provider_used === "mock") {
    tag.textContent = "extractive fallback";
  } else {
    tag.textContent = "";
  }

  el("answer-body").innerHTML = renderAnswer(data.answer);
  attachCiteHandlers();
}

function initAsk() {
  el("ask-button").addEventListener("click", askQuestion);
  el("question-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  });
  el("question-input").addEventListener("input", (e) => {
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + "px";
  });
  el("copy-answer").addEventListener("click", () => {
    const text = el("answer-body").innerText;
    navigator.clipboard.writeText(text).then(() => toast("Answer copied to clipboard", "success"));
  });
}

// ---------------------------------------------------------------- init ----
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initModal();
  initCollapsible();
  initUpload();
  initAsk();
  loadDocuments();
  refreshStatus();
  refreshConfig();
});