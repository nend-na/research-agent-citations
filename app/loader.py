"""
loader.py

Loads documents from a folder (or a list of uploaded file paths) and
extracts raw text. Supports .pdf, .txt, and .md files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List


@dataclass
class RawDocument:
    filename: str
    text: str


def _read_txt_or_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required to read PDF files. Install it with "
            "`pip install pypdf`."
        ) from e

    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def load_document(path: str) -> RawDocument:
    """Load a single document from disk and return its raw text."""
    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        text = _read_pdf(path)
    elif ext in (".txt", ".md"):
        text = _read_txt_or_md(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return RawDocument(filename=filename, text=text)


def load_documents_from_folder(folder: str) -> List[RawDocument]:
    """Load every supported document found in a folder."""
    documents = []
    if not os.path.isdir(folder):
        return documents

    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        ext = os.path.splitext(name)[1].lower()
        if ext in (".pdf", ".txt", ".md") and os.path.isfile(path):
            try:
                documents.append(load_document(path))
            except Exception as e:  # noqa: BLE001 - surface but keep going
                print(f"[loader] Skipping {name}: {e}")
    return documents


def load_documents_from_paths(paths: List[str]) -> List[RawDocument]:
    """Load documents given a list of explicit file paths (e.g. uploads)."""
    documents = []
    for path in paths:
        try:
            documents.append(load_document(path))
        except Exception as e:  # noqa: BLE001
            print(f"[loader] Skipping {path}: {e}")
    return documents
