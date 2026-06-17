"""Ingest PDFs (or raw page text) into retrievable Chunks.

`window_text` and `chunk_pages` are pure and unit-tested. `load_pdf_chunks` uses PyMuPDF
to read a folder of PDFs (page text extraction) and is exercised by the demo wiring, not
unit tests.
"""

import glob
import hashlib
import pathlib

from hoprag.types import Chunk


def window_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    """Split text into overlapping char windows (works for CJK; whitespace-normalized)."""
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    text = " ".join(text.split())
    if not text:
        return []
    out, i, step = [], 0, size - overlap
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += step
    return out


def _chunk_id(source: str, page: int, idx: int, text: str) -> str:
    h = hashlib.sha1(f"{source}|{page}|{idx}|{text[:40]}".encode("utf-8")).hexdigest()[:12]
    return f"ch_{h}"


def chunk_pages(pages, source: str, size: int = 500, overlap: int = 80) -> list[Chunk]:
    """pages: iterable of (page_number, page_text). Returns windowed Chunks with
    title 'source p.N' so the UI can cite the source file + page."""
    chunks = []
    for page_no, page_text in pages:
        for idx, window in enumerate(window_text(page_text, size, overlap)):
            chunks.append(Chunk(
                id=_chunk_id(source, page_no, idx, window),
                title=f"{source} p.{page_no}",
                text=window,
                source_qid=source,
            ))
    return chunks


def load_pdf_chunks(folder: str, size: int = 500, overlap: int = 80) -> list[Chunk]:
    """Read every *.pdf in `folder` (PyMuPDF), extract per-page text, return Chunks."""
    import fitz  # PyMuPDF

    chunks = []
    for path in sorted(glob.glob(str(pathlib.Path(folder) / "*.pdf"))):
        name = pathlib.Path(path).stem
        doc = fitz.open(path)
        pages = [(i + 1, doc[i].get_text()) for i in range(doc.page_count)]
        doc.close()
        chunks.extend(chunk_pages(pages, name, size, overlap))
    return chunks
