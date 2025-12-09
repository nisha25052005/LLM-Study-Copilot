import PyPDF2
from typing import List, Dict


def extract_text_from_pdf(file) -> List[Dict]:
    """
    Returns a list of dicts:
    [
      {"page_num": 1, "text": "..."},
      ...
    ]
    """
    reader = PyPDF2.PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page_num": i + 1, "text": text})
    return pages


def chunk_text(
    pages: List[Dict],
    max_chars: int = 800,
    overlap: int = 200
) -> List[Dict]:
    """
    Split pages into overlapping chunks.
    Returns:
    [
      {"chunk_id": 0, "page_num": 1, "text": "..."},
      ...
    ]
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        page_num = page["page_num"]

        start = 0
        while start < len(text):
            end = start + max_chars
            chunk_text = text[start:end]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page_num": page_num,
                    "text": chunk_text,
                }
            )
            chunk_id += 1
            start = end - overlap  # move with overlap

    return chunks
