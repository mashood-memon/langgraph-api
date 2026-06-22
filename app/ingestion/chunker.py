from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import re
from app.config import get_settings

_s = get_settings()

llm = ChatOpenAI(model=_s.primary_model, temperature=0)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def extract_text_pages(pdf_path: str) -> list[dict]:
    """Returns list of {page_num, text} dicts."""
    reader = PdfReader(pdf_path)
    return [
        {"page_num": i + 1, "text": page.extract_text() or ""}
        for i, page in enumerate(reader.pages)
    ]


def generate_doc_summary(full_text: str) -> str:
    """One-shot LLM summary of the full document. Prepended to each chunk."""
    truncated = full_text[:6000]   # keep cost low
    resp = llm.invoke(
        f"Summarize this technical document in 3-5 sentences. Focus on: "
        f"what system it describes, key components, and purpose.\n\n{truncated}"
    )
    return resp.content


def chunk_document(pdf_path: str, doc_id: str) -> list[dict]:
    """
    Full pipeline:
    1. Extract text per page
    2. Generate doc-level summary
    3. Split into chunks
    4. Prepend summary to each chunk (contextual chunking)
    Returns list of {chunk_text, page_num, doc_id, chunk_index}
    """
    pages = extract_text_pages(pdf_path)
    full_text = "\n\n".join(p["text"] for p in pages)

    doc_summary = generate_doc_summary(full_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for page in pages:
        if not page["text"].strip():
            continue
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            # Contextual chunk = doc summary + separator + actual chunk
            contextual_text = f"[Document Context]\n{doc_summary}\n\n[Chunk]\n{split}"
            chunks.append({
                "chunk_text": contextual_text,
                "raw_text": split,
                "page_num": page["page_num"],
                "doc_id": doc_id,
                "chunk_index": len(chunks),
            })

    return chunks