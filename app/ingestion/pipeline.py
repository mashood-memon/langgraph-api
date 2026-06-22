import uuid
from .chunker import chunk_document
from .voyage_embed import embed_text, embed_images, render_pdf_pages
from .bm25_index import build_index
from app.vectorstore.qdrant_client import ensure_collections, upsert_text_chunk, upsert_image_page


async def ingest_document(pdf_path: str, filename: str) -> dict:
    """
    Full ingestion pipeline:
    1. Chunk text with contextual summaries
    2. Embed text chunks via Voyage → Qdrant (content_type="text")
    3. Build BM25 index
    4. Render pages as images, embed via Voyage → Qdrant (content_type="image")
    Both embedding calls use the same model/space — one collection, one client.
    """
    ensure_collections()
    doc_id = str(uuid.uuid4())

    # --- Text path ---
    chunks = chunk_document(pdf_path, doc_id)
    
    # Build BM25 index (in-memory + disk)
    build_index(chunks)
    
    texts = [c["chunk_text"] for c in chunks]
    text_embeddings = embed_text(texts, input_type="document")

    for chunk, embedding in zip(chunks, text_embeddings):
        upsert_text_chunk(
            chunk=chunk["chunk_text"],
            embedding=embedding,
            metadata={
                "doc_id": doc_id,
                "filename": filename,
                "page_num": chunk["page_num"],
                "chunk_index": chunk["chunk_index"],
                "raw_text": chunk["raw_text"],
            }
        )

    # --- Image path (Voyage multimodal-3.5) ---
    page_images = render_pdf_pages(pdf_path)
    image_embeddings = embed_images(page_images, input_type="document")

    for page_num, embedding in enumerate(image_embeddings, start=1):
        upsert_image_page(
            embedding=embedding,
            metadata={
                "doc_id": doc_id,
                "filename": filename,
                "page_num": page_num,
            }
        )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_indexed": len(chunks),
        "pages_indexed": len(image_embeddings),
    }