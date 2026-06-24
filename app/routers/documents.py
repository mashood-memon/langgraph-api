from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile, os, shutil
from app.ingestion.pipeline import ingest_document
from app.agent.graph import rag_graph

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF document."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs supported.")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        stats = await ingest_document(tmp_path, file.filename)
        return JSONResponse(content={"status": "ingested", **stats})
    finally:
        os.unlink(tmp_path)


@router.post("/query")
async def query_documents(payload: dict):
    """
    Query ingested documents via self-correcting RAG loop.
    Body: {"query": "your question here"}
    """
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    initial_state = {
        "query": query,
        "rewritten_query": None,
        "retrieved_docs": [],
        "image_results": [],
        "relevance_score": 0.0,
        "retry_count": 0,
        "answer": None,
        "doc_context": "",
    }

    result = await rag_graph.ainvoke(initial_state)

    return {
        "answer": result["answer"],
        "query_used": result.get("rewritten_query") or query,
        "retries": result["retry_count"],
        "relevance_score": result["relevance_score"],
        "sources": [
            {
                "filename": d.get("metadata", {}).get("filename"),
                "page": d.get("metadata", {}).get("page_num"),
            }
            for d in result["retrieved_docs"][:5]
        ],
    }