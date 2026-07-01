from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
import tempfile, os, shutil, logging, hashlib, uuid
from app.ingestion.pipeline import ingest_document
from app.agent.graph import rag_graph
from app.config import get_settings
from app.limiter import limiter
from app.db.session import get_session, async_session
from app.db.conversations import create_conversation, save_message, get_recent_messages
from app.db.models import Document
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# --- API Key auth (upload only) ---
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_upload_key(key: str = Security(api_key_header)):
    """FastAPI dependency — runs before the endpoint body."""
    if key != get_settings().upload_api_key:
        raise HTTPException(status_code=403, detail="Invalid API key.")


def _hash_file(path: str) -> str:
    """SHA-256 hash of file contents — deterministic fingerprint for dedup."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_already_ingested(file_hash: str) -> bool:
    """Check Qdrant for any point with this file_hash in its payload."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from app.vectorstore.qdrant_client import client, COLLECTION_NAME

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="file_hash", match=MatchValue(value=file_hash))
        ]),
        limit=1,
    )
    return len(points) > 0


async def ingest_and_cleanup(tmp_path: str, filename: str, file_hash: str, doc_id: str):
    """
    Runs AFTER the HTTP response is already sent to the client.
    Wrapper exists to guarantee temp file cleanup even if ingestion crashes.
    """
    try:
        stats = await ingest_document(tmp_path, filename, doc_id, file_hash=file_hash)
        async with async_session() as session:
            doc = await session.get(Document, doc_id)
            if doc:
                doc.status = "ready"
                doc.pipeline_stats = stats
                doc.page_count = stats.get("pages_indexed", 0)
                await session.commit()
        logger.info(f"Ingestion complete: {filename}", extra={"stats": stats})
    except Exception as e:
        logger.error(f"Ingestion failed for {filename}: {e}")
        async with async_session() as session:
            doc = await session.get(Document, doc_id)
            if doc:
                doc.status = "error"
                await session.commit()
    finally:
        os.unlink(tmp_path)   # always clean up the temp file


@router.post("/upload", status_code=202, dependencies=[Depends(verify_upload_key)])
@limiter.limit("5/minute")
async def upload_document(request: Request, file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks(), session=Depends(get_session)):
    settings = get_settings()
    if file.size and file.size > settings.max_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_size_mb}MB limit.")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs supported.")

    tmp_path = None
    try:
        # Save to temp file (fast, sync — happens before response)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Dedup check — hash the file and see if Qdrant already has it
        file_hash = _hash_file(tmp_path)
        if _file_already_ingested(file_hash):
            os.unlink(tmp_path)
            return JSONResponse(
                status_code=409,
                content={"status": "duplicate", "detail": "This file has already been ingested."},
            )

        doc_id = str(uuid.uuid4())
        new_doc = Document(
            id=doc_id,
            filename=file.filename,
            size_mb=round(file.size / (1024 * 1024), 2) if file.size else 0.0,
            status="processing"
        )
        session.add(new_doc)
        await session.commit()
        await session.refresh(new_doc)

        # Queue ingestion to run AFTER this response is sent
        background_tasks.add_task(ingest_and_cleanup, tmp_path, file.filename, file_hash, doc_id)

        # Client gets this immediately — doesn't wait for ingestion
        return JSONResponse(
            status_code=202,
            content={
                "docId": str(new_doc.id),
                "filename": new_doc.filename,
                "pageCount": new_doc.page_count,
                "sizeMb": new_doc.size_mb,
                "uploadedAt": new_doc.uploaded_at.isoformat() if new_doc.uploaded_at else None,
                "status": new_doc.status,
                "pipelineStats": new_doc.pipeline_stats
            },
        )
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass  # best effort cleanup
        logger.error(f"Upload preparation failed for {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while preparing the document for ingestion. Please try again."
        )


from app.middleware.cache import get_cached_response, set_cached_response

@router.post("/query")
async def query_documents(payload: dict, session=Depends(get_session)):
    query = payload.get("query", "").strip()
    user_id = payload.get("user_id", "anonymous")
    conversation_id = payload.get("conversation_id")
    doc_ids = payload.get("doc_ids")

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Check cache for first-turn queries only
    if not conversation_id:
        cached = await get_cached_response(query, doc_ids)
        if cached:
            # We have a hit! Create a conversation so the user can still follow up
            conversation_id = await create_conversation(session, user_id=user_id, title=query[:60])
            await save_message(session, conversation_id, role="user", content=query)
            await save_message(session, conversation_id, role="assistant", content=cached["answer"], sources=cached.get("sources"))
            cached["conversation_id"] = conversation_id
            return cached

    try:
        if not conversation_id:
            conversation_id = await create_conversation(session, user_id=user_id, title=query[:60])

        history = await get_recent_messages(session, conversation_id, limit=10)
        await save_message(session, conversation_id, role="user", content=query)

        initial_state = {
            "query": query,
            "chat_history": history,
            "rewritten_query": None,
            "retrieved_docs": [],
            "image_results": [],
            "relevance_score": 0.0,
            "retry_count": 0,
            "answer": None,
            "doc_context": "",
        }

        result = await rag_graph.ainvoke(initial_state)

        sources = [
            {"filename": d.get("metadata", {}).get("filename"), "page": d.get("metadata", {}).get("page_num")}
            for d in result.get("retrieved_docs", [])[:5]
        ]
        await save_message(session, conversation_id, role="assistant", content=result["answer"], sources=sources)

        response = {
            "conversationId": str(conversation_id),
            "answer": result["answer"],
            "sources": sources,
            "traceData": {
                "retries": result["retry_count"],
                "maxRetries": get_settings().max_retries,
                "relevanceScore": result["relevance_score"],
                "rewrittenQuery": result.get("rewritten_query"),
                "steps": result.get("steps", [])
            }
        }

        # Only cache if it was a context-free, first-turn query
        if not history:
            await set_cached_response(query, response, doc_ids)

        return response

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your query. Please try again."
        )


@router.get("")
async def get_documents(session=Depends(get_session)):
    try:
        result = await session.execute(select(Document).order_by(Document.uploaded_at.desc()))
        docs = result.scalars().all()
        return [
            {
                "docId": str(d.id),
                "filename": d.filename,
                "pageCount": d.page_count,
                "sizeMb": d.size_mb,
                "uploadedAt": d.uploaded_at.isoformat() if d.uploaded_at else None,
                "status": d.status,
                "pipelineStats": d.pipeline_stats
            } for d in docs
        ]
    except Exception as e:
        logger.error(f"Failed to fetch documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch documents. Please try again."
        )


@router.get("/conversations")
async def get_conversations(user_id: str, session=Depends(get_session)):
    try:
        return await list_conversations(session, user_id)
    except Exception as e:
        logger.error(f"Failed to fetch conversations for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch conversations. Please try again."
        )


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str, session=Depends(get_session)):
    try:
        return await get_recent_messages(session, conversation_id, limit=100)
    except Exception as e:
        logger.error(f"Failed to fetch messages for conversation {conversation_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch conversation history. Please try again."
        )