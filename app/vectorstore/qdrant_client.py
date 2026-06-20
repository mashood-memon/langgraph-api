from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import os
from app.config import get_settings


VECTOR_DIM = 1024   # voyage-multimodal-3.5 output dimension

_s = get_settings()
client = QdrantClient(url=_s.qdrant_url, api_key=_s.qdrant_api_key)

COLLECTION_NAME = "document_content"


def ensure_collections():
    """Create the shared collection if it doesn't exist."""
    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def upsert_text_chunk(chunk: str, embedding: list[float], metadata: dict) -> str:
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=point_id,
            vector=embedding,
            payload=metadata | {"text": chunk, "content_type": "text"}
        )]
    )
    return point_id


def upsert_image_page(embedding: list[float], metadata: dict) -> str:
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=point_id,
            vector=embedding,
            payload=metadata | {"content_type": "image"}
        )]
    )
    return point_id


def search_text(query_embedding: list[float], top_k: int = 10) -> list[dict]:
    """Searches the shared collection. Use payload filter if you want text-only or image-only results."""
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"text": r.payload.get("text", ""), "score": r.score, "metadata": r.payload}
        for r in results
    ]


def search_images(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Same collection, filtered to image-type payloads only."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        query_filter=Filter(must=[FieldCondition(key="content_type", match=MatchValue(value="image"))]),
        limit=top_k,
        with_payload=True,
    )
    return [{"score": r.score, "metadata": r.payload} for r in results]