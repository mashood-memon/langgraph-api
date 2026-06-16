from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, SearchRequest
)
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")

# Two collections:
# - "text_chunks"  : 1536-dim OpenAI text embeddings
# - "image_pages"  : 128-dim ColPali image embeddings (per-patch, multi-vector)

TEXT_DIM = 1536
IMAGE_DIM = 128   # ColPali patch embedding dim

client = QdrantClient(url=QDRANT_URL)


def ensure_collections():
    """Create collections if they don't exist."""
    existing = {c.name for c in client.get_collections().collections}

    if "text_chunks" not in existing:
        client.create_collection(
            collection_name="text_chunks",
            vectors_config=VectorParams(size=TEXT_DIM, distance=Distance.COSINE),
        )

    if "image_pages" not in existing:
        # ColPali uses multi-vector (one vector per image patch)
        # Qdrant supports this via named vectors or multi-vectors
        client.create_collection(
            collection_name="image_pages",
            vectors_config=VectorParams(size=IMAGE_DIM, distance=Distance.COSINE),
        )


def upsert_text_chunk(chunk: str, embedding: list[float], metadata: dict) -> str:
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name="text_chunks",
        points=[PointStruct(id=point_id, vector=embedding, payload=metadata | {"text": chunk})]
    )
    return point_id


def search_text(query_embedding: list[float], top_k: int = 10) -> list[dict]:
    results = client.search(
        collection_name="text_chunks",
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    return [{"text": r.payload["text"], "score": r.score, "metadata": r.payload} for r in results]


def upsert_image_page(embedding: list[float], metadata: dict) -> str:
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name="image_pages",
        points=[PointStruct(id=point_id, vector=embedding, payload=metadata)]
    )
    return point_id


def search_images(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    results = client.search(
        collection_name="image_pages",
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    return [{"score": r.score, "metadata": r.payload} for r in results]
