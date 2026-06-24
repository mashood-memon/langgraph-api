from app.vectorstore.qdrant_client import search_text, search_images
from app.ingestion.bm25_index import search_bm25
from app.ingestion.voyage_embed import embed_query

RRF_K = 60   # standard constant; higher = smoother ranking


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], id_key: str = "text") -> list[dict]:
    """
    Merge multiple ranked lists via RRF.
    Each item needs a unique id_key for deduplication.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            item_id = item.get(id_key) or item.get("metadata", {}).get("chunk_index", str(rank))
            item_id = str(item_id)
            scores[item_id] = scores.get(item_id, 0) + 1 / (RRF_K + rank + 1)
            items[item_id] = item

    sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [items[i] | {"rrf_score": scores[i]} for i in sorted_ids]


async def hybrid_search(query: str, top_k: int = 10) -> dict:
    """
    Run semantic vector search + BM25 keyword search, merge via RRF.
    Also runs image search in the same Qdrant collection (filtered by content_type).
    One query embedding call covers both — text and images share the same vector space.
    Returns {text_results, image_results}.
    """
    query_embedding = embed_query(query)

    # Vector search (text content)
    vector_results = search_text(query_embedding, top_k=top_k)

    # BM25 keyword search
    bm25_results = search_bm25(query, top_k=top_k)

    # RRF merge
    fused_text = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        id_key="text"
    )[:top_k]

    # Image search — same embedding, filtered to image-type payloads
    image_results = search_images(query_embedding, top_k=5)

    return {
        "text_results": fused_text,
        "image_results": image_results,
    }
