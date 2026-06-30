from rank_bm25 import BM25Okapi
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from app.vectorstore.qdrant_client import client, COLLECTION_NAME
import re


def tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def search_bm25(query: str, doc_ids: list[str] | None = None, top_k: int = 10) -> list[dict]:
    """
    Build BM25 on-the-fly from Qdrant payloads, scoped to doc_ids if given.
    """
    must_conditions = [FieldCondition(key="content_type", match=MatchValue(value="text"))]
    if doc_ids:
        must_conditions.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=must_conditions),
        limit=1000,   # adjust if a single user's corpus exceeds this
        with_payload=True,
    )

    if not points:
        return []

    corpus = [p.payload.get("raw_text", "") for p in points]
    tokenized_corpus = [tokenize(t) for t in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {"text": corpus[i], "score": float(scores[i]), "metadata": points[i].payload}
        for i in top_indices
    ]