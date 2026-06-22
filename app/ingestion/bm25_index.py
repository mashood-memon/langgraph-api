from rank_bm25 import BM25Okapi
import pickle
import os
import re

BM25_INDEX_PATH = "/tmp/bm25_index.pkl"

_bm25 = None
_corpus_chunks = []   # parallel list to BM25 index


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def build_index(chunks: list[dict]):
    """Build BM25 index from chunk dicts. Call after ingestion."""
    global _bm25, _corpus_chunks
    _corpus_chunks = chunks
    tokenized = [tokenize(c["raw_text"]) for c in chunks]
    _bm25 = BM25Okapi(tokenized)
    # Persist to disk
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump((_bm25, _corpus_chunks), f)


def load_index():
    global _bm25, _corpus_chunks
    if _bm25 is None and os.path.exists(BM25_INDEX_PATH):
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25, _corpus_chunks = pickle.load(f)


def search_bm25(query: str, top_k: int = 10) -> list[dict]:
    """Returns top_k chunks with BM25 scores."""
    load_index()
    if _bm25 is None:
        return []
    tokens = tokenize(query)
    scores = _bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {"text": _corpus_chunks[i]["raw_text"], "score": scores[i], "metadata": _corpus_chunks[i], "rank": rank}
        for rank, i in enumerate(top_indices)
    ]