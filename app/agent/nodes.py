from langchain_openai import ChatOpenAI
from app.retrieval.hybrid import hybrid_search
from typing import TypedDict, Annotated
import operator

grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

MAX_RETRIES = 3   # hard cap — controls API cost


class RAGState(TypedDict):
    query: str
    rewritten_query: str | None
    retrieved_docs: list[dict]
    image_results: list[dict]
    relevance_score: float
    retry_count: int
    answer: str | None
    doc_context: str


async def retrieve_node(state: RAGState) -> RAGState:
    """Run hybrid search with current query (or rewritten query if available)."""
    active_query = state.get("rewritten_query") or state["query"]
    results = await hybrid_search(active_query, top_k=10)
    return {
        **state,
        "retrieved_docs": results["text_results"],
        "image_results": results["image_results"],
    }


async def grade_node(state: RAGState) -> RAGState:
    """
    Score relevance of retrieved docs to the query.
    Returns average score 0-1. Threshold: 0.5.
    """
    docs_text = "\n\n---\n\n".join(
        d.get("text", d.get("raw_text", ""))[:500]
        for d in state["retrieved_docs"][:5]
    )
    prompt = (
        f"Query: {state['query']}\n\n"
        f"Retrieved documents:\n{docs_text}\n\n"
        f"Rate the relevance of these documents to the query from 0.0 to 1.0. "
        f"Return ONLY a float number, nothing else."
    )
    resp = await grader_llm.ainvoke(prompt)
    try:
        score = float(resp.content.strip())
    except ValueError:
        score = 0.5   # default if LLM doesn't comply

    return {**state, "relevance_score": min(max(score, 0.0), 1.0)}


async def rewrite_query_node(state: RAGState) -> RAGState:
    """Rewrite the query to improve retrieval on next attempt."""
    prompt = (
        f"The following query returned irrelevant technical documents.\n"
        f"Original query: {state['query']}\n"
        f"Rewrite it to be more specific and retrieval-friendly. "
        f"Focus on technical identifiers, error codes, or system names. "
        f"Return ONLY the rewritten query."
    )
    resp = await rewriter_llm.ainvoke(prompt)
    return {
        **state,
        "rewritten_query": resp.content.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
    }


async def generate_node(state: RAGState) -> RAGState:
    """Generate final answer from retrieved context."""
    context = "\n\n".join(
        d.get("text", d.get("raw_text", ""))
        for d in state["retrieved_docs"][:6]
    )
    has_images = len(state.get("image_results", [])) > 0

    image_note = "[Note: Relevant diagram/image pages were also retrieved and considered.]\n\n" if has_images else ""

    prompt = (
        f"You are a technical documentation expert.\n\n"
        f"Context from retrieved documents:\n{context}\n\n"
        f"{image_note}"
        f"Query: {state['query']}\n\n"
        f"Answer concisely and precisely. If information is missing, say so."
    )

    resp = await grader_llm.ainvoke(prompt)
    return {**state, "answer": resp.content}


def route_after_grade(state: RAGState) -> str:
    """Router: decide whether to generate or rewrite based on relevance score."""
    score = state.get("relevance_score", 0)
    retries = state.get("retry_count", 0)

    if score >= 0.5 or retries >= MAX_RETRIES:
        return "generate"
    return "rewrite"