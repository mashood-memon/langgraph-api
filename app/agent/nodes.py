from langchain_openai import ChatOpenAI
from app.retrieval.hybrid import hybrid_search
from typing import TypedDict, Annotated
import operator

from app.config import get_settings

grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rewriter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


class RAGState(TypedDict):
    query: str
    chat_history: list[dict]      # [{"role": "user"/"assistant", "content": "..."}]
    rewritten_query: str | None
    retrieved_docs: list[dict]
    image_results: list[dict]
    relevance_score: float
    retry_count: int
    answer: str | None
    doc_context: str
    needs_retrieval: bool          # set by classify_intent_node
    steps: Annotated[list[dict], operator.add]


async def classify_intent_node(state: RAGState) -> RAGState:
    """
    Lightweight intent check: does this query need document retrieval,
    or can we answer from conversation history alone?
    """
    history = state.get("chat_history", [])[-4:]
    history_text = (
        "\n".join(f"{m['role']}: {m['content']}" for m in history)
        if history else "(none)"
    )

    prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"New user message: {state['query']}\n\n"
        "Does this message require searching external documents to answer, "
        "or can it be answered from the conversation context alone?\n"
        "Examples that do NOT need retrieval: greetings, thanks, chitchat, "
        "follow-up opinions, clarifications about what was just said.\n"
        "Examples that DO need retrieval: factual questions, requests for "
        "specific information, technical queries.\n\n"
        "Reply with exactly one word: RETRIEVE or RESPOND"
    )
    resp = await grader_llm.ainvoke(prompt)
    decision = resp.content.strip().upper()
    return {
        **state,
        "needs_retrieval": decision != "RESPOND",
        "steps": [{"name": "classify_intent", "status": "normal"}]
    }


def route_after_intent(state: RAGState) -> str:
    """Router: skip retrieval if the query doesn't need it."""
    if state.get("needs_retrieval", True):
        return "retrieve"
    return "generate"


async def retrieve_node(state: RAGState) -> RAGState:
    """Run hybrid search with current query (or rewritten query if available)."""
    active_query = state.get("rewritten_query") or state["query"]
    results = await hybrid_search(active_query, top_k=10)
    status = "retry" if state.get("retry_count", 0) > 0 else "normal"
    return {
        **state,
        "retrieved_docs": results["text_results"],
        "image_results": results["image_results"],
        "steps": [{"name": "retrieve", "status": status}]
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

    status = "retry" if state.get("retry_count", 0) > 0 else "normal"
    return {
        **state,
        "relevance_score": min(max(score, 0.0), 1.0),
        "steps": [{"name": "grade", "status": status}]
    }


async def rewrite_query_node(state: RAGState) -> RAGState:
    history = state.get("chat_history", [])[-4:]   
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Current query: {state['query']}\n\n"
        f"The current query may reference earlier turns (e.g. 'it', 'that error', 'the same service'). "
        f"Rewrite it to be fully self-contained and retrieval-friendly, resolving any such references. "
        f"Return ONLY the rewritten query."
    )
    resp = await rewriter_llm.ainvoke(prompt)
    return {
        **state,
        "rewritten_query": resp.content.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
        "steps": [{"name": "rewrite", "status": "retry"}]
    }


async def generate_node(state: RAGState) -> RAGState:
    # Only keep docs that clear a per-doc relevance bar, then cap at 6
    MIN_RRF_SCORE = 0.015
    relevant_docs = [
        d for d in state.get("retrieved_docs", [])
        if d.get("rrf_score", 0) >= MIN_RRF_SCORE
    ][:6]

    if not relevant_docs:
        context = "(no relevant documents found)"
    else:
        context = "\n\n".join(
            d.get("text", d.get("raw_text", "")) for d in relevant_docs
        )

    history = state.get("chat_history", [])[-4:]
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history) if history else "(no prior conversation)"

    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"Context from retrieved documents:\n{context}\n\n"
        f"Current query: {state['query']}\n\n"
        f"Answer concisely and precisely, taking the conversation above into account "
        f"if the query refers back to it. If information is missing, say so."
    )
    resp = await grader_llm.ainvoke(prompt)
    status = "retry" if state.get("retry_count", 0) > 0 else "normal"
    return {
        **state, 
        "answer": resp.content,
        "steps": [{"name": "generate", "status": status}]
    }


def route_after_grade(state: RAGState) -> str:
    """Router: decide whether to generate or rewrite based on relevance score."""
    score = state.get("relevance_score", 0)
    retries = state.get("retry_count", 0)
    settings = get_settings()

    if score >= settings.relevance_threshold or retries >= settings.max_retries:
        return "generate"
    return "rewrite"