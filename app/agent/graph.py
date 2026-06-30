from langgraph.graph import StateGraph, END
from .nodes import (
    RAGState, classify_intent_node, route_after_intent,
    retrieve_node, grade_node,
    rewrite_query_node, generate_node, route_after_grade
)

def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("generate", generate_node)

    # Entry: classify intent first
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route_after_intent, {
        "retrieve": "retrieve",
        "generate": "generate",      # skip retrieval entirely
    })
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate",
        "rewrite": "rewrite",
    })
    graph.add_edge("rewrite", "retrieve")   # the loop
    graph.add_edge("generate", END)

    return graph.compile()

rag_graph = build_rag_graph()

