from langgraph.graph import StateGraph, END
from .nodes import (
    RAGState, retrieve_node, grade_node,
    rewrite_query_node, generate_node, route_after_grade
)

def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate",
        "rewrite": "rewrite",
    })
    graph.add_edge("rewrite", "retrieve")   # the loop
    graph.add_edge("generate", END)

    return graph.compile()

rag_graph = build_rag_graph()
