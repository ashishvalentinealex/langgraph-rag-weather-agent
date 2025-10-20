# src/graph.py
"""LangGraph workflow for combining weather queries and PDF-based RAG.

The ``build_pipeline`` function creates a LangGraph graph that processes an
input question and decides whether to fetch real-time weather data or perform
retrieval-augmented generation over a PDF. The graph contains:

* decision – classify the user question as "weather" or "pdf".
* weather  – call OpenWeather One Call 3.0 and summarise current conditions.
* rag      – retrieve the most relevant chunks from Qdrant.
* answer   – generate the final answer with the LLM and optional context.
"""

from __future__ import annotations

from typing import TypedDict, Optional, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from .weather import fetch_weather_for_city
from .rag import retrieve_context


class GraphState(TypedDict):
    """Type for the graph state.

    question: user input
    route:    output of decision node ("weather" or "pdf")
    context:  retrieved info (weather summary or RAG text)
    answer:   final model answer
    """
    question: str
    route: Optional[str]
    context: Optional[str]
    answer: Optional[str]


def build_pipeline(
    llm: ChatOpenAI,
    vector_store,
) -> StateGraph:
    """Create and compile a LangGraph pipeline."""

    # --- node functions ---
    def decision_node(state: GraphState) -> Dict[str, Any]:
        """Decide whether the question is about weather or the PDF."""
        question = state["question"]
        messages = [
            SystemMessage(content="You are a classifier. If the user is asking about the weather, "
                                  "output 'weather'. Otherwise output 'pdf'. Return only that word."),
            HumanMessage(content=question),
        ]
        resp = llm.invoke(messages)
        route = (resp.content or "").strip().lower()
        route = "weather" if "weather" in route else "pdf"
        print(f"DEBUG decision: {route} for question: {question}")
        return {"route": route}

    def weather_node(state: GraphState) -> Dict[str, Any]:
        """Fetch and summarise weather using OpenWeather One Call 3.0."""
        question = state["question"]
        print("DEBUG question:", question)
        try:
            summary = fetch_weather_for_city(
                user_query=question,
                default_city="Hyderabad",
                country_hint="IN",
            )
        except Exception as ex:
            summary = f"Sorry, I couldn’t fetch the weather right now: {ex}"
        return {"context": summary, "route": "weather"}

    def rag_node(state: GraphState) -> Dict[str, Any]:
        """Retrieve context from the PDF via similarity search."""
        question = state["question"]
        context = retrieve_context(vector_store, question, k=5) or ""
        return {"context": context, "route": "pdf"}

    def answer_node(state: GraphState) -> Dict[str, Any]:
        """Generate the final answer using the LLM and available context."""
        question = state["question"]
        context = state.get("context") or ""

        # Force the model to use the context, not ignore it
        if context.strip():
            system_prompt = (
                "You are a helpful assistant. "
                "The following context contains the exact answer to the user's question. "
                "Use the context directly and DO NOT reply with 'I don't know'. "
                "Respond concisely.\n\n"
                f"Context:\n{context}\n\n"
            )
        else:
            system_prompt = (
                "You are a helpful assistant. Answer the question to the best of your ability."
            )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ]
        resp = llm.invoke(messages)
        answer = (resp.content or "").strip()
        print("DEBUG answer:", answer)
        return {"answer": answer}

    # --- graph wiring (LangGraph v1) ---
    graph = StateGraph(GraphState)
    graph.add_node("decision", decision_node)
    graph.add_node("weather", weather_node)
    graph.add_node("rag", rag_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("decision")

    def route_edge(state: Dict[str, Any]) -> str:
        return state["route"]

    graph.add_conditional_edges(
        "decision",
        route_edge,
        {"weather": "weather", "pdf": "rag"},
    )

    graph.add_edge("weather", "answer")
    graph.add_edge("rag", "answer")
    graph.add_edge("answer", END)  

    return graph.compile()
