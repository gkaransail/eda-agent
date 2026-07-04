from langgraph.graph import END, START, StateGraph

from src.agents.data_cleaner import data_cleaner
from src.agents.eda_agent import eda_agent
from src.agents.file_ingestion import file_ingestion
from src.agents.header_cleaner import header_cleaner
from src.agents.outlier_detector import outlier_detector
from src.agents.report_compiler import report_compiler
from src.agents.type_inference import type_inference
from src.state.state import AnalysisState


def _has_errors(state: AnalysisState) -> str:
    return "error" if state.get("errors") else "continue"


def build_graph() -> StateGraph:
    graph = StateGraph(AnalysisState)

    graph.add_node("file_ingestion", file_ingestion)
    graph.add_node("header_cleaner", header_cleaner)
    graph.add_node("type_inference", type_inference)
    graph.add_node("data_cleaner", data_cleaner)
    graph.add_node("eda_agent", eda_agent)
    graph.add_node("outlier_detector", outlier_detector)
    graph.add_node("report_compiler", report_compiler)

    graph.add_edge(START, "file_ingestion")
    graph.add_conditional_edges("file_ingestion", _has_errors, {"error": END, "continue": "header_cleaner"})
    graph.add_edge("header_cleaner", "type_inference")
    graph.add_edge("type_inference", "data_cleaner")

    # parallel fan-out: both EDA and outlier run on cleaned_df
    graph.add_edge("data_cleaner", "eda_agent")
    graph.add_edge("data_cleaner", "outlier_detector")

    # fan-in: both must complete before report compiler
    graph.add_edge(["eda_agent", "outlier_detector"], "report_compiler")
    graph.add_edge("report_compiler", END)

    return graph.compile()
