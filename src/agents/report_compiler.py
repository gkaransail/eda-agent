from langchain_core.messages import HumanMessage

from src.state.state import AnalysisState
from src.utils.llm import get_llm


def report_compiler(state: AnalysisState) -> dict:
    try:
        context = {
            "shape": state.get("eda_report", {}).get("shape"),
            "null_pct": state.get("eda_report", {}).get("null_pct"),
            "dtypes": state.get("dtypes_report"),
            "outlier_summary": {
                col: {
                    "iqr_count": data["iqr_outlier_count"],
                    "z_count": data["z_score_outlier_count"],
                }
                for col, data in state.get("outlier_report", {}).get("columns", {}).items()
            },
            "errors": state.get("errors", []),
        }

        prompt = f"""You are a data analyst. Summarize the following analysis results in plain English.
Be concise. Highlight the most important findings: data quality issues, notable distributions,
significant outliers, and recommended next steps.

Analysis results:
{context}
"""
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])

        return {
            "final_report": {
                "narrative": response.content,
                "eda": state.get("eda_report", {}),
                "outliers": state.get("outlier_report", {}),
                "dtypes": state.get("dtypes_report", {}),
                "column_map": state.get("column_map", {}),
                "errors": state.get("errors", []),
            },
            "errors": [],
        }
    except Exception as e:
        return {"errors": [f"ReportCompiler error: {e}"], "final_report": {}}
