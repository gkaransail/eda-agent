import pandas as pd

from src.state.state import AnalysisState


def file_ingestion(state: AnalysisState) -> dict:
    try:
        path = state["file_path"]
        if path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path, sheet_name=0)
        elif path.endswith(".csv"):
            df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
        else:
            return {"errors": [f"Unsupported file type: {path}"]}

        if df.empty:
            return {"errors": ["File loaded but DataFrame is empty."]}

        return {"raw_df": df, "cleaned_df": df.copy(), "errors": []}
    except Exception as e:
        return {"errors": [f"FileIngestion error: {e}"]}
