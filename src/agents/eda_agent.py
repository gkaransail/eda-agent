from src.state.state import AnalysisState


def eda_agent(state: AnalysisState) -> dict:
    try:
        df = state["cleaned_df"]
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        eda_report = {
            "shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
            "summary_stats": df[num_cols].describe().round(4).to_dict() if num_cols else {},
            "top_value_counts": {
                col: df[col].value_counts().head(5).to_dict() for col in cat_cols
            },
            "correlations": (
                df[num_cols].corr().round(4).to_dict() if len(num_cols) >= 2 else {}
            ),
        }

        return {"eda_report": eda_report, "errors": []}
    except Exception as e:
        return {"errors": [f"EDAAgent error: {e}"]}
