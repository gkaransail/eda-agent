import pandas as pd

from src.state.state import AnalysisState


def _try_parse_datetime(series: pd.Series) -> pd.Series | None:
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        if parsed.notna().sum() / max(len(parsed), 1) >= 0.8:
            return parsed
    except Exception:
        pass
    return None


def _try_parse_numeric(series: pd.Series) -> pd.Series | None:
    try:
        cleaned = series.astype(str).str.replace(r"[,$%]", "", regex=True)
        parsed = pd.to_numeric(cleaned, errors="coerce")
        if parsed.notna().sum() / max(len(parsed), 1) >= 0.8:
            return parsed
    except Exception:
        pass
    return None


def type_inference(state: AnalysisState) -> dict:
    try:
        df = state["cleaned_df"].copy()
        dtypes_report: dict[str, dict] = {}

        for col in df.columns:
            original_dtype = str(df[col].dtype)
            action = "unchanged"

            if df[col].dtype == object:
                dt = _try_parse_datetime(df[col])
                if dt is not None:
                    df[col] = dt
                    action = "cast_to_datetime"
                else:
                    num = _try_parse_numeric(df[col])
                    if num is not None:
                        df[col] = num
                        action = "cast_to_numeric"

            dtypes_report[col] = {
                "original": original_dtype,
                "inferred": str(df[col].dtype),
                "action": action,
            }

        return {"cleaned_df": df, "dtypes_report": dtypes_report, "errors": []}
    except Exception as e:
        return {"errors": [f"TypeInference error: {e}"]}
