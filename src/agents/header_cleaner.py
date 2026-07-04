import re

from src.state.state import AnalysisState


def _clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name


def header_cleaner(state: AnalysisState) -> dict:
    try:
        df = state["cleaned_df"].copy()
        original_cols = df.columns.tolist()
        cleaned_cols = [_clean_column_name(c) for c in original_cols]

        seen: dict[str, int] = {}
        deduped = []
        for col in cleaned_cols:
            if col in seen:
                seen[col] += 1
                deduped.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                deduped.append(col)

        column_map = dict(zip(original_cols, deduped))
        df.columns = deduped
        return {"cleaned_df": df, "column_map": column_map, "errors": []}
    except Exception as e:
        return {"errors": [f"HeaderCleaner error: {e}"]}
