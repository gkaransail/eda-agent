from src.state.state import AnalysisState


def data_cleaner(state: AnalysisState) -> dict:
    try:
        df = state["cleaned_df"].copy()

        str_cols = df.select_dtypes(include="object").columns
        df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

        for col in str_cols:
            if df[col].nunique() <= 20:
                df[col] = df[col].str.lower()

        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        return {"cleaned_df": df, "errors": []}
    except Exception as e:
        return {"errors": [f"DataCleaner error: {e}"]}
