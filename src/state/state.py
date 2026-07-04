import operator
from typing import Annotated, TypedDict

import pandas as pd


class AnalysisState(TypedDict):
    file_path: str
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    column_map: dict
    dtypes_report: dict
    eda_report: dict
    outlier_report: dict
    errors: Annotated[list[str], operator.add]  # reducer merges parallel writes
    final_report: dict
