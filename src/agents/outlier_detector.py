import numpy as np
from scipy import stats

from src.state.state import AnalysisState

_IQR_FACTOR = 1.5
_Z_THRESHOLD = 3.0


def outlier_detector(state: AnalysisState) -> dict:
    try:
        df = state["cleaned_df"]
        num_cols = df.select_dtypes(include="number").columns.tolist()
        flagged: dict[str, dict] = {}

        for col in num_cols:
            series = df[col].dropna()
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            iqr_mask = (df[col] < q1 - _IQR_FACTOR * iqr) | (df[col] > q3 + _IQR_FACTOR * iqr)

            z_scores = np.abs(stats.zscore(series))
            z_outlier_indices = series.index[z_scores > _Z_THRESHOLD].tolist()

            flagged[col] = {
                "iqr_outlier_count": int(iqr_mask.sum()),
                "iqr_outlier_indices": df.index[iqr_mask].tolist(),
                "z_score_outlier_count": len(z_outlier_indices),
                "z_score_outlier_indices": z_outlier_indices,
                "bounds": {
                    "lower": round(q1 - _IQR_FACTOR * iqr, 4),
                    "upper": round(q3 + _IQR_FACTOR * iqr, 4),
                },
            }

        return {"outlier_report": {"columns": flagged, "method": "IQR + Z-score"}, "errors": []}
    except Exception as e:
        return {"errors": [f"OutlierDetector error: {e}"]}
