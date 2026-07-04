from pathlib import Path


def validate_file_path(path: str) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, f"File not found: {path}"
    if p.suffix not in {".xlsx", ".xls", ".csv"}:
        return False, f"Unsupported format: {p.suffix}. Use .xlsx, .xls, or .csv"
    return True, ""


def default_state(file_path: str) -> dict:
    return {
        "file_path": file_path,
        "raw_df": None,
        "cleaned_df": None,
        "column_map": {},
        "dtypes_report": {},
        "eda_report": {},
        "outlier_report": {},
        "errors": [],
        "final_report": {},
    }
