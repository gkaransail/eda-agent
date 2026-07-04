import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.graph.graph import build_graph
from src.utils.helpers import default_state, validate_file_path


def main(file_path: str):
    valid, err = validate_file_path(file_path)
    if not valid:
        print(f"Error: {err}")
        sys.exit(1)

    print(f"Running EDA pipeline on: {file_path}")
    graph = build_graph()
    state = graph.invoke(default_state(file_path))

    if state["errors"]:
        print("\nErrors encountered:")
        for e in state["errors"]:
            print(f"  - {e}")

    print("\n--- Final Report ---")
    print(state["final_report"].get("narrative", "No narrative generated."))

    output_path = Path(file_path).stem + "_report.json"
    with open(output_path, "w") as f:
        report = state["final_report"].copy()
        report.pop("eda", None)   # strip large df-derived dicts for readable output
        json.dump(report, f, indent=2, default=str)

    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <path_to_excel_or_csv>")
        sys.exit(1)
    main(sys.argv[1])
