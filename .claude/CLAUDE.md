# EDA Agent — Claude Context

## Project Purpose
Multi-agent pipeline that ingests an Excel/CSV file, cleans it, runs EDA, detects outliers, and compiles a structured report. Built with LangGraph (backend) + Streamlit (frontend, Phase 4).

## Architecture

### Agent Graph (LangGraph)
```
FileIngestion → HeaderCleaner → TypeInference → DataCleaner
                                                      │
                                          ┌───────────┴───────────┐
                                          ▼                       ▼
                                      EDAAgent           OutlierDetector
                                          └───────────┬───────────┘
                                                      ▼
                                               ReportCompiler
```

### Key Rule: LLM only at ReportCompiler
Nodes 1–6 are pure Python (pandas, scipy, re). Claude is only called in ReportCompiler to narrate findings in plain English. Keep agents deterministic and cheap.

## Directory Structure
```
src/
  agents/         # One file per LangGraph node
  graph/          # graph.py — assembles nodes + edges
  state/          # state.py — AnalysisState TypedDict
  utils/          # shared helpers (file parsing, dtype utils)
data/samples/     # test Excel/CSV files
tests/            # unit tests per agent node
```

## State Object
All agents read from and write to `AnalysisState`. Never pass data between agents any other way.

```python
class AnalysisState(TypedDict):
    file_path: str
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    column_map: dict          # original → cleaned header names
    dtypes_report: dict       # inferred vs original types
    eda_report: dict          # summary stats, distributions, correlations
    outlier_report: dict      # flagged rows + method used
    errors: list[str]
    final_report: dict
```

## Coding Conventions
- Each agent node is a standalone function: `def node_name(state: AnalysisState) -> AnalysisState`
- Nodes never raise exceptions — catch errors, append to `state["errors"]`, return state
- All DataFrames stay as pandas — no polars, no dask unless explicitly discussed
- Type hints everywhere
- No comments explaining what code does — only why (non-obvious constraints)

## Environment
- Python 3.11+
- Key deps: langgraph, langchain-core, pandas, scipy, openpyxl, python-dotenv
- LLM is provider-agnostic — set LLM_PROVIDER in .env (groq | together | openrouter | ollama | huggingface | anthropic)
- Default provider: Groq (free tier, fast). Anthropic is optional.
- LLM factory lives in src/utils/llm.py — only place to change if adding a new provider

## Phases
- Phase 1: Nodes 1–4 (ingestion + cleaning pipeline) ← current
- Phase 2: Nodes 5–6 (EDA + outliers)
- Phase 3: Node 7 (report compiler with Claude)
- Phase 4: Streamlit wrapper
