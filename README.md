# EDA Agent

A multi-agent Exploratory Data Analysis pipeline built with LangGraph and Streamlit. Upload any Excel or CSV file and get automated data cleaning, 21 interactive EDA charts, outlier detection, and a downloadable formatted Excel report.

---

## Features

- **Automated Data Cleaning** — messy headers, type inference, duplicates, nulls, inconsistent categoricals
- **21 Interactive EDA Charts** — histograms, violin plots, correlation heatmaps, Q-Q plots, scatter matrices, parallel coordinates, and more
- **Outlier Detection** — IQR + Z-score per numeric column with visual drill-down
- **AI Narrative** — plain English summary of findings powered by your chosen LLM
- **Formatted Excel Download** — 4-sheet report: cleaned data, EDA summary, outlier report, data changes
- **Provider-agnostic LLM** — Groq, Together AI, OpenRouter, Ollama, HuggingFace, Anthropic with automatic fallback chain

---

## Architecture

```
FileIngestion → HeaderCleaner → TypeInference → DataCleaner
                                                      │
                                          ┌───────────┴───────────┐
                                          ▼                       ▼
                                      EDAAgent           OutlierDetector
                                          └───────────┬───────────┘
                                                      ▼
                                               ReportCompiler (LLM)
```

Nodes 1–6 are pure Python (pandas, scipy). The LLM is only called in ReportCompiler for the narrative summary — all charts and analysis work without any API key.

---

## Quick Start

**1. Clone and install**
```bash
git clone https://github.com/gkaransail/eda-agent.git
cd eda-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Set up environment**
```bash
cp .env.example .env
# Add at least one API key (Groq is free — get one at console.groq.com)
```

**3. Run the app**
```bash
streamlit run app.py
```

Then open **http://localhost:8501** and upload an Excel or CSV file.

---

## LLM Providers

Configure your provider in `.env`:

```env
LLM_PROVIDER=groq   # groq | together | openrouter | ollama | huggingface | anthropic
GROQ_API_KEY=your_key_here
```

The fallback chain in `llm_hierarchy.yaml` defines priority order — if one provider fails (rate limit, quota), it automatically tries the next. Edit the file to reorder or add providers without changing any code.

| Provider | Free? | Notes |
|---|---|---|
| Groq | ✅ Free tier | Fastest, recommended default |
| Ollama | ✅ Free (local) | No internet needed, run `ollama pull llama3.1` |
| Together AI | Credits on signup | Good open-source models |
| OpenRouter | Pay-per-use | 100+ models via one API |
| HuggingFace | Free tier | Slower inference |
| Anthropic | Paid | Optional |

> The AI narrative is the only part that needs an LLM. All 21 charts, outlier detection, and Excel export work with no API key.

---

## EDA Charts (21 total)

| # | Chart |
|---|---|
| 1 | Histogram with rug plot |
| 2 | Box plot |
| 3 | Violin plot |
| 4 | Skewness & Kurtosis |
| 5 | Scatter plot with trend line |
| 6 | Scatter matrix (pairplot) |
| 7 | Top & Bottom 5 extremes |
| 8 | Correlation heatmap |
| 9 | Missing value heatmap |
| 10 | Row completeness distribution |
| 11 | Column cardinality |
| 12 | Grouped box plot (category × numeric) |
| 13 | Categorical distributions |
| 14 | CDF (Cumulative Distribution Function) |
| 15 | Q-Q Plot (normality check) |
| 16 | 2D Density Heatmap |
| 17 | Bubble chart (3 variables) |
| 18 | Cross-tab heatmap (category × category) |
| 19 | Parallel coordinates plot |
| 20 | Mean ± Std error bar chart |
| 21 | Strip plot |

---

## Project Structure

```
eda_agent/
├── app.py                  # Streamlit frontend
├── run.py                  # CLI runner (no UI)
├── llm_hierarchy.yaml      # LLM fallback chain config
├── create_dummy_data.py    # Generate sample test data
├── requirements.txt
├── .env.example
├── src/
│   ├── agents/             # One file per LangGraph node
│   │   ├── file_ingestion.py
│   │   ├── header_cleaner.py
│   │   ├── type_inference.py
│   │   ├── data_cleaner.py
│   │   ├── eda_agent.py
│   │   ├── outlier_detector.py
│   │   └── report_compiler.py
│   ├── graph/
│   │   └── graph.py        # LangGraph node wiring
│   ├── state/
│   │   └── state.py        # AnalysisState TypedDict
│   └── utils/
│       ├── llm.py          # LLM provider factory
│       ├── excel_exporter.py  # Formatted Excel report
│       └── helpers.py
└── data/samples/           # Sample test files
```

---

## CLI Usage

Run the pipeline without the UI:

```bash
python run.py data/samples/sales_data.xlsx
```

Outputs the AI narrative to terminal and saves a JSON report.

---

## Tech Stack

- **LangGraph** — multi-agent pipeline orchestration
- **Streamlit** — frontend
- **pandas / scipy / numpy** — data processing
- **Plotly** — interactive charts
- **openpyxl** — formatted Excel export
- **LangChain** — LLM provider abstraction
