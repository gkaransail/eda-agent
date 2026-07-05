import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="EDA Agent", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 EDA Agent")
    st.caption("Multi-agent data analysis pipeline")
    st.divider()
    provider = os.getenv("LLM_PROVIDER", "groq").upper()
    model = os.getenv(f"{provider}_MODEL", "—")
    st.markdown(f"**LLM Provider:** `{provider}`")
    st.markdown(f"**Model:** `{model}`")
    st.divider()
    hierarchy_path = Path("llm_hierarchy.yaml")
    if hierarchy_path.exists():
        import yaml
        with open(hierarchy_path) as f:
            h = yaml.safe_load(f)
        st.markdown("**Fallback Chain**")
        for i, p in enumerate(h.get("providers", [])):
            st.caption(f"{'🟢' if i == 0 else str(i) + '.'} {p['name']}")
    st.divider()
    st.caption("Upload an Excel or CSV file to begin.")

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Exploratory Data Analyst")
st.markdown("Upload a file — the agent pipeline cleans, analyses, and summarises your data automatically.")

uploaded = st.file_uploader("Upload Excel or CSV", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Upload a file above to get started.")
    st.stop()

cache_key = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("cache_key") != cache_key:
    st.session_state.pop("result", None)
    st.session_state["cache_key"] = cache_key

if "result" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    with st.spinner("Running analysis pipeline…"):
        from src.graph.graph import build_graph
        from src.utils.helpers import default_state
        graph = build_graph()
        st.session_state["result"] = graph.invoke(default_state(tmp_path))
    os.unlink(tmp_path)

result = st.session_state["result"]

if result.get("errors"):
    with st.expander("⚠️ Pipeline errors", expanded=False):
        for e in result["errors"]:
            st.error(e)

# ── Pull data ──────────────────────────────────────────────────────────────────
cleaned_df: pd.DataFrame = result.get("cleaned_df")
raw_df:     pd.DataFrame = result.get("raw_df")
eda         = result.get("eda_report", {})
outlier_rep = result.get("outlier_report", {})
outlier_cols = outlier_rep.get("columns", {})
dtypes_rep  = result.get("dtypes_report", {})
column_map  = result.get("column_map", {})
null_pct    = eda.get("null_pct", {})
shape       = eda.get("shape", {})
correlations = eda.get("correlations", {})

num_cols = cleaned_df.select_dtypes(include="number").columns.tolist() if cleaned_df is not None else []
cat_cols = cleaned_df.select_dtypes(include="object").columns.tolist() if cleaned_df is not None else []
dt_cols  = cleaned_df.select_dtypes(include="datetime").columns.tolist() if cleaned_df is not None else []

all_outlier_idx: set = set()
for col_data in outlier_cols.values():
    all_outlier_idx.update(col_data.get("iqr_outlier_indices", []))

# ── Top metrics ────────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Rows", shape.get("rows", "—"))
m2.metric("Columns", shape.get("columns", "—"))
m3.metric("Numeric Cols", len(num_cols))
m4.metric("Categorical Cols", len(cat_cols))
m5.metric("Cols with Nulls", sum(1 for v in null_pct.values() if v > 0))
m6.metric("Outlier Flags", sum(v.get("iqr_outlier_count", 0) for v in outlier_cols.values()))

# ── Download button ────────────────────────────────────────────────────────────
from src.utils.excel_exporter import generate_excel_report

col_dl, _ = st.columns([1, 4])
with col_dl:
    xlsx_bytes = generate_excel_report(result)
    st.download_button(
        label="⬇️  Download Excel Report",
        data=xlsx_bytes,
        file_name=f"{Path(uploaded.name).stem}_eda_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_summary, tab_data, tab_eda, tab_outliers = st.tabs(["Summary", "Data", "EDA", "Outliers"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab_summary:
    narrative = result.get("final_report", {}).get("narrative", "")
    if narrative:
        st.subheader("AI Analysis")
        st.markdown(narrative)

    if cleaned_df is not None:
        st.divider()
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Data Quality Overview")
            null_df = pd.DataFrame(null_pct.items(), columns=["Column", "Null %"]).sort_values("Null %")
            fig = px.bar(null_df, x="Null %", y="Column", orientation="h",
                         color="Null %", color_continuous_scale="RdYlGn_r",
                         range_color=[0, 100], title="Null % per Column")
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Column Types")
            dtype_counts = cleaned_df.dtypes.astype(str).map(
                lambda x: "Numeric" if any(t in x for t in ["int", "float"])
                else "DateTime" if "datetime" in x else "Text"
            ).value_counts().reset_index()
            dtype_counts.columns = ["Type", "Count"]
            fig2 = px.pie(dtype_counts, names="Type", values="Count", hole=0.5,
                          color_discrete_map={"Numeric": "#4C78A8", "DateTime": "#F58518", "Text": "#54A24B"})
            fig2.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Raw Data")
        if raw_df is not None:
            st.dataframe(raw_df.head(100), use_container_width=True)
    with col_b:
        st.subheader("Cleaned Data")
        if cleaned_df is not None:
            st.dataframe(cleaned_df.head(100), use_container_width=True)

    st.divider()
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Header Changes")
        map_df = pd.DataFrame(column_map.items(), columns=["Original", "Cleaned"])
        changed = map_df[map_df["Original"] != map_df["Cleaned"]]
        st.dataframe(changed if not changed.empty else map_df, use_container_width=True, hide_index=True)
        if changed.empty:
            st.success("All headers were already clean.")

    with col_d:
        st.subheader("Type Inference")
        if dtypes_rep:
            dtype_df = pd.DataFrame([
                {"Column": c, "Original": v["original"], "Inferred": v["inferred"], "Action": v["action"]}
                for c, v in dtypes_rep.items()
            ])
            cast_df = dtype_df[dtype_df["Action"] != "unchanged"]
            st.dataframe(cast_df if not cast_df.empty else dtype_df, use_container_width=True, hide_index=True)
            if cast_df.empty:
                st.success("No type conversions needed.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA  (13 charts)
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:

    # ── 1 & 2: Distribution Explorer — histogram + box ─────────────────────────
    if num_cols:
        st.subheader("1. Distribution Explorer")
        selected_num = st.selectbox("Select numeric column", num_cols, key="dist_col")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(cleaned_df, x=selected_num, nbins=40, marginal="rug",
                               title=f"Histogram — {selected_num}",
                               color_discrete_sequence=["#4C78A8"])
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(cleaned_df, y=selected_num, points="outliers",
                         title=f"Box Plot — {selected_num}",
                         color_discrete_sequence=["#F58518"])
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # ── 3: Violin plot ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("2. Violin Plot")
        fig = px.violin(cleaned_df, y=selected_num, box=True, points="outliers",
                        title=f"Violin — {selected_num}",
                        color_discrete_sequence=["#54A24B"])
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)

        # ── 4: Skewness & Kurtosis ─────────────────────────────────────────────
        st.divider()
        st.subheader("3. Skewness & Kurtosis")
        col3, col4 = st.columns(2)
        with col3:
            skew_df = cleaned_df[num_cols].skew().reset_index()
            skew_df.columns = ["Column", "Skewness"]
            skew_df = skew_df.sort_values("Skewness")
            fig = px.bar(skew_df, x="Skewness", y="Column", orientation="h",
                         color="Skewness", color_continuous_scale="RdBu_r",
                         title="Skewness per Numeric Column")
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            kurt_df = cleaned_df[num_cols].kurt().reset_index()
            kurt_df.columns = ["Column", "Kurtosis"]
            kurt_df = kurt_df.sort_values("Kurtosis")
            fig = px.bar(kurt_df, x="Kurtosis", y="Column", orientation="h",
                         color="Kurtosis", color_continuous_scale="Viridis",
                         title="Kurtosis per Numeric Column")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── 5: Scatter plot with trend line ────────────────────────────────────
        if len(num_cols) >= 2:
            st.divider()
            st.subheader("4. Scatter Plot with Trend Line")
            col5, col6 = st.columns(2)
            with col5:
                x_col = st.selectbox("X axis", num_cols, key="scatter_x")
            with col6:
                y_default = num_cols[1] if num_cols[1] != x_col else num_cols[0]
                y_col = st.selectbox("Y axis", num_cols, index=num_cols.index(y_default), key="scatter_y")
            color_col = cat_cols[0] if cat_cols else None
            fig = px.scatter(cleaned_df, x=x_col, y=y_col, color=color_col,
                             trendline="ols",
                             title=f"{x_col} vs {y_col}",
                             opacity=0.7)
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=420)
            st.plotly_chart(fig, use_container_width=True)

        # ── 6: Scatter Matrix / Pairplot ───────────────────────────────────────
        if len(num_cols) >= 2:
            st.divider()
            st.subheader("5. Scatter Matrix (Pairplot)")
            pair_cols = num_cols[:6]
            sample_df = cleaned_df[pair_cols].dropna()
            if len(sample_df) > 800:
                sample_df = sample_df.sample(800, random_state=42)
            color_series = cleaned_df[cat_cols[0]] if cat_cols else None
            if color_series is not None:
                sample_df = sample_df.copy()
                sample_df[cat_cols[0]] = cleaned_df.loc[sample_df.index, cat_cols[0]]
            fig = px.scatter_matrix(sample_df, dimensions=pair_cols,
                                    color=cat_cols[0] if cat_cols else None,
                                    title="Scatter Matrix (first 6 numeric cols)",
                                    opacity=0.5)
            fig.update_traces(diagonal_visible=False, showupperhalf=False)
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=600)
            st.plotly_chart(fig, use_container_width=True)

        # ── 7: Top & Bottom 5 values ───────────────────────────────────────────
        st.divider()
        st.subheader("6. Top & Bottom 5 Extremes")
        ext_col = st.selectbox("Select column for extremes", num_cols, key="ext_col")
        col7, col8 = st.columns(2)
        with col7:
            top5 = cleaned_df.nlargest(5, ext_col)[[ext_col] + cat_cols[:2]]
            st.markdown(f"**Top 5 — {ext_col}**")
            st.dataframe(top5, use_container_width=True, hide_index=True)
        with col8:
            bot5 = cleaned_df.nsmallest(5, ext_col)[[ext_col] + cat_cols[:2]]
            st.markdown(f"**Bottom 5 — {ext_col}**")
            st.dataframe(bot5, use_container_width=True, hide_index=True)

    # ── 8: Correlation Heatmap ─────────────────────────────────────────────────
    if correlations and len(num_cols) >= 2:
        st.divider()
        st.subheader("7. Correlation Heatmap")
        corr_df = pd.DataFrame(correlations)
        fig = px.imshow(corr_df, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        text_auto=".2f", aspect="auto", title="Pearson Correlation Matrix")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=480)
        st.plotly_chart(fig, use_container_width=True)

    # ── 9: Missing value heatmap ───────────────────────────────────────────────
    if cleaned_df is not None and cleaned_df.isnull().any().any():
        st.divider()
        st.subheader("8. Missing Value Heatmap")
        sample = cleaned_df if len(cleaned_df) <= 200 else cleaned_df.sample(200, random_state=42)
        null_matrix = sample.isnull().astype(int)
        fig = px.imshow(null_matrix.T, color_continuous_scale=["#EBF5FB", "#E74C3C"],
                        title="Missing Values (red = null) — sample of 200 rows",
                        labels=dict(x="Row Index", y="Column", color="Is Null"),
                        aspect="auto")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── 10: Row completeness histogram ────────────────────────────────────────
    if cleaned_df is not None:
        st.divider()
        st.subheader("9. Row Completeness")
        completeness = (cleaned_df.notna().sum(axis=1) / len(cleaned_df.columns) * 100).round(1)
        fig = px.histogram(completeness, nbins=20,
                           title="Distribution of Row Completeness %",
                           labels={"value": "Completeness %", "count": "Rows"},
                           color_discrete_sequence=["#54A24B"])
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── 11: Column cardinality ─────────────────────────────────────────────────
    if cleaned_df is not None:
        st.divider()
        st.subheader("10. Column Cardinality (Unique Values)")
        card_df = cleaned_df.nunique().reset_index()
        card_df.columns = ["Column", "Unique Values"]
        card_df = card_df.sort_values("Unique Values", ascending=True)
        fig = px.bar(card_df, x="Unique Values", y="Column", orientation="h",
                     color="Unique Values", color_continuous_scale="Blues",
                     title="Unique Value Count per Column")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0),
                          height=max(300, len(card_df) * 28), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── 12: Grouped box plot (cat vs num) ──────────────────────────────────────
    if cat_cols and num_cols:
        st.divider()
        st.subheader("11. Categorical vs Numeric (Grouped Box)")
        col9, col10 = st.columns(2)
        with col9:
            grp_cat = st.selectbox("Category column", cat_cols, key="grp_cat")
        with col10:
            grp_num = st.selectbox("Numeric column", num_cols, key="grp_num")
        top_cats = cleaned_df[grp_cat].value_counts().head(10).index
        plot_df = cleaned_df[cleaned_df[grp_cat].isin(top_cats)]
        fig = px.box(plot_df, x=grp_cat, y=grp_num, color=grp_cat,
                     title=f"{grp_num} by {grp_cat}",
                     points="outliers")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── 13: Time series ────────────────────────────────────────────────────────
    if dt_cols and num_cols:
        st.divider()
        st.subheader("12. Time Series")
        col11, col12 = st.columns(2)
        with col11:
            ts_date = st.selectbox("Date column", dt_cols, key="ts_date")
        with col12:
            ts_val = st.selectbox("Value column", num_cols, key="ts_val")
        ts_df = cleaned_df[[ts_date, ts_val]].dropna().sort_values(ts_date)
        fig = px.line(ts_df, x=ts_date, y=ts_val, title=f"{ts_val} over time",
                      color_discrete_sequence=["#4C78A8"])
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── 14: Categorical distributions (bar charts) ─────────────────────────────
    if cat_cols:
        st.divider()
        st.subheader("13. Categorical Distributions")
        top_vals = eda.get("top_value_counts", {})
        chunks = [cat_cols[i:i+3] for i in range(0, len(cat_cols), 3)]
        for chunk in chunks:
            cols = st.columns(len(chunk))
            for i, col_name in enumerate(chunk):
                with cols[i]:
                    counts = top_vals.get(col_name, {})
                    if counts:
                        vc_df = pd.DataFrame(counts.items(), columns=["Value", "Count"])
                        fig = px.bar(vc_df, x="Count", y="Value", orientation="h",
                                     title=col_name, color="Count",
                                     color_continuous_scale="Blues")
                        fig.update_layout(height=280, margin=dict(l=0, r=0, t=40, b=0),
                                          showlegend=False, coloraxis_showscale=False)
                        st.plotly_chart(fig, use_container_width=True)

    # ── 14: CDF ───────────────────────────────────────────────────────────────
    if num_cols:
        st.divider()
        st.subheader("14. Cumulative Distribution Function (CDF)")
        cdf_col = st.selectbox("Select column for CDF", num_cols, key="cdf_col")
        cdf_data = cleaned_df[cdf_col].dropna().sort_values()
        cdf_y = np.arange(1, len(cdf_data) + 1) / len(cdf_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cdf_data, y=cdf_y, mode="lines",
                                 line=dict(color="#4C78A8", width=2),
                                 fill="tozeroy", fillcolor="rgba(76,120,168,0.15)"))
        fig.update_layout(title=f"CDF — {cdf_col}", xaxis_title=cdf_col,
                          yaxis_title="Cumulative Probability", yaxis_range=[0, 1],
                          margin=dict(l=0, r=0, t=40, b=0), height=380)
        st.plotly_chart(fig, use_container_width=True)

    # ── 15: Q-Q Plot (normality check) ────────────────────────────────────────
    if num_cols:
        st.divider()
        st.subheader("15. Q-Q Plot (Normality Check)")
        qq_col = st.selectbox("Select column for Q-Q plot", num_cols, key="qq_col")
        from scipy import stats as scipy_stats
        qq_data = cleaned_df[qq_col].dropna()
        (theoretical, sample), _ = scipy_stats.probplot(qq_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theoretical, y=sample, mode="markers",
                                 marker=dict(color="#4C78A8", size=5, opacity=0.7),
                                 name="Data"))
        mn, mx = min(theoretical), max(theoretical)
        slope, intercept, *_ = scipy_stats.linregress(theoretical, sample)
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn * slope + intercept, mx * slope + intercept],
                                 mode="lines", line=dict(color="#E45756", dash="dash"),
                                 name="Normal reference"))
        fig.update_layout(title=f"Q-Q Plot — {qq_col} (closer to line = more normal)",
                          xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles",
                          margin=dict(l=0, r=0, t=40, b=0), height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── 16: 2D Density Heatmap ────────────────────────────────────────────────
    if len(num_cols) >= 2:
        st.divider()
        st.subheader("16. 2D Density Heatmap")
        col_da, col_db = st.columns(2)
        with col_da:
            dens_x = st.selectbox("X axis", num_cols, key="dens_x")
        with col_db:
            remaining = [c for c in num_cols if c != dens_x]
            dens_y = st.selectbox("Y axis", remaining, key="dens_y")
        fig = px.density_heatmap(cleaned_df, x=dens_x, y=dens_y,
                                 color_continuous_scale="Viridis",
                                 marginal_x="histogram", marginal_y="histogram",
                                 title=f"2D Density — {dens_x} vs {dens_y}")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=480)
        st.plotly_chart(fig, use_container_width=True)

    # ── 17: Bubble Chart ──────────────────────────────────────────────────────
    if len(num_cols) >= 3:
        st.divider()
        st.subheader("17. Bubble Chart (3 Variables)")
        col_ba, col_bb, col_bc = st.columns(3)
        with col_ba:
            bub_x = st.selectbox("X axis", num_cols, key="bub_x")
        with col_bb:
            bub_y = st.selectbox("Y axis", [c for c in num_cols if c != bub_x], key="bub_y")
        with col_bc:
            bub_sz = st.selectbox("Bubble size", [c for c in num_cols if c not in [bub_x, bub_y]], key="bub_sz")
        bub_color = cat_cols[0] if cat_cols else None
        fig = px.scatter(cleaned_df, x=bub_x, y=bub_y, size=bub_sz,
                         color=bub_color, size_max=40, opacity=0.7,
                         title=f"{bub_x} vs {bub_y} (size = {bub_sz})")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=460)
        st.plotly_chart(fig, use_container_width=True)

    # ── 18: Cross-tab Heatmap (cat vs cat) ────────────────────────────────────
    if len(cat_cols) >= 2:
        st.divider()
        st.subheader("18. Cross-tab Heatmap (Category × Category)")
        col_ca, col_cb = st.columns(2)
        with col_ca:
            ct_row = st.selectbox("Row category", cat_cols, key="ct_row")
        with col_cb:
            ct_col = st.selectbox("Column category", [c for c in cat_cols if c != ct_row], key="ct_col")
        top_rows = cleaned_df[ct_row].value_counts().head(10).index
        top_cols = cleaned_df[ct_col].value_counts().head(10).index
        ct_df = cleaned_df[cleaned_df[ct_row].isin(top_rows) & cleaned_df[ct_col].isin(top_cols)]
        ct = pd.crosstab(ct_df[ct_row], ct_df[ct_col])
        fig = px.imshow(ct, text_auto=True, color_continuous_scale="Blues",
                        title=f"Co-occurrence: {ct_row} × {ct_col}")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── 19: Parallel Coordinates ──────────────────────────────────────────────
    if len(num_cols) >= 3:
        st.divider()
        st.subheader("19. Parallel Coordinates Plot")
        par_cols = num_cols[:8]
        par_df = cleaned_df[par_cols].dropna()
        if len(par_df) > 500:
            par_df = par_df.sample(500, random_state=42)
        color_col_par = num_cols[0]
        fig = px.parallel_coordinates(par_df, dimensions=par_cols, color=color_col_par,
                                      color_continuous_scale="Viridis",
                                      title="Parallel Coordinates (sample of 500 rows)")
        fig.update_layout(margin=dict(l=80, r=80, t=60, b=20), height=460)
        st.plotly_chart(fig, use_container_width=True)

    # ── 20: Mean ± Std Error Bar Chart ────────────────────────────────────────
    if cat_cols and num_cols:
        st.divider()
        st.subheader("20. Mean ± Std by Category")
        col_ea, col_eb = st.columns(2)
        with col_ea:
            err_cat = st.selectbox("Category", cat_cols, key="err_cat")
        with col_eb:
            err_num = st.selectbox("Numeric", num_cols, key="err_num")
        top_cats_err = cleaned_df[err_cat].value_counts().head(12).index
        grp = (cleaned_df[cleaned_df[err_cat].isin(top_cats_err)]
               .groupby(err_cat)[err_num].agg(["mean", "std"]).reset_index())
        grp.columns = [err_cat, "Mean", "Std"]
        grp = grp.sort_values("Mean", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=grp[err_cat], y=grp["Mean"],
                             error_y=dict(type="data", array=grp["Std"].fillna(0)),
                             marker_color="#4C78A8", name="Mean ± Std"))
        fig.update_layout(title=f"Mean ± Std of {err_num} by {err_cat}",
                          xaxis_title=err_cat, yaxis_title=err_num,
                          margin=dict(l=0, r=0, t=40, b=0), height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── 21: Strip Plot ────────────────────────────────────────────────────────
    if cat_cols and num_cols:
        st.divider()
        st.subheader("21. Strip Plot (All Data Points by Category)")
        col_sa, col_sb = st.columns(2)
        with col_sa:
            strip_cat = st.selectbox("Category", cat_cols, key="strip_cat")
        with col_sb:
            strip_num = st.selectbox("Numeric", num_cols, key="strip_num")
        top_cats_strip = cleaned_df[strip_cat].value_counts().head(10).index
        strip_df = cleaned_df[cleaned_df[strip_cat].isin(top_cats_strip)]
        fig = px.strip(strip_df, x=strip_cat, y=strip_num, color=strip_cat,
                       title=f"All {strip_num} values by {strip_cat}",
                       stripmode="overlay")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=420, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Summary stats table ────────────────────────────────────────────────────
    summary = eda.get("summary_stats", {})
    if summary:
        st.divider()
        st.subheader("Summary Statistics Table")
        st.dataframe(pd.DataFrame(summary).T, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — OUTLIERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_outliers:
    if not outlier_cols:
        st.info("No numeric columns found for outlier detection.")
    else:
        st.caption(f"Method: {outlier_rep.get('method', 'IQR + Z-score')}")

        summary_rows = [
            {"Column": col, "IQR Outliers": d["iqr_outlier_count"],
             "Z-Score Outliers": d["z_score_outlier_count"],
             "Lower Bound": d["bounds"]["lower"], "Upper Bound": d["bounds"]["upper"]}
            for col, d in outlier_cols.items()
        ]
        summary_df = pd.DataFrame(summary_rows).sort_values("IQR Outliers", ascending=False)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        flagged = summary_df[summary_df["IQR Outliers"] > 0]
        if not flagged.empty:
            st.divider()
            fig = px.bar(flagged.sort_values("IQR Outliers"),
                         x="IQR Outliers", y="Column", orientation="h",
                         title="Outlier Count by Column (IQR)",
                         color="IQR Outliers", color_continuous_scale="Reds")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Inspect by Column")
        flagged_names = [r["Column"] for r in summary_rows if r["IQR Outliers"] > 0]
        all_num_names = [r["Column"] for r in summary_rows]
        selected = st.selectbox("Select column", flagged_names if flagged_names else all_num_names, key="out_col")

        if selected and cleaned_df is not None:
            data = outlier_cols[selected]
            iqr_idx = set(data["iqr_outlier_indices"])
            plot_df = cleaned_df[[selected]].copy().dropna()
            plot_df["Status"] = plot_df.index.map(lambda i: "Outlier" if i in iqr_idx else "Normal")

            col_a, col_b = st.columns(2)
            with col_a:
                fig = px.box(plot_df, y=selected, color="Status", points="all",
                             title=f"Box Plot — {selected}",
                             color_discrete_map={"Outlier": "#E45756", "Normal": "#4C78A8"})
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)
            with col_b:
                plot_df["Index"] = plot_df.index
                fig = px.scatter(plot_df, x="Index", y=selected, color="Status",
                                 title=f"Scatter — {selected}",
                                 color_discrete_map={"Outlier": "#E45756", "Normal": "#4C78A8"})
                fig.add_hline(y=data["bounds"]["upper"], line_dash="dash", line_color="red",
                              annotation_text="Upper bound")
                fig.add_hline(y=data["bounds"]["lower"], line_dash="dash", line_color="orange",
                              annotation_text="Lower bound")
                fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

            if iqr_idx:
                st.markdown(f"**{len(iqr_idx)} flagged rows:**")
                st.dataframe(cleaned_df.loc[sorted(iqr_idx)], use_container_width=True)
