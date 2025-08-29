# Adera Data Dashboard — resilient version (fixes KeyError)
# --------------------------------------------------------------
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from io import BytesIO
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# Page config
# =========================
st.set_page_config(page_title='Adera Data Dashboard', layout='wide')
st.title('Adera Data Dashboard')

# =========================
# Utilities
# =========================
@st.cache_data(show_spinner=False)
def load_excel(path: str, sheet_name: str = "ProDB", header_row: int = 2):
    # header=2 → Excel row 3 is column names
    return pd.read_excel(path, sheet_name=sheet_name, header=header_row, engine="openpyxl")

def clean_and_normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Trim, collapse spaces, normalize spaces around parentheses.
    2) Map common variants to canonical names per your schema.
    """
    def _clean(s: str) -> str:
        s = str(s).replace("\n", " ")
        s = re.sub(r"\s+", " ", s).strip()
        # normalize spaces around parentheses
        s = re.sub(r"\s*\(\s*", " (", s)
        s = re.sub(r"\s*\)\s*", ")", s)
        return s

    df.columns = [_clean(c) for c in df.columns]

    # variant → canonical (keys lowercase for easy matching)
    variant_map = {
        # potentials
        "pot. gross(bfpd)": "Pot. Gross (bfpd)",
        "pot. gross (bfpd)": "Pot. Gross (bfpd)",
        "pot. nett(bopd)": "Pot. Nett (bopd)",
        "pot. nett (bopd)": "Pot. Nett (bopd)",
        "pot. wc(%)": "Pot. WC (%)",
        "pot. wc (%)": "Pot. WC (%)",
        "pot. gas(mmscfd)": "Pot. Gas (MMscfd)",
        "pot. gas (mmscfd)": "Pot. Gas (MMscfd)",
        # actuals
        "act. gross(bfpd)": "Act. Gross (bfpd)",
        "act. gross (bfpd)": "Act. Gross (bfpd)",
        "act. nett(bopd)": "Act. Nett (bopd)",
        "act. nett (bopd)": "Act. Nett (bopd)",
        "act. wc(%)": "Act. WC (%)",
        "act. wc (%)": "Act. WC (%)",
        "act. gas prod(mmscfd)": "Act. Gas Prod (MMscfd)",
        "act. gas prod (mmscfd)": "Act. Gas Prod (MMscfd)",
        # misc typos
        "pump eff(%)": "Pump Eff (%)",
        "pump eff (%)": "Pump Eff (%)",
        "pump eff": "Pump Eff (%)",
        "liftingmethod": "Lifting Method",
        "lifting method ": "Lifting Method",
    }

    canonical = []
    for c in df.columns:
        lc = c.lower()
        canonical.append(variant_map.get(lc, c))
    df.columns = canonical
    return df

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def safe_download_buttons_for_fig(fig: go.Figure, base_filename: str, key_prefix: str):
    st.plotly_chart(fig, use_container_width=True)
    # Try PNG via kaleido
    png_buffer = None
    try:
        png_bytes = pio.to_image(fig, format="png", scale=2)
        png_buffer = BytesIO(png_bytes)
    except Exception:
        png_buffer = None
    if png_buffer is not None:
        st.download_button(
            label="Download as PNG",
            data=png_buffer,
            file_name=f"{base_filename}.png",
            mime="image/png",
            key=f"{key_prefix}_png"
        )
    html_bytes = fig.to_html(full_html=False).encode("utf-8")
    st.download_button(
        label="Download as HTML",
        data=html_bytes,
        file_name=f"{base_filename}.html",
        mime="text/html",
        key=f"{key_prefix}_html"
    )

def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Month"] = df["Date"].dt.month
        df["Year"]  = df["Date"].dt.year
    else:
        df["Month"] = np.nan
        df["Year"] = np.nan
    return df

def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return only the columns that exist in df (avoids KeyError on selection)."""
    return [c for c in cols if c in df.columns]

def rmse_score(y_true, y_pred):
    # Backward/forward compatible RMSE
    try:
        # Newer sklearn
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        # Older sklearn fallback
        import numpy as np
        return np.sqrt(mean_squared_error(y_true, y_pred))

# =========================
# Data load & prep
# =========================
with st.sidebar:
    st.header("Data Settings")
    excel_path = st.text_input("Excel file path", value="Processed Database.xlsx")
    sheet_name = st.text_input("Sheet name", value="ProDB")
    header_row = st.number_input("Header row index (0-based)", value=2, min_value=0, step=1)
    st.caption("Your headers are on Excel row 3 → 0-based index = 2.")

df_raw = load_excel(excel_path, sheet_name=sheet_name, header_row=header_row)
df = df_raw.copy()
df = clean_and_normalize_headers(df)

# Expected columns from your schema (incl. Lifting Method at AE)
expected_cols = [
    "No","Date","Structure","Well","Down Time","Pot. Gross (bfpd)","Pot. Nett (bopd)","Pot. WC (%)",
    "Pot. Gas (MMscfd)","Last Test Date","Act. Gross (bfpd)","Act. Nett (bopd)","Act. WC (%)",
    "Act. Gas Prod (MMscfd)","Loss/Gain","Zone","Choke","Pump Eff (%)","Production System",
    "SL/Freq","SPM/AMP","PLGR/CAP","Pcsg","Ptbg","Pfl","Psep","SFL","DFL","PSD","Remark",
    "Lifting Method"
]
present_cols = [c for c in expected_cols if c in df.columns]
missing_cols = [c for c in expected_cols if c not in df.columns]

with st.expander("🔎 Detected columns"):
    st.write(list(df.columns))
if missing_cols:
    pass

# Convert types (only if present)
df = add_time_parts(df)
numeric_candidates = [
    "Down Time","Pot. Gross (bfpd)","Pot. Nett (bopd)","Pot. WC (%)","Pot. Gas (MMscfd)",
    "Act. Gross (bfpd)","Act. Nett (bopd)","Act. WC (%)","Act. Gas Prod (MMscfd)",
    "Loss/Gain","Choke","Pump Eff (%)","Pcsg","Ptbg","Pfl","Psep","SFL","DFL","PSD"
]
df = coerce_numeric(df, numeric_candidates)

# Tabs
tabs = st.tabs([
    "Statistics",
    "Rate Change Alerts",
    "Pareto Chart",
    "Time Series Visualization",
    "Anomaly Detection",
    "Predictive Modeling"
])

# Helper: guard for required cols
def have(*cols):
    return all(c in df.columns for c in cols)

# ================= Statistics Tab =================
with tabs[0]:
    st.header('Statistics')

    if not have("Well", "Date"):
        st.warning("Columns 'Well' and/or 'Date' are missing; cannot compute statistics.")
    else:
        # Year / Month selection
        st.subheader("Select Month and Year for Analysis")
        avail_years = sorted(df['Year'].dropna().unique(), reverse=True)
        selected_year_stat = st.selectbox("Select Year", avail_years if len(avail_years) else [np.nan], key="stat_year")
        months_for_year = sorted(df[df['Year'] == selected_year_stat]['Month'].dropna().unique())
        selected_month_stat = st.selectbox("Select Month", months_for_year if len(months_for_year) else [np.nan], key="stat_month")

        # Top-N selector
        top_n = st.selectbox("Select number of top producers", [3, 5, 10, 20], index=3, key="stat_top_n")

        # Filter to month/year and take latest test per well
        stat_df = df.copy()
        if not np.isnan(selected_year_stat):
            stat_df = stat_df[stat_df['Year'] == selected_year_stat]
        if len(months_for_year):
            stat_df = stat_df[stat_df['Month'] == selected_month_stat]
        stat_df = stat_df.sort_values("Date").groupby("Well", as_index=False).last()

        # Build agg map only for present columns (avoids KeyError)
        agg_map = {}
        if "Act. Nett (bopd)" in df.columns:
            agg_map["Act. Nett (bopd)"] = "max"
        if "Act. Gas Prod (MMscfd)" in df.columns:
            agg_map["Act. Gas Prod (MMscfd)"] = "max"
        if "Date" in df.columns:
            agg_map["Date"] = "min"

        if not agg_map:
            st.info("Not enough columns to compute peaks.")
            peak_df = pd.DataFrame()
        else:
            peak_df = df.groupby("Well", as_index=False).agg(agg_map)
            rename_cols = {}
            if "Date" in agg_map:
                rename_cols["Date"] = "First Test Date"
            if "Act. Nett (bopd)" in agg_map:
                rename_cols["Act. Nett (bopd)"] = "Peak Oil (bopd)"
            if "Act. Gas Prod (MMscfd)" in agg_map:
                rename_cols["Act. Gas Prod (MMscfd)"] = "Peak Gas (MMscfd)"
            if rename_cols:
                peak_df = peak_df.rename(columns=rename_cols)

        if len(peak_df):
            stat_df = pd.merge(stat_df, peak_df, on="Well", how="left")
            if "First Test Date" in stat_df.columns:
                stat_df['Lifespan Days'] = (stat_df['Date'] - stat_df['First Test Date']).dt.days
            else:
                stat_df['Lifespan Days'] = np.nan

            def format_lifespan(days):
                days = 0 if pd.isna(days) else int(days)
                years = days // 365
                months = (days % 365) // 30
                return years, months

            if len(stat_df):
                stat_df['Years'], stat_df['Months'] = zip(*stat_df['Lifespan Days'].apply(format_lifespan))
                stat_df['Well Lifespan'] = stat_df.apply(
                    lambda r: f"{r['Years']} year{'s' if r['Years']!=1 else ''}, {r['Months']} month{'s' if r['Months']!=1 else ''}" +
                              (" (New)" if r['Years'] < 2 else " (Mature)"),
                    axis=1
                )

            # --- Top Oil ---
            st.subheader("Top Oil Producers")
            if "Act. Nett (bopd)" in stat_df.columns:
                top_oil = stat_df.sort_values("Act. Nett (bopd)", ascending=False).head(top_n)
            
                peak_oil_col = "Peak Oil (bopd)" if "Peak Oil (bopd)" in top_oil.columns else "Act. Nett (bopd)"
                display_cols = safe_cols(
                    top_oil,
                    ["Well", "Structure", "Act. Nett (bopd)", peak_oil_col, "Well Lifespan"]
                )
                st.dataframe(top_oil[display_cols])
            
                oil_fig = px.bar(top_oil, x="Well", y="Act. Nett (bopd)", color="Structure" if "Structure" in top_oil.columns else None,
                                 title="Top Oil Producers")
                safe_download_buttons_for_fig(oil_fig, "top_oil_producers", "stat_oil")
            else:
                st.info("Column 'Act. Nett (bopd)' not found.")

            # --- Top Gas ---
            st.subheader("Top Gas Producers")
            if "Act. Gas Prod (MMscfd)" in stat_df.columns:
                top_gas = stat_df.sort_values("Act. Gas Prod (MMscfd)", ascending=False).head(top_n)
            
                peak_gas_col = "Peak Gas (MMscfd)" if "Peak Gas (MMscfd)" in top_gas.columns else "Act. Gas Prod (MMscfd)"
                display_cols = safe_cols(
                    top_gas,
                    ["Well", "Structure", "Act. Gas Prod (MMscfd)", peak_gas_col, "Well Lifespan"]
                )
                st.dataframe(top_gas[display_cols])
            
                gas_fig = px.bar(top_gas, x="Well", y="Act. Gas Prod (MMscfd)", color="Structure" if "Structure" in top_gas.columns else None,
                                 title="Top Gas Producers")
                safe_download_buttons_for_fig(gas_fig, "top_gas_producers", "stat_gas")
            else:
                st.info("Column 'Act. Gas Prod (MMscfd)' not found.")

# ================= Rate Change Alerts Tab (UPDATED — No min points; compare latest-in-range vs previous overall, optional Recover filter) =================
with tabs[1]:
    st.header("Rate Change Alerts — Oil & Gas")
    st.caption(
        "For each well that has at least one test inside the selected window, we take the latest test in-window "
        "and compare it to the immediately previous test (which may be before the window). Split into Oil and Gas tables."
    )

    if not have("Well", "Date"):
        st.info("Need 'Well' and 'Date' columns.")
    else:
        # --- Date range filter (inclusive) ---
        dmin = pd.to_datetime(df["Date"], errors="coerce").min()
        dmax = pd.to_datetime(df["Date"], errors="coerce").max()
        if pd.isna(dmin) or pd.isna(dmax):
            st.info("No valid dates found in 'Date' column.")
        else:
            start_date, end_date = st.date_input(
                "Select start & end date (inclusive)",
                value=(dmin.date(), dmax.date())
            )
            if start_date > end_date:
                st.warning("Start date must be <= end date.")
            else:
                wells = sorted(df['Well'].dropna().unique())

                # --- Recover filter option ---
                include_recover = st.checkbox(
                    "Include wells with Lifting Method = 'Recover'",
                    value=False,
                    help=(
                        "When unticked (default), any rows where Lifting Method == 'Recover' are ignored "
                        "for both the in-window latest point and the previous comparison point."
                    ),
                )

                def filter_recover(data: pd.DataFrame) -> pd.DataFrame:
                    if include_recover:
                        return data
                    if "Lifting Method" not in data.columns:
                        return data
                    lm = data["Lifting Method"].astype(str).str.strip().str.lower()
                    return data[~lm.eq("recover")]

                # --- Threshold options ---
                st.subheader("Threshold Settings")
                threshold_mode = st.radio(
                    "Choose threshold type",
                    ["Percent (%)", "Absolute"],
                    index=0,
                    horizontal=True,
                )

                if threshold_mode == "Percent (%)":
                    pct_threshold = st.slider("Percent change threshold (%)", 1, 100, 20)
                    oil_abs_threshold = None
                    gas_abs_threshold = None
                else:
                    st.caption("Absolute thresholds use original units: bopd for oil, MMscfd for gas.")
                    oil_abs_threshold = st.number_input(
                        "Oil absolute change threshold (bopd)", value=50.0, min_value=0.0, step=10.0
                    )
                    gas_abs_threshold = st.number_input(
                        "Gas absolute change threshold (MMscfd)", value=0.2, min_value=0.0, step=0.05
                    )
                    pct_threshold = None

                # Helper funcs
                def pct_change(a, b):
                    if pd.isna(a) or pd.isna(b) or b == 0:
                        return np.nan
                    return 100.0 * (a - b) / abs(b)

                def abs_change(a, b):
                    if pd.isna(a) or pd.isna(b):
                        return np.nan
                    return a - b

                oil_alerts = []
                gas_alerts = []

                # Iterate wells
                for w in wells:
                    d_all_raw = df[df['Well'] == w].sort_values('Date')
                    d_all = filter_recover(d_all_raw)

                    # rows inside the window for this well (already Recover-filtered if applied)
                    d_in = d_all[(d_all['Date'].dt.date >= start_date) & (d_all['Date'].dt.date <= end_date)]
                    if d_in.empty:
                        continue  # no in-window non-Recover data → skip

                    # latest in-window row
                    last = d_in.iloc[-1]

                    # previous test overall (can be before the window); also Recover-filtered if applied
                    prev_candidates = d_all[d_all['Date'] < last['Date']]
                    if prev_candidates.empty:
                        continue  # no previous non-Recover test exists → cannot form a pair
                    prev = prev_candidates.iloc[-1]

                    # Extract values
                    last_oil = last.get("Act. Nett (bopd)")
                    prev_oil = prev.get("Act. Nett (bopd)")
                    last_gas = last.get("Act. Gas Prod (MMscfd)")
                    prev_gas = prev.get("Act. Gas Prod (MMscfd)")
                    last_choke = last.get("Choke")
                    prev_choke = prev.get("Choke")

                    # Compute changes
                    oil_pct = pct_change(last_oil, prev_oil)
                    gas_pct = pct_change(last_gas, prev_gas)
                    choke_pct = pct_change(last_choke, prev_choke)

                    oil_abs = abs_change(last_oil, prev_oil)
                    gas_abs = abs_change(last_gas, prev_gas)

                    # Flagging logic per mode
                    if threshold_mode == "Percent (%)":
                        oil_flag = (not pd.isna(oil_pct)) and abs(oil_pct) >= pct_threshold
                        gas_flag = (not pd.isna(gas_pct)) and abs(gas_pct) >= pct_threshold
                        oil_delta, gas_delta = oil_pct, gas_pct
                        oil_delta_label, gas_delta_label = "%", "%"
                    else:
                        oil_flag = (not pd.isna(oil_abs)) and abs(oil_abs) >= float(oil_abs_threshold)
                        gas_flag = (not pd.isna(gas_abs)) and abs(gas_abs) >= float(gas_abs_threshold)
                        oil_delta, gas_delta = oil_abs, gas_abs
                        oil_delta_label, gas_delta_label = "abs (bopd)", "abs (MMscfd)"

                    base_row = {
                        "Well": w,
                        "Structure": last.get("Structure", np.nan),
                        "Prev Date": prev.get("Date", np.nan),
                        "Last Date": last.get("Date", np.nan),
                        # Context
                        "Lifting Method": last.get("Lifting Method", np.nan),
                        "Pump Eff (%)": last.get("Pump Eff (%)", np.nan),
                        # Choke info
                        "Prev Choke": prev_choke,
                        "Last Choke": last_choke,
                        "Choke Δ%": choke_pct,
                    }

                    if oil_flag:
                        oil_row = base_row.copy()
                        oil_row.update({
                            "Prev Oil (bopd)": prev_oil,
                            "Last Oil (bopd)": last_oil,
                            "Oil Δ": oil_delta,
                            "Δ Type": oil_delta_label,
                        })
                        oil_alerts.append(oil_row)

                    if gas_flag:
                        gas_row = base_row.copy()
                        gas_row.update({
                            "Prev Gas (MMscfd)": prev_gas,
                            "Last Gas (MMscfd)": last_gas,
                            "Gas Δ": gas_delta,
                            "Δ Type": gas_delta_label,
                        })
                        gas_alerts.append(gas_row)

                # --- Render tables ---
                col_oil, col_gas = st.columns(2)

                with col_oil:
                    st.subheader("Oil Alerts")
                    if oil_alerts:
                        oil_df = pd.DataFrame(oil_alerts)
                        if "Last Date" in oil_df.columns:
                            oil_df = oil_df.sort_values("Last Date", ascending=False)
                        st.dataframe(oil_df)
                    else:
                        st.success("No oil wells crossed the threshold for the selected window.")

                with col_gas:
                    st.subheader("Gas Alerts")
                    if gas_alerts:
                        gas_df = pd.DataFrame(gas_alerts)
                        if "Last Date" in gas_df.columns:
                            gas_df = gas_df.sort_values("Last Date", ascending=False)
                        st.dataframe(gas_df)
                    else:
                        st.success("No gas wells crossed the threshold for the selected window.")

# ================= Pareto Chart Tab =================
with tabs[2]:
    st.header("Pareto Chart")

    # ---- Metric & Category (only show existing) ----
    metric = st.selectbox(
        "Metric (absolute values are used for Pareto accumulation)",
        [m for m in ["Act. Nett (bopd)", "Act. Gas Prod (MMscfd)", "Down Time"] if m in df.columns] or ["Act. Nett (bopd)"]
    )
    category = st.selectbox(
        "Category",
        [c for c in ["Well", "Structure", "Zone", "Lifting Method"] if c in df.columns] or ["Well"]
    )

    if not have("Date", "Well"):
        st.info("Need 'Date' and 'Well' columns.")
    else:
        # ---- Month/Year selectors ----
        avail_years = sorted(df['Year'].dropna().unique(), reverse=True)
        sel_year = st.selectbox("Year", avail_years if len(avail_years) else [np.nan], key="pareto_year")

        months_for_year = sorted(df[df['Year'] == sel_year]['Month'].dropna().unique()) if not pd.isna(sel_year) else []
        sel_month = st.selectbox("Month", months_for_year if len(months_for_year) else [np.nan], key="pareto_month")

        # ---- Top N selector ----
        top_choice = st.selectbox("Show Top", ["All", 5, 10, 20], index=3)

        # ---- Filter to selected month/year ----
        work = df.copy()
        if not np.isnan(sel_year):
            work = work[work["Year"] == sel_year]
        if len(months_for_year):
            work = work[work["Month"] == sel_month]

        if work.empty or metric not in work.columns or category not in work.columns:
            st.info("No data for the selected month/year (or selected columns missing).")
        else:
            # ---- ALWAYS 'Latest per Well' within the month/year ----
            base_cols = ["Well", "Date", "Structure", category, metric]
            keep_cols = [c for c in base_cols if c in work.columns]
            keep_cols = list(dict.fromkeys(keep_cols))  # de-dup names, preserve order

            tmp = work[keep_cols].copy()
            tmp = tmp.loc[:, ~tmp.columns.duplicated()].copy()  # extra safety

            data_pareto = (
                tmp.dropna(subset=["Well", "Date"])
                   .sort_values("Date")
                   .groupby("Well", as_index=False)
                   .last()
            )

            if "Structure" not in data_pareto.columns:
                data_pareto["Structure"] = "Unknown"

            # ---------- SAFE aggregation using a temp key ----------
            # Build mini table and rename category -> 'cat_key' to avoid name collisions in groupby results
            cols_needed = [category, "Structure", metric]
            cols_needed = [c for c in list(dict.fromkeys(cols_needed)) if c in data_pareto.columns]
            p = data_pareto[cols_needed].copy()
            p = p.loc[:, ~p.columns.duplicated()].copy()

            if category not in p.columns or metric not in p.columns:
                st.info("Required columns not available after filtering.")
            else:
                p = p.dropna(subset=[category, metric])
                if p.empty:
                    st.info("No rows to aggregate after filtering.")
                else:
                    p["abs_metric"] = p[metric].abs()
                    p = p.rename(columns={category: "cat_key"})  # <--- key step

                    # Totals per category (for ordering & cumulative %)
                    totals = (
                        p[["cat_key", "abs_metric"]]
                        .groupby("cat_key", as_index=False)
                        .sum()
                        .rename(columns={"abs_metric": "cat_total"})
                        .sort_values("cat_total", ascending=False)
                    )

                    grand_total = totals["cat_total"].sum()
                    if grand_total == 0:
                        st.info("Total is zero after filtering; nothing to plot.")
                    else:
                        # Top N categories by total
                        if top_choice != "All":
                            top_n = int(top_choice)
                            top_cats = totals["cat_key"].head(top_n).tolist()
                            totals = totals[totals["cat_key"].isin(top_cats)]
                        else:
                            top_cats = totals["cat_key"].tolist()

                        # Order categories
                        cat_order = totals["cat_key"].tolist()

                        # Per-(category, structure) contributions for stacked bars
                        if "Structure" not in p.columns:
                            p["Structure"] = "Unknown"

                        by_cat_struct = (
                            p[p["cat_key"].isin(cat_order)]
                            .groupby(["cat_key", "Structure"], as_index=False)["abs_metric"]
                            .sum()
                        )

                        # Cumulative % over ordered categories
                        totals_ordered = totals.set_index("cat_key").loc[cat_order].reset_index()
                        totals_ordered["cumperc"] = 100 * totals_ordered["cat_total"].cumsum() / grand_total

                        # ---- Plot: stacked bars by Structure, cumulative % line by category ----
                        fig = go.Figure()

                        structures = sorted(by_cat_struct["Structure"].dropna().unique().tolist())
                        for s in structures:
                            sub = by_cat_struct[by_cat_struct["Structure"] == s]
                            y_vals = [float(sub.loc[sub["cat_key"] == c, "abs_metric"].sum()) for c in cat_order]
                            fig.add_bar(x=cat_order, y=y_vals, name=str(s))

                        fig.add_scatter(
                            x=cat_order,
                            y=totals_ordered["cumperc"],
                            name="Cumulative %",
                            yaxis="y2",
                            mode="lines+markers"
                        )

                        title_suffix = f"{int(sel_month)}/{int(sel_year)}" if not (np.isnan(sel_year) or len(months_for_year) == 0) else "All Data"
                        # Pretty x-axis labels: show the *original* category name
                        fig.update_layout(
                            title=f"Pareto of {metric} by {category} — {title_suffix}",
                            xaxis_title=category,
                            yaxis_title=f"{metric} (|value|)",
                            barmode="stack",
                            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        # Replace x tick labels from cat_key to original category values
                        fig.update_xaxes(tickmode="array", tickvals=cat_order, ticktext=cat_order)

                        safe_download_buttons_for_fig(fig, "pareto_chart", "pareto")
                        
# ================= Time Series Visualization Tab =================
with tabs[3]:
    st.header("Time Series Visualization")

    if not have("Well", "Date"):
        st.info("Need 'Well' and 'Date' columns.")
    else:
        wells_ts = sorted(df['Well'].dropna().unique())
        well_sel = st.selectbox("Select Well", wells_ts if len(wells_ts) else ["(none)"])

        series_options = [c for c in [
            "Act. Nett (bopd)", "Act. Gross (bfpd)", "Act. Gas Prod (MMscfd)",
            "Pot. Nett (bopd)", "Pot. Gross (bfpd)",
            "Pcsg", "Ptbg", "Pfl", "Psep", "Pump Eff (%)", "Choke"
        ] if c in df.columns]
        y_cols = st.multiselect("Select series to plot", series_options, default=[s for s in ["Act. Nett (bopd)", "Act. Gas Prod (MMscfd)"] if s in series_options])

        d = df[df['Well'] == well_sel].sort_values("Date")
        if len(d) and len(y_cols):
            fig = go.Figure()
            for c in y_cols:
                fig.add_trace(go.Scatter(x=d["Date"], y=d[c], mode="lines+markers", name=c))
            fig.update_layout(title=f"Time Series — {well_sel}", xaxis_title="Date", yaxis_title="Value")
            safe_download_buttons_for_fig(fig, f"time_series_{well_sel}", "ts")

# ================= Anomaly Detection Tab =================
with tabs[4]:
    st.header("Anomaly Detection (Isolation Forest)")
    target_cols_default = [c for c in ["Act. Nett (bopd)", "Act. Gas Prod (MMscfd)", "Pump Eff (%)", "Pcsg", "Ptbg", "Pfl", "Psep"] if c in df.columns]
    features = st.multiselect("Features for anomaly detection", target_cols_default, default=target_cols_default)
    contamination = st.slider("Expected anomaly fraction", 0.01, 0.30, 0.05, step=0.01)
    scope = st.radio("Scope", ["Per Well", "All Wells Combined"], index=0, horizontal=True)

    def run_iforest(data: pd.DataFrame, feat: list[str], cont: float):
        X = data[feat].dropna()
        if len(X) < 10:
            return None, None
        model = IsolationForest(n_estimators=300, contamination=cont, random_state=42)
        scores = model.fit_predict(X)  # -1 = anomaly
        return X.index, scores

    if not features:
        st.info("Select at least one feature.")
    else:
        if scope == "Per Well" and "Well" in df.columns:
            results = []
            for w in sorted(df['Well'].dropna().unique()):
                d = df[df["Well"] == w].copy()
                idx, scores = run_iforest(d, [c for c in features if c in d.columns], contamination)
                if idx is None:
                    continue
                anom_idx = idx[np.where(scores == -1)]
                if len(anom_idx):
                    base_cols = ["Date", "Well", "Structure"]
                    feat_cols = [c for c in features if c in d.columns]
                    take_cols = safe_cols(d, base_cols + feat_cols)
                    
                    # If nothing to show, skip
                    if not take_cols:
                        continue
                    
                    out = d.loc[anom_idx, take_cols]
                    if "Date" in out.columns:
                        out = out.sort_values("Date")
                    results.append(out)
                    
            if results:
                res = pd.concat(results).sort_values(["Well","Date"])
                st.dataframe(res)
            else:
                st.success("No anomalies detected with current settings.")
        else:
            d = df.copy()
            feat = [c for c in features if c in d.columns]
            idx, scores = run_iforest(d, feat, contamination)
            if idx is None:
                st.info("Not enough data for anomaly detection.")
            else:
                anom_idx = idx[np.where(scores == -1)]
                base_cols = ["Date", "Well", "Structure"]
                take_cols = safe_cols(d, base_cols + feat)
                
                res = d.loc[anom_idx, take_cols]
                # Sort only by columns that exist
                sort_keys = [c for c in ["Well", "Date"] if c in res.columns]
                if sort_keys:
                    res = res.sort_values(sort_keys)
                if len(res):
                    st.dataframe(res)
                else:
                    st.success("No anomalies detected with current settings.")

# ================= Predictive Modeling Tab =================
with tabs[5]:
    st.header("Predictive Modeling")
    target = "Act. Nett (bopd)"
    default_feats_all = [
        "Pot. Nett (bopd)", "Pot. Gross (bfpd)", "Pot. Gas (MMscfd)", "Pot. WC (%)",
        "Act. Gross (bfpd)", "Act. Gas Prod (MMscfd)", "Act. WC (%)",
        "Pcsg", "Ptbg", "Pfl", "Psep", "Pump Eff (%)", "Choke"
    ]
    default_feats = [c for c in default_feats_all if c in df.columns]
    features = st.multiselect("Select features", default_feats, default=default_feats[:8] if len(default_feats) else [])
    model_type = st.radio("Model", ["Random Forest", "Linear Regression"], index=0, horizontal=True)
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
    random_state = 42

    if target not in df.columns:
        st.info(f"Target column '{target}' not found.")
    elif not features:
        st.info("Select at least one feature.")
    else:
        data_cols = ["Well","Date", target] + features
        data = df[data_cols].dropna().copy()
        if len(data) < 30:
            st.info("Not enough complete rows to train (need ≥ 30). Try selecting fewer features or cleaning data.")
        else:
            X = data[features].values
            y = data[target].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            model = RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=-1) if model_type == "Random Forest" else LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # works on all versions
            r2   = r2_score(y_test, y_pred)

            col1, col2 = st.columns(2)
            with col1: st.metric("RMSE (bopd)", f"{rmse:,.2f}")
            with col2: st.metric("R²", f"{r2:,.3f}")

            plot_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            fig = px.scatter(plot_df, x="Actual", y="Predicted", trendline="ols", title="Predicted vs Actual (Test Set)")
            safe_download_buttons_for_fig(fig, "predicted_vs_actual", "pm_pva")

            if model_type == "Random Forest":
                importances = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
                fig_imp = px.bar(importances, x="Feature", y="Importance", title="Feature Importances (Random Forest)")
                safe_download_buttons_for_fig(fig_imp, "feature_importances", "pm_imp")

# =========================
# Footer
# =========================
st.caption("Credit: Radya Evandhika Novaldi - Jr. Engineer Petroleum")







