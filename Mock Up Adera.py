# Adera Data Dashboard — full app
# --------------------------------------------------------------
# Streamlit app for "Processed Database.xlsx" (sheet: ProDB)
# Header row is row 3 in Excel (0-indexed header=2), data starts row 4.
# --------------------------------------------------------------

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
    # header=2 means Excel row 3 is used for column names
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, engine="openpyxl")
    return df

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def safe_download_buttons_for_fig(fig: go.Figure, base_filename: str, key_prefix: str):
    """
    Try to produce a PNG for download using kaleido.
    If not available, provide an HTML download fallback (never crash).
    """
    st.plotly_chart(fig, use_container_width=True)

    # Try PNG via kaleido
    png_buffer = None
    kaleido_ok = False
    try:
        png_bytes = pio.to_image(fig, format="png", scale=2)
        png_buffer = BytesIO(png_bytes)
        kaleido_ok = True
    except Exception:
        kaleido_ok = False

    # PNG button (if available)
    if kaleido_ok and png_buffer is not None:
        st.download_button(
            label="Download as PNG",
            data=png_buffer,
            file_name=f"{base_filename}.png",
            mime="image/png",
            key=f"{key_prefix}_png"
        )

    # HTML fallback (always available)
    html_bytes = fig.to_html(full_html=False).encode("utf-8")
    st.download_button(
        label="Download as HTML",
        data=html_bytes,
        file_name=f"{base_filename}.html",
        mime="text/html",
        key=f"{key_prefix}_html"
    )

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Soft-normalize common header variants without breaking if not present.
    (Silently ignore keys that are missing.)
    """
    rename_map = {
        # Potentials
        'Pot. Gross(bfpd)': 'Pot. Gross (bfpd)',
        'Pot. Nett(bopd)': 'Pot. Nett (bopd)',
        'Pot. WC(%)': 'Pot. WC (%)',
        'Pot. Gas(MMSCFD)': 'Pot. Gas (MMscfd)',
        # Actuals
        'Act. Gross(bfpd)': 'Act. Gross (bfpd)',
        'Act. Nett(bopd)': 'Act. Nett (bopd)',
        'Act. WC(%)': 'Act. WC (%)',
        'Act. Gas Prod(MMscfd)': 'Act. Gas Prod (MMscfd)',
        # Efficiencies
        'Pump EFf(%)': 'Pump Eff (%)',
        'Pump EFf (%)': 'Pump Eff (%)',
        # Lifting method
        'LiftingMethod': 'Lifting Method',
        'Lifting Method ': 'Lifting Method',
        # Gas units casing normalization
        'Pot. Gas (MMSCFD)': 'Pot. Gas (MMscfd)',
        'Act. Gas Prod (MMSCFD)': 'Act. Gas Prod (MMscfd)',
    }
    existing_map = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing_map:
        df = df.rename(columns=existing_map)
    return df

def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    df['Year']  = df['Date'].dt.year
    return df

# =========================
# Data load & preparation
# =========================
with st.sidebar:
    st.header("Data Settings")
    excel_path = st.text_input("Excel file path", value="Processed Database.xlsx")
    sheet_name = st.text_input("Sheet name", value="ProDB")
    header_row = st.number_input("Header row index (0-based)", value=2, min_value=0, step=1)
    st.caption("Your headers are on Excel row 3 → 0-based index = 2.")

df_raw = load_excel(excel_path, sheet_name=sheet_name, header_row=header_row)
df = df_raw.copy()
df = normalize_headers(df)

# Ensure expected columns exist (based on your schema)
expected_cols = [
    "No","Date","Structure","Well","Down Time","Pot. Gross (bfpd)","Pot. Nett (bopd)","Pot. WC (%)",
    "Pot. Gas (MMscfd)","Last Test Date","Act. Gross (bfpd)","Act. Nett (bopd)","Act. WC (%)",
    "Act. Gas Prod (MMscfd)","Loss/Gain","Zone","Choke","Pump Eff (%)","Production System",
    "SL/Freq","SPM/AMP","PLGR/CAP","Pcsg","Ptbg","Pfl","Psep","SFL","DFL","PSD","Remark",
    "Lifting Method"  # Column AE
]
# Don't fail if some are missing; just proceed with what's available.
present_cols = [c for c in expected_cols if c in df.columns]
missing_cols = [c for c in expected_cols if c not in df.columns]
if missing_cols:
    with st.expander("⚠️ Missing columns (informational):"):
        st.write(missing_cols)

# Convert types
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

# ================= Statistics Tab =================
with tabs[0]:
    st.header('Statistics')

    # Year / Month selection
    st.subheader("Select Month and Year for Analysis")
    avail_years = sorted(df['Year'].dropna().unique(), reverse=True)
    selected_year_stat = st.selectbox("Select Year", avail_years, key="stat_year")
    months_for_year = sorted(df[df['Year'] == selected_year_stat]['Month'].dropna().unique())
    selected_month_stat = st.selectbox("Select Month", months_for_year, key="stat_month")

    # Top-N selector
    top_n = st.selectbox("Select number of top producers", [3, 5, 10, 20], index=3, key="stat_top_n")

    # Filter to the month/year and take latest test per well
    stat_df = df[(df['Year'] == selected_year_stat) & (df['Month'] == selected_month_stat)].copy()
    stat_df = stat_df.sort_values("Date").groupby("Well", as_index=False).last()

    # Peak values & first test date across all data
    peak_df = df.groupby("Well", as_index=False).agg({
        "Act. Nett (bopd)": "max",
        "Act. Gas Prod (MMscfd)": "max",
        "Date": "min"
    }).rename(columns={
        "Date": "First Test Date",
        "Act. Nett (bopd)": "Peak Oil (bopd)",
        "Act. Gas Prod (MMscfd)": "Peak Gas (MMscfd)"
    })

    stat_df = pd.merge(stat_df, peak_df, on="Well", how="left")

    # Lifespan
    stat_df['Lifespan Days'] = (stat_df['Date'] - stat_df['First Test Date']).dt.days

    def format_lifespan(days):
        days = 0 if pd.isna(days) else int(days)
        years = days // 365
        months = (days % 365) // 30
        return years, months

    if len(stat_df):
        stat_df['Years'], stat_df['Months'] = zip(*stat_df['Lifespan Days'].apply(format_lifespan))
    else:
        stat_df['Years'], stat_df['Months'] = [], []

    def lifespan_label(years, months):
        label = f"{years} year{'s' if years != 1 else ''}, {months} month{'s' if months != 1 else ''}"
        label += " (New)" if years < 2 else " (Mature)"
        return label

    if len(stat_df):
        stat_df['Well Lifespan'] = stat_df.apply(lambda r: lifespan_label(r['Years'], r['Months']), axis=1)

    # Top Oil
    st.subheader("Top Oil Producers")
    if "Act. Nett (bopd)" in stat_df.columns:
        top_oil = stat_df.sort_values("Act. Nett (bopd)", ascending=False).head(top_n)
        st.dataframe(top_oil[["Well", "Structure", "Act. Nett (bopd)", "Peak Oil (bopd)", "Well Lifespan"]])
        oil_fig = px.bar(top_oil, x="Well", y="Act. Nett (bopd)", color="Structure", title="Top Oil Producers")
        safe_download_buttons_for_fig(oil_fig, "top_oil_producers", "stat_oil")
    else:
        st.info("Column 'Act. Nett (bopd)' not found.")

    # Top Gas
    st.subheader("Top Gas Producers")
    if "Act. Gas Prod (MMscfd)" in stat_df.columns:
        top_gas = stat_df.sort_values("Act. Gas Prod (MMscfd)", ascending=False).head(top_n)
        st.dataframe(top_gas[["Well", "Structure", "Act. Gas Prod (MMscfd)", "Peak Gas (MMscfd)", "Well Lifespan"]])
        gas_fig = px.bar(top_gas, x="Well", y="Act. Gas Prod (MMscfd)", color="Structure", title="Top Gas Producers")
        safe_download_buttons_for_fig(gas_fig, "top_gas_producers", "stat_gas")
    else:
        st.info("Column 'Act. Gas Prod (MMscfd)' not found.")

# ================= Rate Change Alerts Tab =================
with tabs[1]:
    st.header("Rate Change Alerts")
    st.caption("Compares the latest two tests per well and flags significant changes.")

    wells = sorted(df['Well'].dropna().unique())
    min_points = st.slider("Minimum data points per well", 2, 10, 2)
    pct_threshold = st.slider("Percent change threshold (%)", 1, 100, 20)

    alerts = []
    for w in wells:
        d = df[df['Well'] == w].sort_values('Date')
        if len(d) < min_points:
            continue
        last = d.iloc[-1]
        prev = d.iloc[-2]

        def pct_change(a, b):
            if pd.isna(a) or pd.isna(b) or b == 0:
                return np.nan
            return 100.0 * (a - b) / abs(b)

        oil_pct = pct_change(last.get("Act. Nett (bopd)"), prev.get("Act. Nett (bopd)"))
        gas_pct = pct_change(last.get("Act. Gas Prod (MMscfd)"), prev.get("Act. Gas Prod (MMscfd)"))

        flag = (not pd.isna(oil_pct) and abs(oil_pct) >= pct_threshold) or \
               (not pd.isna(gas_pct) and abs(gas_pct) >= pct_threshold)

        if flag:
            alerts.append({
                "Well": w,
                "Structure": last.get("Structure", np.nan),
                "Prev Date": prev.get("Date", np.nan),
                "Last Date": last.get("Date", np.nan),
                "Oil Δ%": oil_pct,
                "Gas Δ%": gas_pct,
                "Lifting Method": last.get("Lifting Method", np.nan),
                "Pump Eff (%)": last.get("Pump Eff (%)", np.nan),
                "Choke": last.get("Choke", np.nan),
            })

    if alerts:
        alert_df = pd.DataFrame(alerts).sort_values("Last Date", ascending=False)
        st.dataframe(alert_df)
    else:
        st.success("No wells crossed the threshold based on the latest two tests.")

# ================= Pareto Chart Tab =================
with tabs[2]:
    st.header("Pareto Chart")
    st.caption("Build a Pareto of cumulative contribution by a chosen category.")

    # Choose metric and category
    metric = st.selectbox(
        "Metric (absolute values are used for Pareto accumulation)",
        ["Loss/Gain", "Down Time", "Act. Nett (bopd)", "Act. Gas Prod (MMscfd)"],
        index=0
    )
    category = st.selectbox(
        "Category",
        ["Structure", "Well", "Zone", "Lifting Method", "Production System", "Remark"],
        index=0
    )

    # Aggregate by latest per well to avoid over-counting across dates (toggle)
    agg_mode = st.radio("Aggregation mode", ["Latest per Well", "Sum over all rows"], index=0, horizontal=True)

    data_pareto = df.copy()
    if agg_mode == "Latest per Well":
        data_pareto = data_pareto.sort_values("Date").groupby("Well", as_index=False).last()

    if metric not in data_pareto.columns or category not in data_pareto.columns:
        st.info(f"Columns '{metric}' or '{category}' not found.")
    else:
        p = data_pareto[[category, metric]].dropna()
        # Use absolute contribution for Pareto visualization (typical practice for loss/gain/downtime)
        p["abs_metric"] = p[metric].abs()
        pareto = p.groupby(category, as_index=False)["abs_metric"].sum().sort_values("abs_metric", ascending=False)
        pareto["cumperc"] = 100 * pareto["abs_metric"].cumsum() / pareto["abs_metric"].sum()

        fig = go.Figure()
        fig.add_bar(x=pareto[category], y=pareto["abs_metric"], name=f"{metric} contribution")
        fig.add_scatter(x=pareto[category], y=pareto["cumperc"], name="Cumulative %", yaxis="y2", mode="lines+markers")

        fig.update_layout(
            title=f"Pareto of {metric} by {category}",
            xaxis_title=category,
            yaxis_title=f"{metric} (|value|)",
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100])
        )
        safe_download_buttons_for_fig(fig, "pareto_chart", "pareto")

# ================= Time Series Visualization Tab =================
with tabs[3]:
    st.header("Time Series Visualization")

    wells_ts = sorted(df['Well'].dropna().unique())
    if not wells_ts:
        st.info("No wells found.")
    else:
        well_sel = st.selectbox("Select Well", wells_ts)

        series_options = [
            "Act. Nett (bopd)",
            "Act. Gross (bfpd)",
            "Act. Gas Prod (MMscfd)",
            "Pot. Nett (bopd)",
            "Pot. Gross (bfpd)",
            "Pcsg", "Ptbg", "Pfl", "Psep",
            "Pump Eff (%)", "Choke"
        ]
        y_cols = st.multiselect("Select series to plot", series_options, default=["Act. Nett (bopd)", "Act. Gas Prod (MMscfd)"])

        d = df[df['Well'] == well_sel].sort_values("Date")
        if not len(d):
            st.info("No data for the selected well.")
        elif not y_cols:
            st.info("Pick at least one series.")
        else:
            fig = go.Figure()
            for c in y_cols:
                if c in d.columns:
                    fig.add_trace(go.Scatter(x=d["Date"], y=d[c], mode="lines+markers", name=c))
            fig.update_layout(title=f"Time Series — {well_sel}", xaxis_title="Date", yaxis_title="Value")
            safe_download_buttons_for_fig(fig, f"time_series_{well_sel}", "ts")

# ================= Anomaly Detection Tab =================
with tabs[4]:
    st.header("Anomaly Detection (Isolation Forest)")

    wells_ad = sorted(df['Well'].dropna().unique())
    target_cols_default = ["Act. Nett (bopd)", "Act. Gas Prod (MMscfd)", "Pump Eff (%)", "Pcsg", "Ptbg", "Pfl", "Psep"]
    features = st.multiselect("Features for anomaly detection", target_cols_default, default=target_cols_default)
    contamination = st.slider("Expected anomaly fraction", 0.01, 0.30, 0.05, step=0.01)
    scope = st.radio("Scope", ["Per Well", "All Wells Combined"], index=0, horizontal=True)

    def run_iforest(data: pd.DataFrame, feat: list[str], cont: float):
        X = data[feat].dropna()
        if len(X) < 10:
            return None, None
        model = IsolationForest(n_estimators=300, contamination=cont, random_state=42)
        scores = model.fit_predict(X)  # -1 = anomaly, 1 = normal
        return X.index, scores

    if not features:
        st.info("Select at least one feature.")
    else:
        if scope == "Per Well":
            results = []
            for w in wells_ad:
                d = df[df["Well"] == w].copy()
                idx, scores = run_iforest(d, [c for c in features if c in d.columns], contamination)
                if idx is None:
                    continue
                anom_idx = idx[np.where(scores == -1)]
                if len(anom_idx):
                    out = d.loc[anom_idx, ["Date","Well","Structure"] + [c for c in features if c in d.columns]].sort_values("Date")
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
                res = d.loc[anom_idx, ["Date","Well","Structure"] + feat].sort_values(["Well","Date"])
                if len(res):
                    st.dataframe(res)
                else:
                    st.success("No anomalies detected with current settings.")

# ================= Predictive Modeling Tab =================
with tabs[5]:
    st.header("Predictive Modeling")

    st.caption("Train a simple model to predict **Act. Nett (bopd)** from available features.")

    target = "Act. Nett (bopd)"
    default_feats = [
        "Pot. Nett (bopd)", "Pot. Gross (bfpd)", "Pot. Gas (MMscfd)", "Pot. WC (%)",
        "Act. Gross (bfpd)", "Act. Gas Prod (MMscfd)", "Act. WC (%)",
        "Pcsg", "Ptbg", "Pfl", "Psep", "Pump Eff (%)", "Choke"
    ]
    features = st.multiselect("Select features", [c for c in default_feats if c in df.columns], default=[c for c in default_feats if c in df.columns][:8])
    model_type = st.radio("Model", ["Random Forest", "Linear Regression"], index=0, horizontal=True)
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
    random_state = 42

    if target not in df.columns:
        st.info(f"Target column '{target}' not found.")
    elif not features:
        st.info("Select at least one feature.")
    else:
        data = df[["Well","Date", target] + features].dropna().copy()
        if len(data) < 30:
            st.info("Not enough complete rows to train (need ≥ 30). Try selecting fewer features or cleaning data.")
        else:
            X = data[features].values
            y = data[target].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            if model_type == "Random Forest":
                model = RandomForestRegressor(
                    n_estimators=400,
                    max_depth=None,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                model = LinearRegression()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE (bopd)", f"{rmse:,.2f}")
            with col2:
                st.metric("R²", f"{r2:,.3f}")

            # Scatter: predicted vs actual
            plot_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            fig = px.scatter(plot_df, x="Actual", y="Predicted", trendline="ols", title="Predicted vs Actual (Test Set)")
            safe_download_buttons_for_fig(fig, "predicted_vs_actual", "pm_pva")

            # Feature importance (RF only)
            if model_type == "Random Forest":
                importances = pd.DataFrame({
                    "Feature": features,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)

                fig_imp = px.bar(importances, x="Feature", y="Importance", title="Feature Importances (Random Forest)")
                safe_download_buttons_for_fig(fig_imp, "feature_importances", "pm_imp")

# =========================
# Footer
# =========================
st.caption("Tip: If PNG download fails, install 'kaleido' (`pip install kaleido`).")
