from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.constants import (
    CHART_FONT_COLOR,
    CHART_FONT_SIZE,
    DASHBOARD_SUBTITLE,
    DASHBOARD_TITLE,
    PANEL_1_TITLE,
    PANEL_2_TITLE,
    PANEL_3_TITLE,
    PANEL_4_TITLE,
    RED,
    SLATE_200,
    SLATE_500,
    WHITE
)
import streamlit as st

from src.config import (
    ARIMA_DEFAULT_ORDER,
    CATEGORY_LABELS,
    DEFAULT_HORIZON_INDEX,
    DEFAULT_MODEL_CHOICE_INDEX,
    FORECAST_HORIZON_OPTIONS,
    MODEL_CHOICE_OPTIONS,
    SHAP_TOP_N_FEATURES,
    configure_logging,
)
from src.data_loaders import (
    load_arima_metrics,
    load_features,
    load_shap_bundle,
    load_xgb_bundle,
    resolve_test_start_date,
)
from src.features import FEATURE_COLS
from src.forecasting import (
    aggregate_daily,
    compute_display_window,
    forecast_arima,
    predict_xgboost,
)
from src.metrics import daily_mape, improvement_pct, rmse
from src.shap_analysis import (
    compute_shap_for_selection_cached,
    get_top_driver,
    rank_features,
)
from src.styles import DASHBOARD_CSS
from src.ui_components import (
    render_mape_comparison_chart,
    render_metric_cards,
    render_shap_beeswarm,
)

configure_logging()
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Woolworths NZ — Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

st.markdown(DASHBOARD_TITLE)
st.markdown(
    f"<p style='color:{SLATE_500}; font-size:13px; margin-top:-12px;'>"
    f"{DASHBOARD_SUBTITLE}"
    f"</p>",
    unsafe_allow_html=True,
)
st.divider()


try:
    df = load_features()
    xgb_bundle = load_xgb_bundle()
    arima_metrics = load_arima_metrics()
    shap_bundle = load_shap_bundle()
except FileNotFoundError as exc:
    st.error(f"Required file not found: {exc}\n\nRun `python run_pipeline.py` first.")
    st.stop()

model = xgb_bundle["model"]
model_lower = xgb_bundle.get("model_lower")
model_upper = xgb_bundle.get("model_upper")
has_quantiles = xgb_bundle.get("has_quantiles", False)
xgb_metrics = xgb_bundle["metrics"]
test_start_date = resolve_test_start_date(xgb_bundle, df)


@st.cache_resource
def load_backtest():
    try:
        with open("data/models/backtest_results.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


backtest_results = load_backtest()


ALL_PRODUCTS_LABEL = "All products in category"

with st.sidebar:
    st.markdown("## Filters")

    stores = sorted(df["store_id"].unique())
    depts = sorted(df["dept_id"].unique())

    selected_store = st.selectbox("Store", stores, index=0)

    selected_dept = st.selectbox(
        "Product Category",
        depts,
        format_func=lambda x: CATEGORY_LABELS.get(x, x),
        index=len(depts) - 1 if "FOODS_3" in depts else 0,
    )

    products_in_selection = sorted(
        df[
            (df["store_id"] == selected_store)
            & (df["dept_id"] == selected_dept)
        ]["item_id"].unique()
    )
    product_options = [ALL_PRODUCTS_LABEL] + list(products_in_selection)
    selected_product = st.selectbox(
        "Product (drill-down)",
        product_options,
        help=(
            "Pick a specific product to drill in, "
            "or keep 'All products' for category view"
        ),
    )

    horizon = st.selectbox(
        "Forecast Horizon",
        FORECAST_HORIZON_OPTIONS,
        index=DEFAULT_HORIZON_INDEX,
    )
    horizon_days = int(horizon.split()[0])

    model_choice = st.radio(
        "Model Selection",
        MODEL_CHOICE_OPTIONS,
        index=DEFAULT_MODEL_CHOICE_INDEX,
    )

    st.markdown("")
    st.button("Generate Forecast", type="primary", use_container_width=True)


filt = df[
    (df["store_id"] == selected_store) & (df["dept_id"] == selected_dept)
].copy()

if selected_product != ALL_PRODUCTS_LABEL:
    filt = filt[filt["item_id"] == selected_product]

filt = filt.sort_values("date").reset_index(drop=True)
filt["xgb_pred"] = predict_xgboost(model, filt, FEATURE_COLS)

daily = aggregate_daily(filt)

arima_pred_dict = forecast_arima(
    daily, test_start_date, order=arima_metrics.get("order", ARIMA_DEFAULT_ORDER)
)
daily["arima_pred"] = daily["date"].map(arima_pred_dict)


# PANEL 1 — FORECAST vs ACTUAL WITH CI BAND

st.markdown(f"### {PANEL_1_TITLE} ({horizon})")

product_label = (
    selected_product
    if selected_product != ALL_PRODUCTS_LABEL
    else "all products"
)
st.markdown(
    f"<p style='color:#000; font-size:12px; margin-top:-10px;'>"
    f"Daily unit sales · {selected_dept} · "
    f"Store {selected_store} · {product_label}"
    f"</p>",
    unsafe_allow_html=True,
)

plot_window = compute_display_window(daily, test_start_date, horizon_days)

fig1 = go.Figure()

fig1.add_trace(
    go.Scatter(
        x=plot_window["date"],
        y=plot_window["actual"],
        mode="lines",
        name="Actual Sales",
        line=dict(color="#1E40AF", width=2.5),
        hovertemplate="Date: %{x}<br>Actual Sales: %{y:.0f}<extra></extra>",
    )
)

if model_choice in ("XGBoost", "Compare Both"):
    xgb_test = plot_window[plot_window["date"] >= test_start_date]
    if len(xgb_test) > 0:
        fig1.add_trace(
            go.Scatter(
                x=xgb_test["date"],
                y=xgb_test["xgb_pred"],
                mode="lines",
                name="XGBoost forecast",
                line=dict(color="#EA580C", width=3),
                hovertemplate=(
                    "Date: %{x}<br>XGBoost: %{y:.0f}<extra></extra>"
                ),
            )
        )

if model_choice in ("ARIMA", "Compare Both"):
    arima_plot = plot_window.dropna(subset=["arima_pred"])
    arima_plot = arima_plot[arima_plot["date"] >= test_start_date]
    if len(arima_plot) > 0:
        fig1.add_trace(
            go.Scatter(
                x=arima_plot["date"],
                y=arima_plot["arima_pred"],
                mode="lines",
                name="ARIMA forecast",
                line=dict(color="#0B4300", width=2, dash="dash"),
                hovertemplate=(
                    "Date: %{x}<br>ARIMA: %{y:.0f}<extra></extra>"
                ),
            )
        )

fig1.add_shape(
    type="rect",
    x0=test_start_date,
    x1=plot_window["date"].max(),
    y0=0,
    y1=1,
    yref="paper",
    fillcolor="#FEF3C7",
    opacity=0.25,
    line=dict(width=0),
    layer="below",
)

fig1.add_shape(
    type="line",
    x0=test_start_date,
    x1=test_start_date,
    y0=0,
    y1=1,
    yref="paper",
    line=dict(color=RED, width=1.5, dash="dot"),
)

fig1.add_annotation(
    x=test_start_date,
    y=1,
    yref="paper",
    text="← Training  |  Test →",
    showarrow=False,
    font=dict(color=RED, size=11, family="Arial Black"),
    xanchor="center",
    yanchor="bottom",
    yshift=4,
)

fig1.update_layout(
    height=340,
    template="simple_white",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color=CHART_FONT_COLOR, size=CHART_FONT_SIZE),
    ),
    xaxis=dict(title="", showgrid=False, tickfont=dict(color="#334155", size=12)),
    yaxis=dict(
        title="Units sold",
        gridcolor="#E2E8F0",
        titlefont=dict(color=CHART_FONT_COLOR, size=14),
        tickfont=dict(color="#334155", size=CHART_FONT_SIZE),
    ),
    paper_bgcolor=WHITE,
    plot_bgcolor=WHITE,
    font=dict(color=CHART_FONT_COLOR),
    hovermode="x unified",
)

st.plotly_chart(fig1, use_container_width=True)

if len(plot_window) > 0:
    train_days = len(plot_window[plot_window["date"] < test_start_date])
    test_days = len(plot_window[plot_window["date"] >= test_start_date])
    ci_caption = ""

    st.caption(
        f"Display window: {horizon_days} days  ·  "
        f"Training ({train_days} days): "
        f"{plot_window['date'].min().strftime('%d %b %Y')} to "
        f"{(test_start_date - pd.Timedelta(days=1)).strftime('%d %b %Y')}  ·  "
        f"Test ({test_days} days): "
        f"{test_start_date.strftime('%d %b %Y')} to "
        f"{plot_window['date'].max().strftime('%d %b %Y')}"
    )


# PANEL 2 + 3 — TWO COLUMNS

col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"### {PANEL_2_TITLE}")
    st.markdown(
        f"<p style='color:{SLATE_500}; font-size:12px; margin-top:-10px;'>"
        f"Test set MAPE · Store {selected_store} · by subcategory"
        f"</p>",
        unsafe_allow_html=True,
    )

    store_test = df[
        (df["store_id"] == selected_store) & (df["date"] >= test_start_date)
    ].copy()
    store_test = store_test.sort_values("date").reset_index(drop=True)
    store_test["xgb_pred"] = predict_xgboost(model, store_test, FEATURE_COLS)

    common_depts = sorted(store_test["dept_id"].unique())
    arima_vals: list[float] = []
    xgb_vals: list[float] = []

    for dept in common_depts:
        dept_test = store_test[store_test["dept_id"] == dept]
        dept_daily_test = aggregate_daily(dept_test)
        xgb_vals.append(daily_mape(dept_daily_test, "actual", "xgb_pred"))

        dept_train = (
            df[
                (df["store_id"] == selected_store)
                & (df["dept_id"] == dept)
                & (df["date"] < test_start_date)
            ]
            .groupby("date")["sales"]
            .sum()
            .reset_index()
            .rename(columns={"sales": "actual"})
        )

        arima_dict = forecast_arima(
            pd.concat([dept_train, dept_daily_test[["date", "actual"]]]),
            test_start_date,
            order=arima_metrics.get("order", ARIMA_DEFAULT_ORDER),
        )

        if arima_dict:
            dept_daily_test["arima_pred"] = dept_daily_test["date"].map(arima_dict)
            arima_vals.append(daily_mape(dept_daily_test, "actual", "arima_pred"))
        else:
            arima_vals.append(float("nan"))

    render_mape_comparison_chart(common_depts, arima_vals, xgb_vals)
    st.caption(
        "MAPE calculated on daily-aggregated test predictions per subcategory"
    )

with col_right:
    st.markdown(f"### {PANEL_3_TITLE}")
    st.markdown(
        f"<p style='color:{SLATE_500}; font-size:12px; margin-top:-10px;'>"
        f"Top {SHAP_TOP_N_FEATURES} drivers · "
        f"filtered selection · XGBoost beeswarm"
        f"</p>",
        unsafe_allow_html=True,
    )

    shap_values, x_sub, population_size = compute_shap_for_selection_cached(
        df,
        selected_store,
        selected_dept,
        selected_product,
        FEATURE_COLS,
        model,
    )

    if shap_values is None or x_sub is None or len(x_sub) == 0:
        st.warning(
            "Not enough data in the filtered selection to compute SHAP values."
        )
    else:
        feature_ranking = rank_features(
            shap_values, list(x_sub.columns), top_n=SHAP_TOP_N_FEATURES
        )
        feature_ranking_asc = sorted(feature_ranking, key=lambda x: x[1])
        top_features = [name for name, _ in feature_ranking_asc]

        render_shap_beeswarm(shap_values, x_sub, top_features)
        st.caption(
            f"SHAP recomputed on {len(x_sub)} sampled rows "
            f"from {population_size:,} total predictions in the filtered selection"
        )


# PANEL 4 — BACKTEST STABILITY

if backtest_results is not None:
    st.markdown(f"### {PANEL_4_TITLE}")
    st.markdown(
        f"<p style='color:{SLATE_500}; font-size:12px; margin-top:-10px;'>"
        f"XGBoost MAPE across 5 walk-forward test windows · stability assessment"
        f"</p>",
        unsafe_allow_html=True,
    )

    splits = backtest_results["splits"]
    mean_mape = backtest_results["mean_mape"]
    std_mape = backtest_results["std_mape"]
    stable = backtest_results["stable"]

    if splits:
        fig_bt = go.Figure()

        split_labels = [
            f"Window {s['split']}<br>{s['test_start'].strftime('%b %d')}"
            for s in splits
        ]
        mape_values = [s["mape"] for s in splits]

        fig_bt.add_trace(
            go.Bar(
                x=split_labels,
                y=mape_values,
                marker_color="#EA580C",
                text=[f"{v:.1f}%" for v in mape_values],
                textposition="outside",
                textfont=dict(size=11, color="#000000"),
                showlegend=False,
            )
        )

        fig_bt.add_hline(
            y=mean_mape,
            line_dash="dash",
            line_color="#64748B",
            annotation_text=f"Mean: {mean_mape:.1f}%",
            annotation_position="top right",
        )

        fig_bt.update_layout(
            height=280,
            template="simple_white",
            margin=dict(l=10, r=10, t=10, b=40),
            yaxis_title="MAPE (%)",
            xaxis_title="",
            yaxis=dict(
                gridcolor=SLATE_200, tickfont=dict(size=11, color=SLATE_500)
            ),
            xaxis=dict(tickfont=dict(size=11, color=SLATE_500)),
            paper_bgcolor=WHITE,
            plot_bgcolor=WHITE,
            font=dict(color=CHART_FONT_COLOR),
        )

        st.plotly_chart(fig_bt, use_container_width=True)

        stability_icon = "✅ Stable" if stable else "⚠️ Variable"
        st.caption(
            f"Mean MAPE: {mean_mape:.2f}%  ·  "
            f"Std deviation: {std_mape:.2f}%  ·  "
            f"Stability: {stability_icon}  ·  "
            f"Lower variance = more reliable forecasts across time"
        )


# SUMMARY METRICS

st.markdown("### Summary Metrics — Filtered Selection")

test_filt = filt[filt["date"] >= test_start_date].copy()
test_daily = aggregate_daily(test_filt)

if len(test_daily) > 0:
    xgb_filt_mape = daily_mape(test_daily, "actual", "xgb_pred")
    xgb_filt_rmse = rmse(test_daily["actual"], test_daily["xgb_pred"])

    arima_summary_dict = forecast_arima(
        daily,
        test_start_date,
        order=arima_metrics.get("order", ARIMA_DEFAULT_ORDER),
    )
    if arima_summary_dict:
        test_daily["arima_pred"] = test_daily["date"].map(arima_summary_dict)
        arima_filt_mape = daily_mape(test_daily, "actual", "arima_pred")
    else:
        arima_filt_mape = float("nan")
else:
    xgb_filt_mape = float("nan")
    xgb_filt_rmse = float("nan")
    arima_filt_mape = float("nan")

improvement = improvement_pct(arima_filt_mape, xgb_filt_mape)
top_driver = get_top_driver(shap_values, x_sub)

render_metric_cards(
    xgb_mape=xgb_filt_mape,
    arima_mape=arima_filt_mape,
    xgb_rmse=xgb_filt_rmse,
    top_driver=top_driver,
    improvement=improvement,
)