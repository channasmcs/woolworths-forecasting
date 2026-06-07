
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    COLOUR_ACTUAL,
    COLOUR_ARIMA,
    COLOUR_BOUNDARY,
    COLOUR_SHAP_HIGH,
    COLOUR_SHAP_LOW,
    COLOUR_SHAP_MID,
    COLOUR_TEST_SHADE,
    COLOUR_XGBOOST,
    RANDOM_SEED,
)

from src.constants import (
    NAVY,
    SLATE_200,
    SLATE_300,
    WHITE
)

logger = logging.getLogger(__name__)


def render_forecast_chart(
    plot_window: pd.DataFrame,
    forecast_start_date: pd.Timestamp,
    model_choice: str,
) -> None:
    fig = go.Figure()

    # Actual line — full window
    fig.add_trace(
        go.Scatter(
            x=plot_window["date"],
            y=plot_window["actual"],
            mode="lines",
            name="Actual",
            line=dict(color=COLOUR_ACTUAL, width=2.5),
            hovertemplate="Date: %{x}<br>Actual: %{y:.0f}<extra></extra>",
        )
    )

    # XGBoost — test period only (honest forecast)
    if model_choice in ("XGBoost", "Compare Both"):
        xgb_test = plot_window[plot_window["date"] >= forecast_start_date]
        if len(xgb_test) > 0:
            fig.add_trace(
                go.Scatter(
                    x=xgb_test["date"],
                    y=xgb_test["xgb_pred"],
                    mode="lines",
                    name="XGBoost forecast",
                    line=dict(color=COLOUR_XGBOOST, width=3),
                    hovertemplate=(
                        "Date: %{x}<br>XGBoost: %{y:.0f}<extra></extra>"
                    ),
                )
            )

    # ARIMA — test period only
    if model_choice in ("ARIMA", "Compare Both"):
        arima_plot = plot_window.dropna(subset=["arima_pred"])
        arima_plot = arima_plot[arima_plot["date"] >= forecast_start_date]
        if len(arima_plot) > 0:
            fig.add_trace(
                go.Scatter(
                    x=arima_plot["date"],
                    y=arima_plot["arima_pred"],
                    mode="lines",
                    name="ARIMA forecast",
                    line=dict(color=COLOUR_ARIMA, width=2, dash="dash"),
                    hovertemplate=(
                        "Date: %{x}<br>ARIMA: %{y:.0f}<extra></extra>"
                    ),
                )
            )

    # Test region shading
    fig.add_shape(
        type="rect",
        x0=forecast_start_date,
        x1=plot_window["date"].max(),
        y0=0,
        y1=1,
        yref="paper",
        fillcolor=COLOUR_TEST_SHADE,
        opacity=0.3,
        line=dict(width=0),
        layer="below",
    )

    # Boundary line
    fig.add_shape(
        type="line",
        x0=forecast_start_date,
        x1=forecast_start_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=COLOUR_BOUNDARY, width=1.5, dash="dot"),
    )

    # Boundary annotation
    fig.add_annotation(
        x=forecast_start_date,
        y=1,
        yref="paper",
        text="← Training  |  Test →",
        showarrow=False,
        font=dict(color=COLOUR_BOUNDARY, size=11, family="Arial Black"),
        xanchor="center",
        yanchor="bottom",
        yshift=4,
    )

    fig.update_layout(
        height=340,
        template="simple_white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=NAVY, size=12),
        ),
        xaxis=dict(
            title="",
            showgrid=False,
            tickfont=dict(color="#334155", size=12),
        ),
        yaxis=dict(
            title="Units sold",
            gridcolor=SLATE_200,
            titlefont=dict(color=NAVY, size=14),
            tickfont=dict(color="#334155", size=12),
        ),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(color=NAVY),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_mape_comparison_chart(
    departments: list[str], arima_vals: list[float], xgb_vals: list[float]
) -> None:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=departments,
            x=arima_vals,
            name="ARIMA",
            orientation="h",
            marker_color=COLOUR_ARIMA,
            text=[
                f"{v:.1f}%" if not np.isnan(v) else "n/a" for v in arima_vals
            ],
            textposition="outside",
            textfont=dict(size=10, color="#000000"),
        )
    )

    fig.add_trace(
        go.Bar(
            y=departments,
            x=xgb_vals,
            name="XGBoost",
            orientation="h",
            marker_color=COLOUR_XGBOOST,
            text=[
                f"{v:.1f}%" if not np.isnan(v) else "n/a" for v in xgb_vals
            ],
            textposition="outside",
            textfont=dict(size=10, color="#000000"),
        )
    )

    fig.update_layout(
        height=360,
        barmode="group",
        template="simple_white",
        margin=dict(l=10, r=30, t=10, b=90),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(color=NAVY, size=11),
        ),
        xaxis_title="MAPE (%)",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=11, color="#000000")),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(color=NAVY),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_shap_beeswarm(
    shap_values: np.ndarray, x_sub: pd.DataFrame, top_features: list[str]
) -> None:
    fig = go.Figure()
    n_points = len(x_sub)

    for i, fname in enumerate(top_features):
        col_idx = list(x_sub.columns).index(fname)
        x_vals = shap_values[:, col_idx]
        feat_v = x_sub[fname].values
        vmin = np.nanmin(feat_v)
        vmax = np.nanmax(feat_v)
        norm = (feat_v - vmin) / (vmax - vmin + 1e-9)
        y_jitter = i + np.random.RandomState(i + RANDOM_SEED).uniform(
            -0.28, 0.28, n_points
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_jitter,
                mode="markers",
                marker=dict(
                    size=5,
                    color=norm,
                    colorscale=[
                        [0, COLOUR_SHAP_LOW],
                        [0.5, COLOUR_SHAP_MID],
                        [1, COLOUR_SHAP_HIGH],
                    ],
                    opacity=0.55,
                    showscale=(i == len(top_features) - 1),
                    colorbar=dict(
                        title=dict(text="feature<br>value", side="right"),
                        tickvals=[0, 1],
                        ticktext=["low", "high"],
                        len=0.75,
                        thickness=10,
                        x=1.02,
                    )
                    if i == len(top_features) - 1
                    else None,
                ),
                showlegend=False,
                hovertemplate=f"{fname}<br>SHAP: %{{x:.3f}}<extra></extra>",
            )
        )

    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=-0.5,
        y1=len(top_features) - 0.5,
        line=dict(color=SLATE_300, width=1, dash="dot"),
    )

    fig.update_layout(
        height=360,
        template="simple_white",
        margin=dict(l=10, r=80, t=10, b=10),
        showlegend=False,
        xaxis=dict(
            title="SHAP value (impact on prediction)",
            zeroline=False,
            tickfont=dict(size=11, color="#000000"),
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(top_features))),
            ticktext=top_features,
            tickfont=dict(size=11, color="#000000"),
            automargin=True,
        ),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(color=NAVY),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_metric_cards(
    xgb_mape: float,
    arima_mape: float,
    xgb_rmse: float,
    top_driver: str,
    improvement: float,
) -> None:

    col_a, col_b, col_c, col_d = st.columns(4)

    arima_display = f"{arima_mape:.1f}%" if not np.isnan(arima_mape) else "n/a"
    rmse_display = f"{xgb_rmse:.2f}" if not np.isnan(xgb_rmse) else "n/a"
    xgb_display = f"{xgb_mape:.1f}%" if not np.isnan(xgb_mape) else "n/a"

    with col_a:
        st.markdown(
            f"""
            <div class="metric-card metric-card-blue">
                <div class="metric-label">XGBoost MAPE</div>
                <div class="metric-blue">{xgb_display}</div>
                <div class="metric-sub">▼ {improvement:.1f}% vs ARIMA</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            f"""
            <div class="metric-card metric-card-orange">
                <div class="metric-label">ARIMA MAPE</div>
                <div class="metric-orange">{arima_display}</div>
                <div class="metric-sub">baseline reference</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_c:
        st.markdown(
            f"""
            <div class="metric-card metric-card-green">
                <div class="metric-label">XGBoost RMSE</div>
                <div class="metric-green">{rmse_display}</div>
                <div class="metric-sub">units · daily sales</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_d:
        st.markdown(
            f"""
            <div class="metric-card metric-card-purple">
                <div class="metric-label">Top SHAP Driver</div>
                <div class="metric-purple">{top_driver}</div>
                <div class="metric-sub">most influential feature</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
