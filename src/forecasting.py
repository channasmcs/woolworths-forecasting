from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import ARIMA_DEFAULT_ORDER, ARIMA_MIN_TRAINING_DAYS

logger = logging.getLogger(__name__)


def predict_xgboost(
    model: Any, df: pd.DataFrame, feature_cols: list[str]
) -> np.ndarray:
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    return model.predict(df[feature_cols])


def aggregate_daily(
    df: pd.DataFrame,
    actual_col: str = "sales",
    pred_col: str = "xgb_pred",
) -> pd.DataFrame:
    return (
        df.groupby("date")
        .agg(actual=(actual_col, "sum"), xgb_pred=(pred_col, "sum"))
        .reset_index()
        .sort_values("date")
    )


def forecast_arima(
    daily_df: pd.DataFrame,
    test_start_date: pd.Timestamp,
    order: tuple[int, int, int] = ARIMA_DEFAULT_ORDER,
) -> dict[pd.Timestamp, float]:

    # Late import — statsmodels is heavy and only needed here
    from statsmodels.tsa.arima.model import ARIMA

    train_series = (
        daily_df[daily_df["date"] < test_start_date]
        .set_index("date")["actual"]
    )
    test_dates = daily_df[daily_df["date"] >= test_start_date]["date"].tolist()

    if len(train_series) < ARIMA_MIN_TRAINING_DAYS:
        logger.warning(
            f"Only {len(train_series)} training days available, "
            f"need {ARIMA_MIN_TRAINING_DAYS} for ARIMA"
        )
        return {}

    if not test_dates:
        logger.warning("No test dates available for ARIMA forecast")
        return {}

    try:
        model = ARIMA(train_series, order=order).fit()
        forecast = model.forecast(steps=len(test_dates))
        return {d: float(v) for d, v in zip(test_dates, forecast.values, strict=True)}
    except (ValueError, np.linalg.LinAlgError) as e:
        logger.warning(f"ARIMA fit failed (order={order}): {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected ARIMA error: {e}", exc_info=True)
        return {}


def compute_display_window(
    daily_df: pd.DataFrame,
    test_start_date: pd.Timestamp,
    horizon_days: int,
) -> pd.DataFrame:
    half_window = horizon_days // 2
    window_start = test_start_date - pd.Timedelta(days=half_window)
    window_end = test_start_date + pd.Timedelta(days=half_window)

    return daily_df[
        (daily_df["date"] >= window_start) & (daily_df["date"] < window_end)
    ].copy()
