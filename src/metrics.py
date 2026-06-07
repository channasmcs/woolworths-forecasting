from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray | pd.Series | list[float]


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)

    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            f"Shape mismatch: actual={actual_arr.shape}, "
            f"predicted={predicted_arr.shape}"
        )

    mask = actual_arr != 0
    if mask.sum() == 0:
        logger.warning("All actual values are zero — returning NaN")
        return float("nan")

    return float(
        np.mean(
            np.abs(actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask]
        )
        * 100
    )


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)

    if actual_arr.shape != predicted_arr.shape:
        raise ValueError(
            f"Shape mismatch: actual={actual_arr.shape}, "
            f"predicted={predicted_arr.shape}"
        )

    return float(np.sqrt(np.mean((actual_arr - predicted_arr) ** 2)))


def improvement_pct(baseline_mape: float, model_mape: float) -> float:
    if (
        np.isnan(baseline_mape)
        or np.isnan(model_mape)
        or baseline_mape <= 0
    ):
        return 0.0

    return float((baseline_mape - model_mape) / baseline_mape * 100)


def daily_mape(
    daily_df: pd.DataFrame,
    actual_col: str = "actual",
    pred_col: str = "pred",
) -> float:
    if actual_col not in daily_df.columns:
        raise KeyError(f"Missing column: {actual_col}")
    if pred_col not in daily_df.columns:
        raise KeyError(f"Missing column: {pred_col}")

    return mape(daily_df[actual_col], daily_df[pred_col])
