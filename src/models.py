from __future__ import annotations

import logging
import pickle
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA

from src.features import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# EVALUATION — Correct MAPE that excludes zero actuals

def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if len(actual) == 0:
        return {"mape": float("nan"), "rmse": float("nan")}

    mask = actual != 0
    if mask.sum() == 0:
        mape_val = float("nan")
    else:
        mape_val = float(
            np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100
        )

    rmse_val = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    return {"mape": mape_val, "rmse": rmse_val}


# AUTO-ARIMA ORDER SELECTION

def _select_arima_order(
    series: pd.Series,
    p_range: range = range(0, 4),
    d_range: range = range(0, 2),
    q_range: range = range(0, 3),
) -> tuple[int, int, int]:

    best_aic = float("inf")
    best_order = (5, 1, 0)

    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(series, order=(p, d, q)).fit()
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    return best_order


# ARIMA — Aggregated daily evaluation (matches dashboard)

def train_arima(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_size: int = 50,
    save_to: str | Path | None = None,
    auto_order: bool = True,
    default_order: tuple[int, int, int] = (5, 1, 0),
) -> dict[str, Any]:
    logger.info(
        f"Training ARIMA on {sample_size} sampled series (auto_order={auto_order})"
    )

    all_series = train["id"].unique()
    rng = np.random.RandomState(42)
    sampled_ids = (
        rng.choice(all_series, sample_size, replace=False)
        if len(all_series) > sample_size
        else all_series
    )

    # Collect per-row predictions, then aggregate to daily totals
    rows: list[dict[str, Any]] = []
    orders_used: list[tuple[int, int, int]] = []

    for i, series_id in enumerate(sampled_ids):
        if (i + 1) % 10 == 0:
            logger.info(f"  ARIMA progress: {i + 1}/{len(sampled_ids)}")

        train_series = (
            train[train["id"] == series_id]
            .sort_values("date")["sales"]
            .reset_index(drop=True)
        )
        test_subset = (
            test[test["id"] == series_id].sort_values("date").reset_index(drop=True)
        )

        if len(train_series) < 30 or len(test_subset) == 0:
            continue

        order = _select_arima_order(train_series) if auto_order else default_order
        orders_used.append(order)

        try:
            model = ARIMA(train_series, order=order).fit()
            forecast = model.forecast(steps=len(test_subset))
        except Exception as exc:
            logger.warning(f"  ARIMA failed for {series_id}: {exc}")
            continue

        for date, actual, pred in zip(
            test_subset["date"].values,
            test_subset["sales"].values,
            forecast.values,
            strict=True,
        ):
            rows.append({
                "date": date,
                "dept_id": test_subset["dept_id"].iloc[0],
                "actual": float(actual),
                "pred": float(pred),
            })

    if not rows:
        logger.warning("No ARIMA predictions generated")
        return {
            "mape": float("nan"),
            "rmse": float("nan"),
            "order": default_order,
            "n_series": 0,
        }

    # Aggregate to daily totals (matches dashboard methodology)
    arima_df = pd.DataFrame(rows)
    daily = arima_df.groupby("date", as_index=False).agg(
        actual=("actual", "sum"), pred=("pred", "sum")
    )

    overall = evaluate(daily["actual"].values, daily["pred"].values)
    logger.info(f"  ARIMA daily-aggregated MAPE: {overall['mape']:.2f}%")

    # Per-department metrics
    per_dept_metrics: dict[str, dict[str, float]] = {}
    for dept in arima_df["dept_id"].unique():
        dept_subset = arima_df[arima_df["dept_id"] == dept]
        dept_daily = dept_subset.groupby("date", as_index=False).agg(
            actual=("actual", "sum"), pred=("pred", "sum")
        )
        per_dept_metrics[dept] = evaluate(
            dept_daily["actual"].values, dept_daily["pred"].values
        )

    mode_order = (
        Counter(orders_used).most_common(1)[0][0] if orders_used else default_order
    )

    metrics = {
        "mape": overall["mape"],
        "rmse": overall["rmse"],
        "order": mode_order,
        "auto_order_used": auto_order,
        "orders_distribution": dict(Counter(orders_used)) if orders_used else {},
        "per_dept": per_dept_metrics,
        "n_series": len(sampled_ids),
    }

    if save_to:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(metrics, f)
        logger.info(f"ARIMA metrics saved to {save_path}")

    return metrics


# XGBOOST — With quantile regression

def train_xgboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    save_to: str | Path | None = None,
    train_quantiles: bool = True,
) -> tuple[xgb.XGBRegressor, dict[str, Any]]:
    logger.info(f"Training XGBoost on {len(train):,} rows")
    logger.info(f"  Val rows: {len(val):,}, Test rows: {len(test):,}")

    if len(test) == 0:
        raise ValueError("Test dataframe is empty — cannot evaluate XGBoost")

    x_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    x_val = val[FEATURE_COLS]
    y_val = val[TARGET_COL]
    x_test = test[FEATURE_COLS]

    logger.info("  Training median (0.5 quantile) model...")
    model_median = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        early_stopping_rounds=20,
        n_jobs=-1,
        random_state=42,
    )
    model_median.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    logger.info(f"    Median trained ({model_median.best_iteration} iterations)")

    model_lower = None
    model_upper = None

    if train_quantiles:
        logger.info("  Training lower (0.1 quantile) model...")
        model_lower = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:quantileerror",
            quantile_alpha=0.1,
            n_jobs=-1,
            random_state=42,
        )
        model_lower.fit(x_train, y_train, verbose=False)

        logger.info("  Training upper (0.9 quantile) model...")
        model_upper = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:quantileerror",
            quantile_alpha=0.9,
            n_jobs=-1,
            random_state=42,
        )
        model_upper.fit(x_train, y_train, verbose=False)
        logger.info("    Quantile models trained")

    test_preds = model_median.predict(x_test)

    test_eval = pd.DataFrame({
        "date": test["date"].values,
        "dept_id": test["dept_id"].values,
        "store_id": test["store_id"].values,
        "actual": test["sales"].values.astype(float),
        "pred": test_preds.astype(float),
    })

    if model_lower is not None and model_upper is not None:
        test_eval["pred_lower"] = model_lower.predict(x_test).astype(float)
        test_eval["pred_upper"] = model_upper.predict(x_test).astype(float)

    # Aggregate to daily totals (matches dashboard)
    if "pred_lower" in test_eval.columns:
        daily_test = test_eval.groupby("date", as_index=False).agg(
            actual=("actual", "sum"),
            pred=("pred", "sum"),
            pred_lower=("pred_lower", "sum"),
            pred_upper=("pred_upper", "sum"),
        )
        within_ci = (
            (daily_test["actual"] >= daily_test["pred_lower"])
            & (daily_test["actual"] <= daily_test["pred_upper"])
        ).mean() * 100
        coverage_pct = float(within_ci)
        logger.info(f"  80% CI coverage on daily totals: {coverage_pct:.1f}%")
    else:
        daily_test = test_eval.groupby("date", as_index=False).agg(
            actual=("actual", "sum"), pred=("pred", "sum")
        )
        coverage_pct = float("nan")

    overall = evaluate(daily_test["actual"].values, daily_test["pred"].values)
    logger.info(f"  Overall MAPE: {overall['mape']:.2f}%")

    per_dept_metrics: dict[str, dict[str, float]] = {}
    for dept in test_eval["dept_id"].unique():
        dept_subset = test_eval[test_eval["dept_id"] == dept]
        dept_daily = dept_subset.groupby("date", as_index=False).agg(
            actual=("actual", "sum"), pred=("pred", "sum")
        )
        per_dept_metrics[dept] = evaluate(
            dept_daily["actual"].values, dept_daily["pred"].values
        )

    test_start_date = pd.to_datetime(test["date"].min())

    metrics = {
        "mape": overall["mape"],
        "rmse": overall["rmse"],
        "ci_coverage_80": coverage_pct,
        "per_dept": per_dept_metrics,
        "n_test_rows": len(test),
    }

    bundle = {
        "model": model_median,
        "model_lower": model_lower,
        "model_upper": model_upper,
        "metrics": metrics,
        "test_start_date": test_start_date,
        "feature_cols": FEATURE_COLS,
        "has_quantiles": train_quantiles,
    }

    if save_to:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(bundle, f)
        logger.info(f"XGBoost bundle saved to {save_path}")

    return model_median, metrics