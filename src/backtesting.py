from __future__ import annotations

import logging
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.constants import BACKTEST_STABLE_STD
import xgboost as xgb

from src.features import FEATURE_COLS, TARGET_COL
from src.models import evaluate

logger = logging.getLogger(__name__)


def walk_forward_backtest(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size_days: int = 28,
    save_to: str | Path | None = None,
) -> dict[str, Any]:
    logger.info(
        f"Walk-forward backtest: {n_splits} splits × {test_size_days} days"
    )

    df_sorted = df.sort_values("date").reset_index(drop=True)
    max_date = df_sorted["date"].max()

    split_results: list[dict[str, Any]] = []

    for i in range(n_splits):
        test_end = max_date - timedelta(days=test_size_days * i)
        test_start = test_end - timedelta(days=test_size_days)

        train_df = df_sorted[df_sorted["date"] < test_start]
        test_df = df_sorted[
            (df_sorted["date"] >= test_start) & (df_sorted["date"] < test_end)
        ]

        if len(train_df) < 1000 or len(test_df) < 7:
            logger.warning(
                f"  Split {i + 1}: insufficient data "
                f"(train={len(train_df)}, test={len(test_df)}). Skipping."
            )
            continue

        logger.info(
            f"  Split {i + 1}/{n_splits}: "
            f"train={len(train_df):,} rows, test={len(test_df):,} rows, "
            f"window={test_start.date()} to {test_end.date()}"
        )

        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(train_df[FEATURE_COLS], train_df[TARGET_COL], verbose=False)

        predictions = model.predict(test_df[FEATURE_COLS])

        test_eval = pd.DataFrame({
            "date": test_df["date"].values,
            "actual": test_df[TARGET_COL].values.astype(float),
            "pred": predictions.astype(float),
        })

        daily = test_eval.groupby("date", as_index=False).agg(
            actual=("actual", "sum"), pred=("pred", "sum")
        )

        metrics = evaluate(daily["actual"].values, daily["pred"].values)

        split_results.append({
            "split": i + 1,
            "test_start": test_start,
            "test_end": test_end,
            "mape": metrics["mape"],
            "rmse": metrics["rmse"],
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "n_test_days": len(daily),
        })

        logger.info(
            f"    MAPE: {metrics['mape']:.2f}% · RMSE: {metrics['rmse']:.2f}"
        )

    mape_values = [s["mape"] for s in split_results if not np.isnan(s["mape"])]

    if mape_values:
        mean_mape = float(np.mean(mape_values))
        std_mape = float(np.std(mape_values))
    else:
        mean_mape = float("nan")
        std_mape = float("nan")

    stable = bool(std_mape < BACKTEST_STABLE_STD) if not np.isnan(std_mape) else False

    result = {
        "splits": split_results,
        "mean_mape": mean_mape,
        "std_mape": std_mape,
        "stable": stable,
        "n_splits": len(split_results),
        "test_size_days": test_size_days,
    }

    logger.info(
        f"Backtest complete: mean MAPE = {mean_mape:.2f}% "
        f"± {std_mape:.2f}% across {len(split_results)} splits"
    )

    if save_to:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
        logger.info(f"Backtest results saved to {save_path}")

    return result