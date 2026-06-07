
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FEATURE_COLS_SET

logger = logging.getLogger(__name__)

# CONSTANTS (used by app.py / dashboard)

FEATURE_COLS: list[str] = FEATURE_COLS_SET

TARGET_COL: str = "sales"


# FEATURE ENGINEERING

def engineer_features(
    df: pd.DataFrame, save_to: str | Path | None = None
) -> pd.DataFrame:
    logger.info(f"Engineering features on {len(df):,} rows")

    df = df.sort_values(["id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    # ---- Lag features (past sales values) ----
    for lag in (7, 14, 28):
        df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag)

    # ---- Rolling statistics — must use shift(1) to avoid leakage ----
    shifted = df.groupby("id")["sales"].shift(1)
    df["rolling_7_mean"] = shifted.groupby(df["id"]).rolling(7).mean().values
    df["rolling_28_mean"] = shifted.groupby(df["id"]).rolling(28).mean().values
    df["rolling_7_std"] = shifted.groupby(df["id"]).rolling(7).std().values

    # ---- Calendar features ----
    df["day_of_week"] = df["date"].dt.dayofweek

    # `wday` may already exist from the calendar.csv; if not, derive
    if "wday" not in df.columns:
        df["wday"] = df["date"].dt.dayofweek + 1

    # ---- SNAP flag — consolidate state-specific columns ----
    snap_cols = [c for c in ("snap_CA", "snap_TX", "snap_WI") if c in df.columns]
    if snap_cols:
        df["snap_flag"] = df[snap_cols].sum(axis=1).clip(upper=1).astype(int)
    else:
        df["snap_flag"] = 0

    # ---- Price handling ----
    if "sell_price" not in df.columns:
        df["sell_price"] = 0.0
    df["sell_price"] = df["sell_price"].fillna(0).astype(float)

    # ---- Drop rows with NaN features (early dates without enough history) ----
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    logger.info(
        f"Dropped {before - len(df):,} rows with insufficient history; "
        f"{len(df):,} rows remain"
    )

    # ---- Save if requested ----
    if save_to:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        logger.info(f"Features saved to {save_path}")

    return df


# CHRONOLOGICAL SPLIT

def chronological_split(
    df: pd.DataFrame,
    val_days: int = 28,
    test_days: int = 28,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(date_col).copy()
    max_date = df[date_col].max()

    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = df[df[date_col] < val_start]
    val = df[(df[date_col] >= val_start) & (df[date_col] < test_start)]
    test = df[df[date_col] >= test_start]

    logger.info(
        f"Chronological split: train={len(train):,} (< {val_start.date()}) | "
        f"val={len(val):,} ({val_start.date()} to {test_start.date()}) | "
        f"test={len(test):,} ({test_start.date()} to {max_date.date()})"
    )

    return train, val, test
