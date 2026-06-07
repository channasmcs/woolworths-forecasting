from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import (
    ARIMA_METRICS_PATH,
    DEFAULT_TEST_PERIOD_DAYS,
    FEATURES_PATH,
    SHAP_VALUES_PATH,
    XGB_MODEL_PATH,
)

logger = logging.getLogger(__name__)


@st.cache_data
def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Features file not found: {path}. "
            "Run `python run_pipeline.py` to generate it."
        )

    logger.info(f"Loading features from {path}")
    df = pd.read_parquet(path)

    required_cols = {"date", "store_id", "dept_id", "item_id", "sales"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Features missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    logger.info(
        f"Loaded {len(df):,} feature rows "
        f"({df['date'].min().date()} to {df['date'].max().date()})"
    )
    return df


@st.cache_resource
def load_xgb_bundle(path: Path = XGB_MODEL_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {path}. "
            "Run `python run_pipeline.py` to train it."
        )

    logger.info(f"Loading XGBoost bundle from {path}")
    with open(path, "rb") as f:
        bundle: dict[str, Any] = pickle.load(f)
    return bundle


@st.cache_resource
def load_arima_metrics(path: Path = ARIMA_METRICS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ARIMA metrics not found: {path}")

    logger.info(f"Loading ARIMA metrics from {path}")
    with open(path, "rb") as f:
        metrics: dict[str, Any] = pickle.load(f)
    return metrics


@st.cache_resource
def load_shap_bundle(path: Path = SHAP_VALUES_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"SHAP values not found: {path}")

    logger.info(f"Loading SHAP bundle from {path}")
    with open(path, "rb") as f:
        bundle: dict[str, Any] = pickle.load(f)
    return bundle


def resolve_test_start_date(
    xgb_bundle: dict[str, Any], df: pd.DataFrame
) -> pd.Timestamp:
    if "test_start_date" in xgb_bundle:
        date = pd.to_datetime(xgb_bundle["test_start_date"])
        logger.debug(f"Using pipeline-saved test_start_date: {date.date()}")
        return date

    fallback = df["date"].max() - pd.Timedelta(days=DEFAULT_TEST_PERIOD_DAYS)
    logger.warning(
        f"test_start_date not found in xgb_bundle; "
        f"falling back to {fallback.date()} (last {DEFAULT_TEST_PERIOD_DAYS} days)"
    )
    return fallback
