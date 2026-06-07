from __future__ import annotations

import logging
import pickle
import shap

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    RANDOM_SEED,
    SHAP_SAMPLE_SIZE,
    SHAP_TOP_N_FEATURES
)
from src.features import FEATURE_COLS

logger = logging.getLogger(__name__)

# PIPELINE FUNCTION — called by run_pipeline.py

def compute_shap(
    model: Any,
    test_df: pd.DataFrame,
    sample_size: int = SHAP_SAMPLE_SIZE,
    save_to: str | Path | None = None,
) -> dict[str, Any]:

    logger.info(f"Computing SHAP on {sample_size} sampled rows")

    n_sample = min(sample_size, len(test_df))
    sampled = test_df.sample(n=n_sample, random_state=RANDOM_SEED)
    x_sub = sampled[FEATURE_COLS].copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sub)

    # Rank features by mean absolute SHAP value
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranked = sorted(
        zip(FEATURE_COLS, mean_abs, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )
    top_features = [name for name, _ in ranked]
    top_values = [float(v) for _, v in ranked]

    bundle = {
        "shap_values": shap_values,
        "sample_X": x_sub,
        "top_features": top_features,
        "top_values": top_values,
    }

    if save_to:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(bundle, f)
        logger.info(f"SHAP bundle saved to {save_path}")

    return bundle


# DASHBOARD FUNCTIONS — used by app.py

def filter_subset(
    df: pd.DataFrame,
    store: str,
    dept: str,
    product: str,
    all_products_label: str = "All products in category",
) -> pd.DataFrame:
    mask = (df["store_id"] == store) & (df["dept_id"] == dept)
    if product != all_products_label:
        mask &= df["item_id"] == product
    return df[mask].copy()


def compute_shap_values(
    model: Any,
    subset: pd.DataFrame,
    feature_cols: list[str],
    sample_size: int = SHAP_SAMPLE_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray | None, pd.DataFrame | None, int]:

  
    population_size = len(subset)
    if population_size == 0:
        logger.warning("Empty subset, cannot compute SHAP")
        return None, None, 0

    n_sample = min(sample_size, population_size)
    sampled = subset.sample(n=n_sample, random_state=random_state)
    x_sub = sampled[feature_cols].copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sub)
    return shap_values, x_sub, population_size


def rank_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = SHAP_TOP_N_FEATURES,
) -> list[tuple[str, float]]:
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    ranked = sorted(
        zip(feature_names, mean_abs, strict=True), key=lambda x: x[1], reverse=True
    )
    return ranked[:top_n]


def get_top_driver(
    shap_values: np.ndarray | None, x_sub: pd.DataFrame | None
) -> str:
    if shap_values is None or x_sub is None or len(x_sub) == 0:
        return "n/a"
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    return str(x_sub.columns[np.argmax(mean_abs)])


@st.cache_data(show_spinner=False)
def compute_shap_for_selection_cached(
    df: pd.DataFrame,
    store: str,
    dept: str,
    product: str,
    feature_cols: list[str],
    _model: Any,
) -> tuple[np.ndarray | None, pd.DataFrame | None, int]:
    subset = filter_subset(df, store, dept, product)
    return compute_shap_values(_model, subset, feature_cols)
