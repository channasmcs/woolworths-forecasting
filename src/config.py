from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

from src.constants import (
    CHART_ACTUAL,
    CHART_XGBOOST,
    CHART_ARIMA,
    DEPT_LABELS,
    RED
)

 
# PATHS
 

# Project root — assumes this file lives in src/
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = BASE_DIR / "data"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
MODELS_DIR: Final[Path] = DATA_DIR / "models"

FEATURES_PATH: Final[Path] = Path(
    os.getenv("FEATURES_PATH", str(PROCESSED_DIR / "features.parquet"))
)
XGB_MODEL_PATH: Final[Path] = Path(
    os.getenv("XGB_MODEL_PATH", str(MODELS_DIR / "xgb_model.pkl"))
)
ARIMA_METRICS_PATH: Final[Path] = Path(
    os.getenv("ARIMA_METRICS_PATH", str(MODELS_DIR / "arima_metrics.pkl"))
)
SHAP_VALUES_PATH: Final[Path] = Path(
    os.getenv("SHAP_VALUES_PATH", str(MODELS_DIR / "shap_values.pkl"))
)

 
# MODEL CONFIGURATION
 

ARIMA_DEFAULT_ORDER: Final[tuple[int, int, int]] = (5, 1, 0)
ARIMA_MIN_TRAINING_DAYS: Final[int] = 30
DEFAULT_TEST_PERIOD_DAYS: Final[int] = 28

 
# SHAP CONFIGURATION
 

SHAP_SAMPLE_SIZE: Final[int] = 500
SHAP_TOP_N_FEATURES: Final[int] = 10
RANDOM_SEED: Final[int] = 42

 
# UI CONFIGURATION
 

FORECAST_HORIZON_OPTIONS: Final[list[str]] = [
    "7 days",
    "14 days",
    "28 days",
    "60 days",
]
DEFAULT_HORIZON_INDEX: Final[int] = 3  # 60 days

MODEL_CHOICE_OPTIONS: Final[list[str]] = ["ARIMA", "XGBoost", "Compare Both"]
DEFAULT_MODEL_CHOICE_INDEX: Final[int] = 2  # Compare Both

CATEGORY_LABELS: Final[dict[str, str]] = DEPT_LABELS

 
# CHART COLOURS
 

COLOUR_ACTUAL: Final[str] = CHART_ACTUAL          # Blue — actual sales
COLOUR_XGBOOST: Final[str] = CHART_XGBOOST         # Orange — XGBoost forecast
COLOUR_ARIMA: Final[str] = CHART_ARIMA           # Dark green — ARIMA forecast
COLOUR_BOUNDARY: Final[str] = RED        # Red — train/test divider
COLOUR_TEST_SHADE: Final[str] = "#FEF3C7"      # Yellow — test region

COLOUR_SHAP_LOW: Final[str] = "#3B82F6"        # Blue — low feature value
COLOUR_SHAP_MID: Final[str] = "#A78BFA"        # Purple — mid value
COLOUR_SHAP_HIGH: Final[str] = RED       # Red — high feature value

 
# LOGGING
 

LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


def configure_logging() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
