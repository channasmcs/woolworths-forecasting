from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.constants import (
    DEPARTMENTS,
    STORES,
)

logger = logging.getLogger(__name__)


# Stores and categories to keep (matches the dashboard's scope)
STORES_TO_KEEP: list[str] = STORES
CATEGORIES_TO_KEEP: list[str] = DEPARTMENTS


def load_and_clean(
    raw_dir: str | Path = "data/raw",
    save_to: str | Path | None = None,
) -> pd.DataFrame:
    raw_path = Path(raw_dir)

    sales_path = raw_path / "sales_train_evaluation.csv"
    calendar_path = raw_path / "calendar.csv"
    prices_path = raw_path / "sell_prices.csv"

    for p in (sales_path, calendar_path, prices_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    logger.info(f"Loading sales from {sales_path}")
    sales_wide = pd.read_csv(sales_path)
    logger.info(f"  Loaded {len(sales_wide):,} item-store rows")

    logger.info(f"Loading calendar from {calendar_path}")
    calendar = pd.read_csv(calendar_path)

    logger.info(f"Loading prices from {prices_path}")
    prices = pd.read_csv(prices_path)

    # ---- Filter to target stores + categories ----
    sales_wide = sales_wide[
        sales_wide["store_id"].isin(STORES_TO_KEEP)
        & sales_wide["dept_id"].isin(CATEGORIES_TO_KEEP)
    ]
    logger.info(f"  Filtered to {len(sales_wide):,} item-store rows")

    # ---- Reshape wide → long ----
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_long = sales_wide.melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )
    logger.info(f"  Reshaped to {len(sales_long):,} day-item-store rows")

    # ---- Merge calendar (provides date + SNAP + wday) ----
    sales_long = sales_long.merge(calendar, on="d", how="left")
    sales_long["date"] = pd.to_datetime(sales_long["date"])

    # ---- Merge prices (provides sell_price per week) ----
    sales_long = sales_long.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    # ---- Basic cleaning ----
    sales_long["sales"] = sales_long["sales"].astype("float32")
    sales_long["sell_price"] = sales_long["sell_price"].fillna(0).astype("float32")

    # Drop rows where the item didn't exist yet (sell_price is 0)
    sales_long = sales_long[sales_long["sell_price"] > 0].reset_index(drop=True)

    logger.info(f"Cleaned: {len(sales_long):,} rows")

    # ---- Save if requested ----
    if save_to:
        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sales_long.to_parquet(save_path, index=False)
        logger.info(f"Saved to {save_path}")

    return sales_long
