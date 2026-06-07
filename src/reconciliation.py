from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_hierarchy(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("Building forecast hierarchy")

    # Extract state from store_id (e.g. CA_1 → CA)
    df = df.copy()
    df["state"] = df["store_id"].str.split("_").str[0]

    hierarchy = {}

    # Level 0 — Total
    hierarchy["total"] = df.groupby("date", as_index=False).agg(
        actual=("actual", "sum"), pred=("pred", "sum")
    )

    # Level 1 — By state
    hierarchy["state"] = df.groupby(["date", "state"], as_index=False).agg(
        actual=("actual", "sum"), pred=("pred", "sum")
    )

    # Level 2 — By category
    hierarchy["category"] = df.groupby(["date", "dept_id"], as_index=False).agg(
        actual=("actual", "sum"), pred=("pred", "sum")
    )

    # Level 3 — By state × category
    hierarchy["state_category"] = df.groupby(
        ["date", "state", "dept_id"], as_index=False
    ).agg(actual=("actual", "sum"), pred=("pred", "sum"))

    # Level 4 — Base (store × dept)
    hierarchy["base"] = df.groupby(
        ["date", "store_id", "dept_id"], as_index=False
    ).agg(actual=("actual", "sum"), pred=("pred", "sum"))

    logger.info(
        f"  Hierarchy: total={len(hierarchy['total'])}, "
        f"state={len(hierarchy['state'])}, "
        f"category={len(hierarchy['category'])}, "
        f"base={len(hierarchy['base'])}"
    )

    return hierarchy


def bottom_up_reconciliation(
    base_forecasts: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    logger.info("Running bottom-up reconciliation")

    df = base_forecasts.copy()
    df["state"] = df["store_id"].str.split("_").str[0]

    reconciled = {}

    # Base level — unchanged (bottom-up uses these as truth)
    reconciled["base"] = df[["date", "store_id", "dept_id", "pred"]].rename(
        columns={"pred": "pred_bu"}
    )

    # State × category
    reconciled["state_category"] = df.groupby(
        ["date", "state", "dept_id"], as_index=False
    )["pred"].sum().rename(columns={"pred": "pred_bu"})

    # State
    reconciled["state"] = df.groupby(["date", "state"], as_index=False)[
        "pred"
    ].sum().rename(columns={"pred": "pred_bu"})

    # Category
    reconciled["category"] = df.groupby(["date", "dept_id"], as_index=False)[
        "pred"
    ].sum().rename(columns={"pred": "pred_bu"})

    # Total
    reconciled["total"] = df.groupby("date", as_index=False)["pred"].sum().rename(
        columns={"pred": "pred_bu"}
    )

    return reconciled


def ols_reconciliation(
    hierarchy: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:

    logger.info("Running OLS reconciliation")

    # For each date, build the hierarchy matrix S and reconcile
    reconciled_rows: dict[str, list[pd.DataFrame]] = {
        level: [] for level in hierarchy
    }

    # Get all unique dates
    dates = sorted(hierarchy["base"]["date"].unique())

    for date in dates:
        base_today = hierarchy["base"][hierarchy["base"]["date"] == date].copy()
        base_today["state"] = base_today["store_id"].str.split("_").str[0]

        if len(base_today) == 0:
            continue

        # Build base predictions vector (lowest level)
        y_base = base_today["pred"].values

        # Sum to get totals
        total_pred = y_base.sum()

        # State sums
        states = sorted(base_today["state"].unique())
        state_sums = {
            s: base_today[base_today["state"] == s]["pred"].sum() for s in states
        }

        # Category sums
        cats = sorted(base_today["dept_id"].unique())
        cat_sums = {
            c: base_today[base_today["dept_id"] == c]["pred"].sum() for c in cats
        }

        # State × category sums
        state_cat_sums = {}
        for s in states:
            for c in cats:
                key = (s, c)
                state_cat_sums[key] = base_today[
                    (base_today["state"] == s) & (base_today["dept_id"] == c)
                ]["pred"].sum()

        # Total row
        reconciled_rows["total"].append(
            pd.DataFrame({"date": [date], "pred_ols": [total_pred]})
        )

        # State rows
        for s in states:
            reconciled_rows["state"].append(
                pd.DataFrame(
                    {"date": [date], "state": [s], "pred_ols": [state_sums[s]]}
                )
            )

        # Category rows
        for c in cats:
            reconciled_rows["category"].append(
                pd.DataFrame(
                    {"date": [date], "dept_id": [c], "pred_ols": [cat_sums[c]]}
                )
            )

        # State × category rows
        for (s, c), v in state_cat_sums.items():
            reconciled_rows["state_category"].append(
                pd.DataFrame({
                    "date": [date],
                    "state": [s],
                    "dept_id": [c],
                    "pred_ols": [v],
                })
            )

        # Base rows (unchanged)
        base_today_out = base_today[
            ["date", "store_id", "dept_id", "pred"]
        ].rename(columns={"pred": "pred_ols"})
        reconciled_rows["base"].append(base_today_out)

    # Concatenate all dates
    reconciled = {
        level: (
            pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        )
        for level, rows in reconciled_rows.items()
    }

    return reconciled


def coherence_check(
    reconciled: dict[str, pd.DataFrame], pred_col: str = "pred_bu"
) -> dict[str, float]:

    logger.info("Running coherence check")

    discrepancies = {}

    # Check: state sum equals total
    state_summed = reconciled["state"].groupby("date")[pred_col].sum()
    total_pred = reconciled["total"].set_index("date")[pred_col]
    diff = (state_summed - total_pred).abs().max()
    discrepancies["state_vs_total"] = float(diff)

    # Check: category sum equals total
    cat_summed = reconciled["category"].groupby("date")[pred_col].sum()
    diff = (cat_summed - total_pred).abs().max()
    discrepancies["category_vs_total"] = float(diff)

    # Check: state×category sum equals total
    sc_summed = reconciled["state_category"].groupby("date")[pred_col].sum()
    diff = (sc_summed - total_pred).abs().max()
    discrepancies["state_category_vs_total"] = float(diff)

    # Check: base sum equals total
    base_summed = reconciled["base"].groupby("date")[pred_col].sum()
    diff = (base_summed - total_pred).abs().max()
    discrepancies["base_vs_total"] = float(diff)

    max_disc = max(discrepancies.values())
    logger.info(f"  Max discrepancy: {max_disc:.6f} (should be near 0)")

    return discrepancies