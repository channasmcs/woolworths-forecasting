

import pickle
import time
import warnings

import pandas as pd

from src.backtesting import walk_forward_backtest
from src.data_loader import load_and_clean
from src.features import FEATURE_COLS, chronological_split, engineer_features
from src.models import train_arima, train_xgboost
from src.reconciliation import (
    bottom_up_reconciliation,
    build_hierarchy,
    coherence_check,
)
from src.shap_analysis import compute_shap

warnings.filterwarnings("ignore")


def main():
    start = time.time()

    print("=" * 70)
    print("  WOOLWORTHS NZ — DEMAND FORECASTING PIPELINE")
    print("=" * 70)

    print("\n[1/7] Loading + cleaning data...")
    df = load_and_clean(
        raw_dir="data/raw", save_to="data/processed/sales_clean.parquet"
    )
    print(f"      ✅ {len(df):,} rows saved")

    print("\n[2/7] Engineering features...")
    df = engineer_features(df, save_to="data/processed/features.parquet")
    print(f"      ✅ Features ready ({df.shape[1]} columns)")

    print("\n[3/7] Chronological train/val/test split...")
    train, val, test = chronological_split(df)
    print(
        f"      ✅ Train: {len(train):,}  "
        f"Val: {len(val):,}  Test: {len(test):,}"
    )

    print("\n[4/7] Training ARIMA baseline with auto-order selection...")
    arima_metrics = train_arima(
        train,
        test,
        sample_size=50,
        auto_order=True,
        save_to="data/models/arima_metrics.pkl",
    )
    print(
        f'      ✅ ARIMA MAPE: {arima_metrics["mape"]:.2f}%  '
        f'RMSE: {arima_metrics["rmse"]:.2f}'
    )
    print(f'      Mode order: {arima_metrics["order"]}')

    print("\n[5/7] Training XGBoost + quantile models (5-10 minutes)...")
    xgb_model, xgb_metrics = train_xgboost(
        train,
        val,
        test,
        train_quantiles=True,
        save_to="data/models/xgb_model.pkl",
    )
    print(
        f'      ✅ XGBoost MAPE: {xgb_metrics["mape"]:.2f}%  '
        f'RMSE: {xgb_metrics["rmse"]:.2f}'
    )
    if "ci_coverage_80" in xgb_metrics:
        print(
            f'      80% CI coverage: {xgb_metrics["ci_coverage_80"]:.1f}% '
            "(target ~80%)"
        )

    print("\n[6/7] Walk-forward backtest (5 historical windows)...")
    backtest_results = walk_forward_backtest(
        df,
        n_splits=5,
        test_size_days=28,
        save_to="data/models/backtest_results.pkl",
    )
    print(
        f'      ✅ Mean MAPE: {backtest_results["mean_mape"]:.2f}% '
        f'± {backtest_results["std_mape"]:.2f}%'
    )
    print(
        "      Stability: "
        f'{"✅ Stable" if backtest_results["stable"] else "⚠️ Variable"}'
    )

    print("\n[7/7] Hierarchical reconciliation + SHAP...")

    pred_df = pd.DataFrame({
        "date": test["date"].values,
        "store_id": test["store_id"].values,
        "dept_id": test["dept_id"].values,
        "actual": test["sales"].values.astype(float),
        "pred": xgb_model.predict(test[FEATURE_COLS]).astype(float),
    })

    hierarchy = build_hierarchy(pred_df)
    reconciled = bottom_up_reconciliation(pred_df)
    discrepancies = coherence_check(reconciled, pred_col="pred_bu")

    with open("data/models/reconciliation.pkl", "wb") as f:
        pickle.dump({
            "hierarchy": hierarchy,
            "reconciled": reconciled,
            "discrepancies": discrepancies,
        }, f)

    max_disc = max(discrepancies.values())
    print(
        f"      ✅ Reconciliation complete · "
        f"max discrepancy: {max_disc:.6f} units"
    )

    shap_summary = compute_shap(
        xgb_model, test, save_to="data/models/shap_values.pkl"
    )
    print(f'      ✅ Top SHAP feature: {shap_summary["top_features"][0]}')

    elapsed = time.time() - start

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print("\n  Model       MAPE       RMSE")
    print("  ----------  ---------  --------")
    print(
        f'  ARIMA       {arima_metrics["mape"]:>6.2f}%    '
        f'{arima_metrics["rmse"]:>6.2f}'
    )
    print(
        f'  XGBoost     {xgb_metrics["mape"]:>6.2f}%    '
        f'{xgb_metrics["rmse"]:>6.2f}'
    )

    if arima_metrics["mape"] > 0:
        improvement = (
            (arima_metrics["mape"] - xgb_metrics["mape"]) / arima_metrics["mape"]
        ) * 100
        print(f"\n  Improvement: {improvement:.1f}% reduction in MAPE")

    print("\n  Backtest Stability (5 windows):")
    print(
        f'    Mean MAPE: {backtest_results["mean_mape"]:.2f}% '
        f'± {backtest_results["std_mape"]:.2f}%'
    )

    if "ci_coverage_80" in xgb_metrics:
        print("\n  Calibration:")
        print(
            f'    80% CI coverage: {xgb_metrics["ci_coverage_80"]:.1f}% '
            "(target ~80%)"
        )

    print("\n  Hierarchical Coherence:")
    print(f"    Max discrepancy: {max_disc:.6f} units (target: near 0)")

    print("\n  Top 5 SHAP features:")
    for i, (name, val) in enumerate(
        zip(
            shap_summary["top_features"][:5],
            shap_summary["top_values"][:5],
            strict=True,
        ),
        1,
    ):
        print(f"    {i}. {name:<20}  {val:.3f}")

    print(f"\n  Total pipeline time: {elapsed / 60:.1f} minutes")
    print("\n  Launch the dashboard:")
    print("    streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()