from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    aggregate_daily,
    compute_display_window,
    predict_xgboost,
)


class TestPredictXgboost:

    def test_calls_model_predict(self) -> None:
        class FakeModel:
            def predict(self, df: pd.DataFrame) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0])

        df = pd.DataFrame(
            {"feature_a": [1, 2, 3], "feature_b": [4, 5, 6], "extra": [7, 8, 9]}
        )
        result = predict_xgboost(FakeModel(), df, ["feature_a", "feature_b"])
        assert list(result) == [1.0, 2.0, 3.0]

    def test_missing_feature_raises(self) -> None:
        class FakeModel:
            def predict(self, df: pd.DataFrame) -> np.ndarray:
                return np.array([1.0])

        df = pd.DataFrame({"feature_a": [1]})
        with pytest.raises(KeyError, match="Missing feature columns"):
            predict_xgboost(FakeModel(), df, ["feature_a", "feature_missing"])


class TestAggregateDaily:

    def test_sums_within_each_date(self) -> None:
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-01", "2024-01-02"]
                ),
                "sales": [10, 20, 30],
                "xgb_pred": [11, 21, 29],
            }
        )
        result = aggregate_daily(df)
        assert len(result) == 2
        assert result[result["date"] == "2024-01-01"]["actual"].iloc[0] == 30
        assert result[result["date"] == "2024-01-02"]["actual"].iloc[0] == 30
        assert result[result["date"] == "2024-01-01"]["xgb_pred"].iloc[0] == 32

    def test_returns_sorted_by_date(self) -> None:
        """Result should be sorted ascending by date."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-03", "2024-01-01", "2024-01-02"]
                ),
                "sales": [1, 2, 3],
                "xgb_pred": [1, 2, 3],
            }
        )
        result = aggregate_daily(df)
        dates = result["date"].tolist()
        assert dates == sorted(dates)


class TestComputeDisplayWindow:

    def _make_daily(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=60, freq="D"),
                "actual": range(60),
            }
        )

    def test_window_centered_on_split(self) -> None:
        daily = self._make_daily()
        test_start = pd.Timestamp("2024-01-31")
        window = compute_display_window(daily, test_start, horizon_days=20)

        # 10 days before + 10 days after
        before = window[window["date"] < test_start]
        after = window[window["date"] >= test_start]
        assert len(before) == 10
        assert len(after) == 10

    def test_small_horizon(self) -> None:
        daily = self._make_daily()
        test_start = pd.Timestamp("2024-01-31")
        window = compute_display_window(daily, test_start, horizon_days=7)

        # horizon_days // 2 == 3
        before = window[window["date"] < test_start]
        after = window[window["date"] >= test_start]
        assert len(before) == 3
        assert len(after) == 3

    def test_empty_result_when_split_outside_data(self) -> None:
        """If split is far outside data range, window may be empty."""
        daily = self._make_daily()
        test_start = pd.Timestamp("2030-01-01")  # way past data
        window = compute_display_window(daily, test_start, horizon_days=20)
        assert len(window) == 0


class TestNoLeakageInFeatures:

    def test_shift_uses_past_only(self) -> None:
        df = pd.DataFrame(
            {"sales": [10, 20, 30, 40, 50, 60, 70, 80]}
        )
        df["lag_7"] = df["sales"].shift(7)
        # Row 7 (8th row) has lag_7 = row 0 (10)
        assert df["lag_7"].iloc[7] == 10
        # Earlier rows have NaN
        assert pd.isna(df["lag_7"].iloc[6])

    def test_rolling_with_shift_excludes_current(self) -> None:
        df = pd.DataFrame({"sales": [10, 20, 30, 40, 50]})
        # Correct pattern: shift first, then rolling
        df["roll_3"] = df["sales"].shift(1).rolling(3).mean()
        # Row 3: roll_3 = mean(10, 20, 30) = 20
        # The current row's sales (40) is NOT included
        assert df["roll_3"].iloc[3] == 20.0

    def test_rolling_without_shift_leaks(self) -> None:
        df = pd.DataFrame({"sales": [10, 20, 30, 40, 50]})
        # Wrong pattern: includes current value
        df["roll_3_bad"] = df["sales"].rolling(3).mean()
        # Row 3: includes 20, 30, 40 — but 40 IS the current value's sales
        # → mean = 30 (uses current row in its own feature)
        assert df["roll_3_bad"].iloc[3] == 30.0
        # This is leakage — production code must use shift first
