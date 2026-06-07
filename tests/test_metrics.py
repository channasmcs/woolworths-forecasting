from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import daily_mape, improvement_pct, mape, rmse


class TestMape:

    def test_perfect_prediction_gives_zero(self) -> None:
        assert mape([100, 200, 300], [100, 200, 300]) == 0.0

    def test_known_simple_case(self) -> None:
        # actual=100, predicted=110 → error 10%
        # actual=200, predicted=180 → error 10%
        assert mape([100, 200], [110, 180]) == pytest.approx(10.0)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            mape([1, 2, 3], [1, 2])

    def test_handles_zero_actuals(self) -> None:
        # Only the second row counts: |100-110|/100 = 10%
        result = mape([0, 100], [50, 110])
        assert result == pytest.approx(10.0)

    def test_all_zeros_returns_nan(self) -> None:
        assert np.isnan(mape([0, 0, 0], [10, 20, 30]))

    def test_accepts_pandas_series(self) -> None:
        actual = pd.Series([100, 200])
        predicted = pd.Series([110, 180])
        assert mape(actual, predicted) == pytest.approx(10.0)

    def test_accepts_numpy_arrays(self) -> None:
        actual = np.array([100, 200])
        predicted = np.array([110, 180])
        assert mape(actual, predicted) == pytest.approx(10.0)


class TestRmse:

    def test_perfect_prediction_gives_zero(self) -> None:
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_simple_case(self) -> None:
        # errors: 1, -1 → squared: 1, 1 → mean: 1 → sqrt: 1
        assert rmse([10, 20], [11, 19]) == pytest.approx(1.0)

    def test_penalises_large_errors_more_than_mape(self) -> None:
        # One big error of 100, nine small errors of 1
        actual = [100] * 10
        predicted = [100] * 9 + [0]  # last one off by 100
        # squared errors: nine 0s and one 10000 → mean 1000 → sqrt ~31.6
        assert rmse(actual, predicted) == pytest.approx(np.sqrt(1000))

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            rmse([1, 2, 3], [1, 2])


class TestImprovementPct:

    def test_positive_improvement(self) -> None:
        # ARIMA=10, XGBoost=8 → 20% improvement
        assert improvement_pct(10.0, 8.0) == pytest.approx(20.0)

    def test_no_improvement_equal_mape(self) -> None:
        assert improvement_pct(10.0, 10.0) == 0.0

    def test_negative_improvement_when_worse(self) -> None:
        # ARIMA=10, XGBoost=12 → -20%
        assert improvement_pct(10.0, 12.0) == pytest.approx(-20.0)

    def test_returns_zero_on_nan_baseline(self) -> None:
        assert improvement_pct(float("nan"), 10.0) == 0.0

    def test_returns_zero_on_nan_model(self) -> None:
        assert improvement_pct(10.0, float("nan")) == 0.0

    def test_returns_zero_on_zero_baseline(self) -> None:
        assert improvement_pct(0.0, 5.0) == 0.0


class TestDailyMape:

    def test_computes_on_dataframe(self) -> None:
        df = pd.DataFrame(
            {"actual": [100, 200], "pred": [110, 180]}
        )
        assert daily_mape(df, "actual", "pred") == pytest.approx(10.0)

    def test_missing_actual_column_raises(self) -> None:
        df = pd.DataFrame({"pred": [1, 2, 3]})
        with pytest.raises(KeyError, match="actual"):
            daily_mape(df, "actual", "pred")

    def test_missing_pred_column_raises(self) -> None:
        df = pd.DataFrame({"actual": [1, 2, 3]})
        with pytest.raises(KeyError, match="pred"):
            daily_mape(df, "actual", "pred")
