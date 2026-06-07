# Primary palette
NAVY = "#0F172A"          # Main background
NAVY_2 = "#1E293B"        # Card backgrounds
NAVY_3 = "#334155"        # Borders, muted backgrounds
ORANGE = "#F97316"        # Primary accent (XGBoost, highlights)
WHITE = "#FFFFFF"

# Slate scale
SLATE_200 = "#E2E8F0"
SLATE_300 = "#CBD5E1"     # Body text on dark backgrounds
SLATE_400 = "#94A3B8"     # Secondary muted text
SLATE_500 = "#64748B"     # Captions

# Status colors
GREEN = "#10B981"         # Positive / success / accurate forecast
RED = "#EF4444"           # Warning / problem / stockout
BLUE = "#60A5FA"          # Information / actual sales line
YELLOW = "#FBBF24"        # Test period highlight

# Chart colors
CHART_ACTUAL = BLUE           # Actual sales line
CHART_XGBOOST = ORANGE        # XGBoost forecast
CHART_ARIMA = "#0B4300"       # ARIMA forecast (dark green dashed)
CHART_CI_FILL = "rgba(234, 88, 12, 0.15)"  # 80% confidence band
CHART_FONT_COLOR = "#0F172A"
CHART_FONT_SIZE = 12

# Backtest stability assessment
BACKTEST_STABLE_STD = 2.0     # Std dev below this = STABLE label

# Dataset Config
STORES = ["CA_1", "TX_1", "WI_1"]
DEPARTMENTS = ["FOODS_1", "FOODS_2", "FOODS_3"]

DEPT_LABELS = {
    "FOODS_1": "FOODS_1 (Beverages)",
    "FOODS_2": "FOODS_2 (Grocery)",
    "FOODS_3": "FOODS_3 (Fresh)",
}

FEATURE_COLS_SET = [
    "lag_7", "lag_14", "lag_28",
    "rolling_7_mean", "rolling_28_mean", "rolling_7_std",
    "day_of_week", "wday",
    "snap_flag", "sell_price"
]

#Dashboard config
DASHBOARD_TITLE = "📦 Woolworths NZ — Demand Forecasting Dashboard"
DASHBOARD_SUBTITLE = "M5 Foods Department · ARIMA vs XGBoost · SHAP + Backtest"

PANEL_1_TITLE = "Panel 1 · Forecast vs Actual"
PANEL_2_TITLE = "Panel 2 · MAPE Comparison"
PANEL_3_TITLE = "Panel 3 · SHAP Feature Importance"
PANEL_4_TITLE = "Panel 4 · Backtest Stability"