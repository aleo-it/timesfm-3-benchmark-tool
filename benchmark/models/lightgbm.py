"""LightGBM adapter with lag feature engineering."""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from benchmark.models.base import BaseForecaster


class LightGBMForecaster(BaseForecaster):
    """LightGBM with engineered lag features."""

    def __init__(self, lags: list[int] | None = None, n_estimators: int = 200):
        self._lags = lags or [1, 2, 3, 6, 12]
        self._n_estimators = n_estimators
        self._model = None
        self._y = None
        self._freq = None

    @property
    def name(self) -> str:
        return "LightGBM"

    def _create_features(self, y: pd.Series) -> pd.DataFrame:
        """Create lag + calendar features."""
        df = pd.DataFrame({"y": y.values}, index=y.index)

        # Lag features — only use lags that fit the series length
        active_lags = [lag for lag in self._lags if lag < len(y)]
        for lag in active_lags:
            df[f"lag_{lag}"] = df["y"].shift(lag)

        # Calendar features (for monthly data)
        df["month"] = df.index.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Trend feature
        df["trend"] = np.arange(len(df))

        # Drop rows with NaN from lagging
        df = df.dropna()

        self._active_lags = active_lags
        return df

    def fit(self, y: pd.Series, freq: str) -> None:
        self._y = y
        self._freq = freq

        df = self._create_features(y)
        feature_cols = [c for c in df.columns if c != "y"]

        self._model = LGBMRegressor(
            n_estimators=self._n_estimators,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self._model.fit(df[feature_cols], df["y"])
        self._feature_cols = feature_cols

    def predict(self, horizon: int) -> pd.Series:
        """Recursive multi-step prediction."""
        predictions = []
        y_history = self._y.copy()

        for _ in range(horizon):
            # Create features for the last point
            last_idx = y_history.index[-1]
            next_idx = last_idx + pd.tseries.frequencies.to_offset(self._freq)

            # Build feature vector
            features = {}
            for lag in self._active_lags:
                if len(y_history) >= lag:
                    features[f"lag_{lag}"] = y_history.iloc[-lag]
                else:
                    features[f"lag_{lag}"] = 0

            features["month"] = next_idx.month
            features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
            features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
            features["trend"] = len(y_history)

            # Predict one step
            X = pd.DataFrame([features])[self._feature_cols]
            pred = self._model.predict(X)[0]
            predictions.append(pred)

            # Append prediction to history for next step
            y_history = pd.concat([
                y_history,
                pd.Series([pred], index=[next_idx]),
            ])

        # Create forecast dates
        last_date = self._y.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )

        return pd.Series(predictions, index=forecast_dates, name="forecast")
