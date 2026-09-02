"""Naive forecasting baselines."""

import pandas as pd

from benchmark.models.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Seasonal naive baseline (last seasonal value)."""

    def __init__(self, season_length: int = 12):
        self._season_length = season_length
        self._y = None
        self._freq = None

    @property
    def name(self) -> str:
        return f"Naive (s={self._season_length})"

    def fit(self, y: pd.Series, freq: str) -> None:
        self._y = y
        self._freq = freq

    def predict(self, horizon: int) -> pd.Series:
        """Repeat last seasonal cycle."""
        last_season = self._y.iloc[-self._season_length:]

        # Tile to fill horizon
        reps = (horizon // self._season_length) + 1
        forecast_values = list(last_season.values) * reps
        forecast_values = forecast_values[:horizon]

        # Create dates
        last_date = self._y.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )

        return pd.Series(forecast_values, index=forecast_dates, name="forecast")
