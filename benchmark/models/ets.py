"""Exponential Smoothing (Holt-Winters) adapter."""

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from benchmark.models.base import BaseForecaster


class ETSForecaster(BaseForecaster):
    """Holt-Winters Exponential Smoothing."""

    def __init__(self, trend: str = "add", seasonal: str = "add", seasonal_periods: int = 12):
        self._trend = trend
        self._seasonal = seasonal
        self._seasonal_periods = seasonal_periods
        self._model = None
        self._freq = None

    @property
    def name(self) -> str:
        return "ETS (Holt-Winters)"

    def fit(self, y: pd.Series, freq: str) -> None:
        self._freq = freq

        # Seasonal model needs >= 2 full seasonal cycles of data. If the
        # training window is too short, step down to the largest period that
        # fits (e.g. 52 -> 13) or drop seasonality entirely.
        periods = self._seasonal_periods
        while periods > 1 and len(y) < 2 * periods:
            periods = periods // 2
        periods = max(periods, 2)
        use_seasonal = periods <= len(y) // 2

        try:
            kwargs = dict(
                trend=self._trend,
                initialization_method="estimated",
            )
            if use_seasonal:
                kwargs.update(seasonal=self._seasonal, seasonal_periods=periods)
            self._model = ExponentialSmoothing(y, **kwargs).fit(optimized=True)
        except (ValueError, TypeError):
            # "estimated" initialization can fail on series with zeros/constant
            # stretches (e.g. retail with zero-sales weeks). Fall back to heuristic.
            kwargs["initialization_method"] = "heuristic"
            self._model = ExponentialSmoothing(y, **kwargs).fit(optimized=True)

    def predict(self, horizon: int) -> pd.Series:
        forecast = self._model.forecast(steps=horizon)
        forecast.index.name = None
        return pd.Series(forecast.values, index=forecast.index, name="forecast")
