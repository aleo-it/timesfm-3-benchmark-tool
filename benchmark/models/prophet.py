"""Facebook Prophet adapter."""

import pandas as pd
from prophet import Prophet

from benchmark.models.base import BaseForecaster


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet - additive regression model."""

    def __init__(self):
        self._model = None
        self._freq = None

    @property
    def name(self) -> str:
        return "Prophet"

    def fit(self, y: pd.Series, freq: str) -> None:
        self._freq = freq
        # Prophet requires ds/y DataFrame
        df = pd.DataFrame({
            "ds": y.index,
            "y": y.values,
        })
        self._model = Prophet(
            yearly_seasonality="auto",
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        self._model.fit(df)

    def predict(self, horizon: int) -> pd.Series:
        # Create future dataframe
        future = self._model.make_future_dataframe(
            periods=horizon,
            freq=self._freq,
        )
        forecast = self._model.predict(future)

        # Extract only the forecast period (last 'horizon' rows)
        forecast = forecast.tail(horizon)

        return pd.Series(
            forecast["yhat"].values,
            index=pd.DatetimeIndex(forecast["ds"]),
            name="forecast",
        )
