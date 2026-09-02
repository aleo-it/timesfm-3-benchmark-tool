"""Base forecaster protocol for the benchmark."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """Unified interface all model adapters must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for reports."""

    @abstractmethod
    def fit(self, y: pd.Series, freq: str) -> None:
        """Fit the model on historical data.

        Args:
            y: Time series with DatetimeIndex.
            freq: Pandas frequency string ('MS', 'M', 'D', etc.).
        """

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        """Generate point forecast for the given horizon.

        Args:
            horizon: Number of steps to forecast.

        Returns:
            pd.Series with DatetimeIndex matching the forecast period.
        """
