"""TimesFM 3.0 adapter using timesfm3.TimesFM3Evaluator."""

import numpy as np
import pandas as pd
from timesfm3 import TimesFM3Evaluator, ModelConfig

from benchmark.models.base import BaseForecaster


class TimesFM3Forecaster(BaseForecaster):
    """Google Research TimesFM 3.0 foundation model."""

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._forecaster = None
        self._y = None
        self._freq = None

    @property
    def name(self) -> str:
        return "TimesFM 3.0"

    def fit(self, y: pd.Series, freq: str) -> None:
        self._y = y
        self._freq = freq
        # Initialize forecaster on first fit
        if self._forecaster is None:
            config = ModelConfig(
                checkpoint_path="google/timesfm-3.0-pytorch",
                per_core_batch_size=32,
                device=self._device,
            )
            self._forecaster = TimesFM3Evaluator(config)

    def predict(self, horizon: int) -> pd.Series:
        # Convert to numpy float32
        ts = self._y.values.astype(np.float32)

        # Get forecast (predict_batch returns a generator)
        outputs = list(self._forecaster.predict_batch(
            [ts],
            horizon=horizon,
            return_quantiles=False,
            use_symmetric_averaging=False,  # Disable for fair comparison
            make_positive=False,  # Don't clamp - allows negative values
        ))

        # Extract point forecast
        forecast = outputs[0].forecast.flatten()

        # Create date range for forecast
        last_date = self._y.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )

        return pd.Series(forecast, index=forecast_dates, name="forecast")
