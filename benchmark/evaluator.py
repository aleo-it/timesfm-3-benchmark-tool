"""Rolling-origin cross-validation evaluator."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from benchmark.datasets import Dataset
from benchmark.models.base import BaseForecaster


@dataclass
class FoldResult:
    """Results from a single CV fold."""

    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    y_true: pd.Series
    y_pred: pd.Series
    model_name: str


@dataclass
class EvalResult:
    """Aggregated evaluation results."""

    model_name: str
    series_id: str
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    smape: float  # Symmetric MAPE (0-200 scale)
    per_horizon_mae: dict[int, float]  # horizon -> MAE
    per_horizon_rmse: dict[int, float]  # horizon -> RMSE
    folds: list[FoldResult]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0-200 scale, M4 convention)."""
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero
    mask = denominator > 0
    if not mask.any():
        return 0.0
    return float(
        200 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
    )


def rolling_origin_cv(
    forecaster: BaseForecaster,
    dataset: Dataset,
    initial_window: int | None = None,
    step: int = 1,
    n_folds: int = 5,
) -> EvalResult:
    """Evaluate a forecaster using rolling-origin cross-validation.

    Args:
        forecaster: Model to evaluate.
        dataset: Dataset to evaluate on.
        initial_window: Initial training window size. If None, uses 60% of data.
        step: Steps to slide between folds.
        n_folds: Maximum number of CV folds.

    Returns:
        EvalResult with aggregated metrics and per-fold details.
    """
    y = dataset.y
    horizon = dataset.horizon

    if initial_window is None:
        initial_window = int(len(y) * 0.6)

    folds: list[FoldResult] = []
    per_horizon_errors: dict[int, list[float]] = {
        h: [] for h in range(1, horizon + 1)
    }

    for fold in range(n_folds):
        # Calculate train/test split
        train_end_idx = initial_window + fold * step
        test_end_idx = train_end_idx + horizon

        if test_end_idx > len(y):
            break

        # Split data
        y_train = y.iloc[:train_end_idx]
        y_test = y.iloc[train_end_idx:test_end_idx]

        # Fit and predict
        try:
            forecaster.fit(y_train, freq=dataset.freq)
            y_pred = forecaster.predict(horizon=len(y_test))

            # Align predictions with test index
            y_pred = y_pred.reindex(y_test.index)

            # Store fold result
            fold_result = FoldResult(
                fold=fold,
                train_end=y_train.index[-1],
                test_start=y_test.index[0],
                y_true=y_test,
                y_pred=y_pred,
                model_name=forecaster.name,
            )
            folds.append(fold_result)

            # Collect per-horizon errors
            for h in range(1, horizon + 1):
                if h <= len(y_test):
                    true_val = y_test.iloc[h - 1]
                    pred_val = y_pred.iloc[h - 1]
                    per_horizon_errors[h].append(abs(true_val - pred_val))

        except Exception as e:
            print(f"Warning: Fold {fold} failed for {forecaster.name}: {e}")
            continue

    if not folds:
        raise ValueError(
            f"All folds failed for {forecaster.name} on {dataset.series_id}"
        )

    # Aggregate metrics across all folds
    all_true = np.concatenate([f.y_true.values for f in folds])
    all_pred = np.concatenate([f.y_pred.values for f in folds])

    return EvalResult(
        model_name=forecaster.name,
        series_id=dataset.series_id,
        mae=mae(all_true, all_pred),
        rmse=rmse(all_true, all_pred),
        smape=smape(all_true, all_pred),
        per_horizon_mae={
            h: float(np.mean(v)) for h, v in per_horizon_errors.items() if v
        },
        per_horizon_rmse={
            h: float(np.sqrt(np.mean(np.array(v) ** 2)))
            for h, v in per_horizon_errors.items()
            if v
        },
        folds=folds,
    )
