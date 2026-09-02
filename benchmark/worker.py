"""Worker entrypoint for evaluating a single model in an isolated subprocess.

Each model is evaluated in its own process to avoid native library conflicts
(e.g. torch/OpenMP segfaults when LightGBM runs after TimesFM in the same
process). Results are serialized to a pickle file and read back by run.py.
"""

import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from benchmark.models import get_model_class


def evaluate_one(model_key: str, dataset_file: str, n_folds: int, device: str, out_file: str) -> None:
    """Evaluate a single model on a pickled dataset, write result pickle."""
    from benchmark.datasets import Dataset

    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset).__name__}")

    # Validate device — only relevant for TimesFM (torch-based); skip torch import
    # for other models to avoid torch/OpenMP + LightGBM segfaults.
    if model_key == "timesfm3" and device in ("cuda", "mps"):
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                print(f"WARNING: CUDA unavailable, falling back to cpu", file=sys.stderr)
                device = "cpu"
            elif device == "mps" and not torch.backends.mps.is_available():
                print(f"WARNING: MPS unavailable, falling back to cpu", file=sys.stderr)
                device = "cpu"
        except ImportError:
            device = "cpu"

    from benchmark.evaluator import rolling_origin_cv

    season = 52 if dataset.freq.startswith("W") else 12
    lags = [1, 2, 3, 4, 13, 26, 52] if dataset.freq.startswith("W") else [1, 2, 3, 6, 12]

    model_cls = get_model_class(model_key)
    if model_key == "timesfm3":
        model = model_cls(device=device)
    elif model_key == "naive":
        model = model_cls(season_length=season)
    elif model_key == "ets":
        model = model_cls(seasonal_periods=season)
    elif model_key == "lightgbm":
        model = model_cls(lags=lags)
    else:
        model = model_cls()

    result = rolling_origin_cv(forecaster=model, dataset=dataset, n_folds=n_folds)

    with open(out_file, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    # argv: <dataset_file> <out_file> <n_folds> <device> <model_key>
    _dataset_file = sys.argv[1]
    _out_file = sys.argv[2]
    _n_folds = int(sys.argv[3])
    _device = sys.argv[4]
    _model_key = sys.argv[5]

    _root = Path(__file__).resolve().parent.parent

    evaluate_one(
        model_key=_model_key,
        dataset_file=_dataset_file,
        n_folds=_n_folds,
        device=_device,
        out_file=_out_file,
    )
