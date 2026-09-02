"""Main benchmark runner script."""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from benchmark.datasets import load_m4_monthly, load_airline, load_retail
from benchmark.models import list_models
from benchmark.visualize import generate_report

WORKER = Path(__file__).resolve().parent / "worker.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forecasting Benchmark: TimesFM 3.0 vs Traditional Models",
    )
    parser.add_argument(
        "--dataset",
        choices=["m4", "airline", "retail"],
        default="airline",
        help="Dataset to benchmark on: airline (quick demo), m4 (48k monthly series), or retail (weekly sales)",
    )
    parser.add_argument(
        "--n-series",
        type=int,
        default=5,
        help="Number of M4 series to evaluate (default: 5)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to retail CSV (default: data/retail.csv). Only used with --dataset retail",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list_models(),
        default=list_models(),
        help="Models to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for HTML report (default: results/<dataset>_<timestamp>.html)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device for TimesFM: cpu, cuda (NVIDIA), or mps (Apple Silicon)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run models in parallel (default: sequential). "
        "Note: parallel may increase memory pressure for TimesFM.",
    )
    return parser.parse_args()


def run_model_worker(model_key: str, dataset_file: str, n_folds: int, device: str) -> object:
    """Run a single model in a subprocess and return its EvalResult.

    Each model runs in isolation so a native library crash (e.g. the known
    torch+LightGBM OpenMP conflict) cannot take down other models.
    """
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        out_file = tmp.name

    cmd = [
        sys.executable,
        str(WORKER),
        dataset_file,
        out_file,
        str(n_folds),
        device,
        model_key,
    ]
    env = dict(os.environ)
    project_root = WORKER.parent.parent
    env["PYTHONPATH"] = str(project_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker for '{model_key}' failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()[-500:]}"
        )

    with open(out_file, "rb") as f:
        result = pickle.load(f)
    Path(out_file).unlink(missing_ok=True)
    return result


def main():
    args = parse_args()

    print("=" * 60)
    print("Forecasting Benchmark")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print()

    # Load dataset
    print("Loading dataset...")
    if args.dataset == "airline":
        datasets = [load_airline()]
    elif args.dataset == "retail":
        datasets = load_retail(n_series=args.n_series, csv_path=args.csv_path or "data/retail.csv")
    else:
        datasets = load_m4_monthly(n_series=args.n_series)

    print(f"Loaded {len(datasets)} series")
    print()

    all_results = []
    work_items = [
        (model_key, dataset) for dataset in datasets for model_key in args.models
    ]

    if args.parallel:
        print(f"Running {len(work_items)} evaluations in parallel...")
        with ThreadPoolExecutor(max_workers=min(4, len(work_items))) as pool:
            futures = {}
            for model_key, dataset in work_items:
                future = pool.submit(
                    lambda mk=model_key, ds=dataset: _run_item(mk, ds, args)
                )
                futures[future] = (model_key, dataset.series_id)

            for future in as_completed(futures):
                model_key, series_id = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    print(f"  {model_key} on {series_id}: MAE={result.mae:.2f}")
                except Exception as e:
                    print(f"  {model_key} on {series_id}: FAILED - {e}")
    else:
        for model_key, dataset in work_items:
            print(f"  Running {model_key} on {dataset.series_id}...", end=" ", flush=True)
            try:
                result = _run_item(model_key, dataset, args)
                all_results.append(result)
                print(f"MAE={result.mae:.2f}, sMAPE={result.smape:.1f}%")
            except Exception as e:
                print(f"FAILED: {e}")

    if not all_results:
        print("\nError: No models completed successfully!")
        sys.exit(1)

    # Generate report
    print("\n" + "=" * 60)
    print("Generating report...")

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/{args.dataset}_{timestamp}.html"

    report_path = generate_report(
        results=all_results,
        dataset_name=args.dataset.upper(),
        output_path=args.output,
    )

    print(f"Report saved to: {report_path}")
    print("\nOpen in browser to view interactive charts!")
    return 0


def _run_item(model_key: str, dataset, args) -> object:
    """Serialize dataset and run the model worker subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        dataset_file = tmp.name
    with open(dataset_file, "wb") as f:
        pickle.dump(dataset, f)

    try:
        return run_model_worker(
            model_key=model_key,
            dataset_file=dataset_file,
            n_folds=args.n_folds,
            device=args.device,
        )
    finally:
        Path(dataset_file).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
