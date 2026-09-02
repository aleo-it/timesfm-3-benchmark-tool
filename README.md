# Forecasting Lab

Benchmark [TimesFM 3.0](https://github.com/google-research/timesfm) — Google's zero-shot foundation model for time series forecasting — against traditional forecasting methods on standard datasets.

> TimesFM 3.0 is Google Research's foundation model for multivariate forecasting. It's pre-trained on a large corpus of time series and makes predictions **zero-shot**, with no training on your data. See Google's [official announcement](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) for details.

## Install

```bash
pip install -e .
```

## Quick Start

```bash
# Airline Passengers — ~2 min, no download needed
python -m benchmark.run --dataset airline

# M4 Monthly — downloads ~90MB on first run
python -m benchmark.run --dataset m4 --n-series 3

# Retail sales (weekly) — included example dataset
python -m benchmark.run --dataset retail --csv-path data/example_retail.csv

# Retail sales with your own data
python -m benchmark.run --dataset retail --csv-path /path/to/your/sales.csv

# GPU acceleration for TimesFM
python -m benchmark.run --dataset airline --device cuda   # NVIDIA
python -m benchmark.run --dataset airline --device mps    # Apple Silicon

# Run specific models only
python -m benchmark.run --dataset airline --models timesfm3 prophet ets
```

Report saved to `results/<dataset>_<timestamp>.html` (e.g. `results/m4_20260902_143052.html`). Open in a browser.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `airline` | `airline` (144-point demo), `m4` (48k monthly series), or `retail` (weekly sales) |
| `--models` | all 5 | Space-separated: `timesfm3 prophet lightgbm ets naive` |
| `--n-series` | 5 | Number of M4 series to evaluate |
| `--n-folds` | 3 | Rolling-origin CV folds (more = slower, more stable) |
| `--device` | `cpu` | `cpu`, `cuda` (NVIDIA), or `mps` (Apple Silicon) |
| `--output` | auto | Output path. Default: `results/<dataset>_<timestamp>.html` |

## How It Works

### Evaluation Protocol

We use **rolling-origin cross-validation** (expanding window):

```
Series:  [=====================]
Fold 1:  [train=========][----predict----]
Fold 2:  [train===========][----predict----]
Fold 3:  [train=============-][----predict----]
```

1. Start with 60% of the series as the initial training window
2. Forecast 18 months ahead
3. Slide the window forward by 1 month
4. Repeat for N folds
5. Aggregate errors across all folds

This respects temporal ordering — no future data leaks into training.

### Why Subprocess Isolation

Each model runs in a separate process. Not just for cleanliness — it's a hard requirement:

- **torch + LightGBM**: When TimesFM's PyTorch and LightGBM's OpenMP coexist in the same process, segfaults occur on macOS. This is a known library conflict.
- **Fault isolation**: If Prophet's Stan backend hits a convergence issue, other models continue.
- **Memory management**: Foundation models can consume significant memory; subprocesses are reclaimed after each model finishes.

### TimesFM Fairness Controls

Two TimesFM 3.0 features are disabled to keep comparison fair:

- **Symmetric averaging** (`use_symmetric_averaging=False`): Runs inference forward and backward, then averages. Doubles compute — traditional models don't do this.
- **Positivity clamping** (`make_positive=False`): Clamps all forecasts to ≥ 0. Unfair for series that can go negative (financial returns, temperature anomalies).

### Metrics

| Metric  | Range | Interpretation |
|--------|---------|-------|----------------|
| MAE | 0 - ∞ | Average absolute error in original units |
| RMSE | 0 - ∞ | Penalizes large errors more than MAE |
| sMAPE | 0 - 200% | Symmetric percentage error, handles zeros |

We use sMAPE instead of MAPE because MAPE is asymmetric (penalizes under-forecast more) and undefined when actual values are zero.

### Models Evaluated

| Model | Type | Training | Configuration |
|-------|------|----------|---------------|
| TimesFM 3.0 | Foundation | Zero-shot | No per-series fitting, no symmetric averaging, no positivity clamp. Runs on CPU, CUDA (NVIDIA), or MPS (Apple Silicon) |
| Prophet | Additive regression | Trained | Yearly seasonality=auto, no weekly/daily |
| LightGBM | Gradient boosting | Trained | Lags [1,2,3,6,12], sin/cos month, recursive multi-step |
| ETS | Exponential smoothing | Trained | Additive trend + seasonal, period=12, auto params |
| Naive | Baseline | None | Seasonal naive: repeat last 12 months |

### M4 Dataset

The M4 competition dataset contains 48,000 monthly time series across economics, finance, demographics, and industry. The Monthly subset has an 18-month forecast horizon.

We download `Monthly-train.csv` (~90MB) on first run and cache it in `data/`. Series metadata (starting date, domain, frequency) is read from `M4-info.csv`.

### Retail Dataset

The retail dataset (`data/retail.csv`, semicolon-separated) contains weekly unit sales per product:

```
Product;week;month;year;unit
TEST0000T8BMX1;52;12;2025;8.0
```

Each unique product becomes its own time series, reindexed to a continuous weekly (`W-MON`) date range. Missing weeks are filled with 0 (no sales). Forecast horizon is 13 weeks (one quarter).

A small example dataset is included at `data/example_retail.csv` (3 products, 156 weeks each) — useful for testing the pipeline without proprietary data. Use `--csv-path` to point to your own data:

```bash
python -m benchmark.run --dataset retail --csv-path /path/to/your/sales.csv
```

Weekly frequency automatically adjusts model configuration:
- Naive season length: 52 (yearly)
- ETS seasonal periods: 52, stepped down if the series has fewer than 2 full seasonal cycles
- LightGBM lags: [1, 2, 3, 4, 13, 26, 52]

Because week endings are scattered/non-sequential in the raw file, dates are reconstructed from `year` + `week` (ISO week format). Note: many products have 50-75% zero weeks (no sales), which can inflate sMAPE (up to 200% per point when actual is zero) and make MAE-dominated rankings favor models that predict zeros well.

### Report Aggregation

When evaluating multiple series, the report adapts:

- **Summary table**: Average MAE/RMSE/sMAPE per model across all series
- **Metric bar chart**: Averaged bars per model
- **Per-horizon error**: Averaged MAE at each horizon across series
- **Actual vs predicted**: Shows the representative series (median MAE) per model
- **Per-series breakdown**: Expandable table showing individual results per series

The per-series breakdown lets you see if a model wins on average but loses on specific series (which happens — no free lunch in forecasting).

## Report Contents

- **Summary table** — ranked by MAE, winner highlighted
- **Metric bar chart** — MAE / RMSE / sMAPE side by side
- **Per-horizon error** — how forecast error grows with horizon
- **Actual vs predicted** — one subplot per model
- **Radar chart** — normalized model strengths
- **Per-series breakdown** — shown when evaluating multiple series

When multiple series are evaluated, summary metrics are averaged across series. The per-series breakdown table shows individual results.

## Project Structure

```
benchmark/
├── models/          # Model adapters (BaseForecaster protocol)
├── datasets.py      # M4 Monthly + Airline loaders
├── evaluator.py     # Rolling-origin CV + metrics
├── visualize.py     # HTML report with plotly charts
├── run.py           # CLI entry point
└── worker.py        # Subprocess per-model evaluation
```
