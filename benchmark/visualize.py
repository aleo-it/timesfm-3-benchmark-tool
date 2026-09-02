"""HTML report generator for forecasting benchmark results."""

from pathlib import Path
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from jinja2 import Template

from benchmark.evaluator import EvalResult


COLORS = {
    "TimesFM 3.0": "#FF6B6B",
    "Prophet": "#4ECDC4",
    "LightGBM": "#45B7D1",
    "ETS (Holt-Winters)": "#96CEB4",
    "Naive (s=12)": "#FFEAA7",
}

RESULTS_PER_MODEL = "results_per_model"
RESULTS_AGGREGATED = "results_aggregated"


def _aggregate_by_model(results: list[EvalResult]) -> tuple[list[dict], list[EvalResult]]:
    """Group results by model, return aggregated metrics and one representative per model."""
    by_model = defaultdict(list)
    for r in results:
        by_model[r.model_name].append(r)

    aggregated = []
    representatives = []

    for model_name, model_results in by_model.items():
        avg_mae = np.mean([r.mae for r in model_results])
        avg_rmse = np.mean([r.rmse for r in model_results])
        avg_smape = np.mean([r.smape for r in model_results])

        # Merge per_horizon_mae across series (average per horizon)
        all_horizons = defaultdict(list)
        for r in model_results:
            for h, v in r.per_horizon_mae.items():
                all_horizons[h].append(v)
        avg_per_horizon = {h: np.mean(vals) for h, vals in sorted(all_horizons.items())}

        aggregated.append({
            "model_name": model_name,
            "mae": avg_mae,
            "rmse": avg_rmse,
            "smape": avg_smape,
            "per_horizon_mae": avg_per_horizon,
            "n_series": len(model_results),
        })

        # Pick the series with median MAE as representative for actual-vs-predicted
        sorted_by_mae = sorted(model_results, key=lambda r: r.mae)
        median_idx = len(sorted_by_mae) // 2
        representatives.append(sorted_by_mae[median_idx])

    return aggregated, representatives


def _chart_mode(results: list[EvalResult]) -> str:
    """Determine whether to show per-series or per-model aggregated charts."""
    unique_series = len({r.series_id for r in results})
    return RESULTS_AGGREGATED if unique_series > 1 else RESULTS_PER_MODEL


def create_metric_comparison_chart(results: list[EvalResult]) -> str:
    """Bar chart comparing MAE/RMSE/sMAPE across models."""
    mode = _chart_mode(results)

    if mode == RESULTS_AGGREGATED:
        agg, _ = _aggregate_by_model(results)
        agg.sort(key=lambda x: x["mae"])
        models = [a["model_name"] for a in agg]
        mae_vals = [a["mae"] for a in agg]
        rmse_vals = [a["rmse"] for a in agg]
        smape_vals = [a["smape"] for a in agg]
    else:
        results_sorted = sorted(results, key=lambda r: r.mae)
        models = [r.model_name for r in results_sorted]
        mae_vals = [r.mae for r in results_sorted]
        rmse_vals = [r.rmse for r in results_sorted]
        smape_vals = [r.smape for r in results_sorted]

    colors = [COLORS.get(m, "#888888") for m in models]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE (lower is better)", "RMSE (lower is better)", "sMAPE % (lower is better)"),
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Bar(x=models, y=mae_vals, marker_color=colors, name="MAE", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=models, y=rmse_vals, marker_color=colors, name="RMSE", showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=models, y=smape_vals, marker_color=colors, name="sMAPE", showlegend=False), row=1, col=3)

    suffix = " — averaged across series" if mode == RESULTS_AGGREGATED else ""
    fig.update_layout(height=400, template="plotly_white", title_text=f"Model Comparison{suffix}", title_font_size=20)

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_per_horizon_chart(results: list[EvalResult]) -> str:
    """Line chart showing MAE evolution across forecast horizons."""
    mode = _chart_mode(results)
    fig = go.Figure()

    if mode == RESULTS_AGGREGATED:
        agg, _ = _aggregate_by_model(results)
        for a in agg:
            horizons = sorted(a["per_horizon_mae"].keys())
            mae_vals = [a["per_horizon_mae"][h] for h in horizons]
            fig.add_trace(go.Scatter(
                x=horizons, y=mae_vals, mode="lines+markers",
                name=a["model_name"],
                line=dict(color=COLORS.get(a["model_name"], "#888888"), width=2),
                marker=dict(size=5),
            ))
    else:
        for r in results:
            horizons = sorted(r.per_horizon_mae.keys())
            mae_vals = [r.per_horizon_mae[h] for h in horizons]
            fig.add_trace(go.Scatter(
                x=horizons, y=mae_vals, mode="lines+markers",
                name=r.model_name,
                line=dict(color=COLORS.get(r.model_name, "#888888"), width=2),
                marker=dict(size=5),
            ))

    fig.update_layout(
        title="Error by Forecast Horizon (MAE)",
        xaxis_title="Forecast Horizon (months ahead)",
        yaxis_title="MAE",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_actual_vs_predicted_chart(results: list[EvalResult], dataset_name: str) -> str:
    """One subplot per model showing actual vs predicted (representative series)."""
    mode = _chart_mode(results)

    if mode == RESULTS_AGGREGATED:
        _, reps = _aggregate_by_model(results)
        reps.sort(key=lambda r: r.model_name)
    else:
        reps = sorted(results, key=lambda r: r.model_name)

    n_models = len(reps)
    spacing = min(0.04, 1 / max(n_models, 2) - 0.01)
    fig = make_subplots(
        rows=1, cols=n_models,
        subplot_titles=[r.model_name for r in reps],
        horizontal_spacing=spacing,
    )

    for i, result in enumerate(reps):
        fold = result.folds[-1]

        fig.add_trace(
            go.Scatter(
                x=fold.y_true.index, y=fold.y_true.values,
                mode="lines", name="Actual",
                line=dict(color="#333333", width=2),
                showlegend=(i == 0),
            ),
            row=1, col=i + 1,
        )

        fig.add_trace(
            go.Scatter(
                x=fold.y_pred.index, y=fold.y_pred.values,
                mode="lines+markers", name="Predicted",
                line=dict(color=COLORS.get(result.model_name, "#888888"), width=2, dash="dash"),
                marker=dict(size=4),
                showlegend=(i == 0),
            ),
            row=1, col=i + 1,
        )

    suffix = " — representative series" if mode == RESULTS_AGGREGATED else ""
    fig.update_layout(
        height=350, template="plotly_white",
        title_text=f"Actual vs Predicted — {dataset_name}{suffix}",
        title_font_size=18,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_radar_chart(results: list[EvalResult]) -> str:
    """Radar chart showing model strengths (higher = better)."""
    mode = _chart_mode(results)
    categories = ["MAE", "RMSE", "sMAPE"]

    if mode == RESULTS_AGGREGATED:
        agg, _ = _aggregate_by_model(results)
        entries = agg
        max_vals = {
            "mae": max(a["mae"] for a in agg) or 1,
            "rmse": max(a["rmse"] for a in agg) or 1,
            "smape": max(a["smape"] for a in agg) or 1,
        }
    else:
        entries = results
        max_vals = {
            "mae": max(r.mae for r in results) or 1,
            "rmse": max(r.rmse for r in results) or 1,
            "smape": max(r.smape for r in results) or 1,
        }

    fig = go.Figure()

    for entry in entries:
        mae_val = entry["mae"] if isinstance(entry, dict) else entry.mae
        rmse_val = entry["rmse"] if isinstance(entry, dict) else entry.rmse
        smape_val = entry["smape"] if isinstance(entry, dict) else entry.smape
        name = entry["model_name"] if isinstance(entry, dict) else entry.model_name

        values = [
            1 - (mae_val / max_vals["mae"]),
            1 - (rmse_val / max_vals["rmse"]),
            1 - (smape_val / max_vals["smape"]),
        ]
        values.append(values[0])

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=name,
            line_color=COLORS.get(name, "#888888"),
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Strengths (normalized, higher is better)",
        template="plotly_white",
        height=450,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_series_detail_table(results: list[EvalResult]) -> str:
    """Per-series breakdown table — only shown when multiple series exist."""
    mode = _chart_mode(results)
    if mode == RESULTS_PER_MODEL:
        return ""

    by_series = defaultdict(list)
    for r in results:
        by_series[r.series_id].append(r)

    rows_html = ""
    for series_id in sorted(by_series.keys()):
        series_results = sorted(by_series[series_id], key=lambda r: r.mae)
        rows_html += f'<tr><td colspan="5" style="background:#f0f0f0;font-weight:600;">{series_id}</td></tr>\n'
        for r in series_results:
            winner_cls = "winner" if r is series_results[0] else ""
            rows_html += (
                f'<tr class="{winner_cls}">'
                f'<td></td><td>{r.model_name}</td>'
                f'<td>{r.mae:.2f}</td><td>{r.rmse:.2f}</td><td>{r.smape:.1f}%</td></tr>\n'
            )

    return f"""
    <div class="section">
        <h2>Per-Series Breakdown</h2>
        <table>
            <thead>
                <tr><th></th><th>Model</th><th>MAE</th><th>RMSE</th><th>sMAPE</th></tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """


def generate_report(
    results: list[EvalResult],
    dataset_name: str,
    output_path: str | Path,
) -> Path:
    """Generate complete HTML report with adaptive layout."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = _chart_mode(results)
    n_series = len({r.series_id for r in results})
    n_models = len({r.model_name for r in results})

    # Aggregate summary for the top-level table
    if mode == RESULTS_AGGREGATED:
        agg, _ = _aggregate_by_model(results)
        agg.sort(key=lambda x: x["mae"])
        summary_rows = [
            {"model": a["model_name"], "mae": f"{a['mae']:.2f}",
             "rmse": f"{a['rmse']:.2f}", "smape": f"{a['smape']:.1f}%",
             "note": f"(avg over {a['n_series']} series)"}
            for a in agg
        ]
    else:
        results_sorted = sorted(results, key=lambda r: r.mae)
        summary_rows = [
            {"model": r.model_name, "mae": f"{r.mae:.2f}",
             "rmse": f"{r.rmse:.2f}", "smape": f"{r.smape:.1f}%",
             "note": ""}
            for r in results_sorted
        ]

    metric_chart = create_metric_comparison_chart(results)
    horizon_chart = create_per_horizon_chart(results)
    avp_chart = create_actual_vs_predicted_chart(results, dataset_name)
    radar_chart = create_radar_chart(results)
    detail_table = create_series_detail_table(results)

    meta_line = f"{n_series} series × {n_models} models" if mode == RESULTS_AGGREGATED else f"{n_models} models"

    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Forecasting Benchmark — {{ dataset_name }}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f8f9fa;
                color: #333;
                line-height: 1.6;
            }
            .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
            header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 3rem 2rem;
                text-align: center;
            }
            header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
            header p { font-size: 1.1rem; opacity: 0.9; }
            .meta { font-size: 0.85rem; opacity: 0.75; margin-top: 0.3rem; }
            .section {
                background: white;
                border-radius: 12px;
                padding: 2rem;
                margin: 1.5rem 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            }
            .section h2 {
                font-size: 1.5rem;
                margin-bottom: 1rem;
                color: #444;
                border-bottom: 2px solid #667eea;
                padding-bottom: 0.5rem;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 1rem;
            }
            th, td {
                padding: 0.75rem 1rem;
                text-align: left;
                border-bottom: 1px solid #eee;
            }
            th {
                background: #667eea;
                color: white;
                font-weight: 600;
            }
            tr:hover { background: #f5f5f5; }
            tr:nth-child(even) { background: #fafafa; }
            .winner { background: #d4edda !important; font-weight: 600; }
            .note { font-size: 0.8rem; color: #999; }
            .methodology {
                font-size: 0.9rem;
                color: #666;
                margin-top: 1rem;
            }
            .methodology ul { margin-left: 1.5rem; }
            footer {
                text-align: center;
                padding: 2rem;
                color: #888;
                font-size: 0.9rem;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Forecasting Benchmark</h1>
            <p>TimesFM 3.0 vs Traditional Models — {{ dataset_name }}</p>
            <p class="meta">{{ meta_line }}</p>
        </header>

        <div class="container">
            <div class="section">
                <h2>Summary{% if mode == 'aggregated' %} (Averaged Across Series){% endif %}</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>MAE</th>
                            <th>RMSE</th>
                            <th>sMAPE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in summary %}
                        <tr class="{{ 'winner' if loop.first }}">
                            <td>{{ loop.index }}</td>
                            <td>{{ row.model }} {% if row.note %}<span class="note">{{ row.note }}</span>{% endif %}</td>
                            <td>{{ row.mae }}</td>
                            <td>{{ row.rmse }}</td>
                            <td>{{ row.smape }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <div class="section">
                <h2>Metric Comparison</h2>
                {{ metric_chart }}
            </div>

            <div class="section">
                <h2>Error by Forecast Horizon</h2>
                {{ horizon_chart }}
            </div>

            <div class="section">
                <h2>Actual vs Predicted</h2>
                {{ avp_chart }}
            </div>

            <div class="section">
                <h2>Model Strengths</h2>
                {{ radar_chart }}
            </div>

            {{ detail_table }}

            <div class="section">
                <h2>Methodology</h2>
                <div class="methodology">
                    <p><strong>Evaluation Protocol:</strong></p>
                    <ul>
                        <li>Rolling-origin cross-validation (expanding window)</li>
                        <li>Initial window: 60%% of training data</li>
                        <li>Forecast horizon: 18 months</li>
                        <li>Metrics: MAE, RMSE, sMAPE (symmetric MAPE)</li>
                    </ul>
                    <p style="margin-top: 1rem;"><strong>Model Configuration:</strong></p>
                    <ul>
                        <li>TimesFM 3.0: Zero-shot (no training on target series)</li>
                        <li>Prophet: Default parameters with yearly seasonality</li>
                        <li>LightGBM: Lag features [1,2,3,6,12] + calendar features</li>
                        <li>ETS: Holt-Winters with automatic parameter estimation</li>
                        <li>Naive: Seasonal naive (repeat last 12 months)</li>
                    </ul>
                </div>
            </div>
        </div>

        <footer>
            Generated by Forecasting Lab
        </footer>
    </body>
    </html>
    """

    template = Template(html_template)
    html = template.render(
        dataset_name=dataset_name,
        meta_line=meta_line,
        mode=mode,
        summary=summary_rows,
        metric_chart=metric_chart,
        horizon_chart=horizon_chart,
        avp_chart=avp_chart,
        radar_chart=radar_chart,
        detail_table=detail_table,
    )

    output_path.write_text(html)
    return output_path
