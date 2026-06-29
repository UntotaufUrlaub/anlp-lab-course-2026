import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"

def _as_list(value):
    if isinstance(value, (str, Path)):
        return [value]
    return list(value)


def _load_result_file(result_file):
    path = Path(result_file)

    if not path.is_absolute():
        path = RESULTS_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return path, json.load(f)


def _description_slug(description):
    if not description:
        return "comparison"

    parts = []
    for key in sorted(description):
        value = description[key]
        if isinstance(value, (list, tuple, set)):
            value = "-".join(str(item) for item in value)
        parts.append(f"{key}-{value}")

    slug = "_".join(parts).lower()
    slug = re.sub(r"[^a-z0-9_.-]+", "-", slug)
    return slug.strip("-") or "comparison"


def _format_description(description):
    if not description:
        return ""

    parts = []
    for key in sorted(description):
        value = description[key]
        if isinstance(value, (list, tuple, set)):
            value = ", ".join(str(item) for item in value)
        parts.append(f"{key}: {value}")

    return " | ".join(parts)


def _metric_sort_key(metric):
    if metric == "mrr_mean":
        return ("", 0, "mrr")

    match = re.match(r"([^@]+)@(\d+)_mean$", metric)
    if match:
        return (match.group(1), int(match.group(2)), metric)

    return (metric, 0, metric)


def _selected_metrics(results_by_method, description):
    metrics = sorted(
        {
            metric
            for method_results in results_by_method.values()
            for metric, value in method_results.items()
            if metric.endswith("_mean") and isinstance(value, (int, float))
        },
        key=_metric_sort_key,
    )

    requested_metrics = description.get("metrics") if description else None
    if requested_metrics:
        requested = {
            metric if metric.endswith("_mean") else f"{metric}_mean"
            for metric in _as_list(requested_metrics)
        }
        metrics = [metric for metric in metrics if metric in requested]

    requested_k = description.get("k") if description else None
    if requested_k is not None:
        k_values = {str(k) for k in _as_list(requested_k)}
        metrics = [
            metric
            for metric in metrics
            if metric == "mrr_mean"
            or any(metric.endswith(f"@{k}_mean") for k in k_values)
        ]

    return metrics


def visualize_bar_chart(input_results, description: dict, sameMethod:bool=False):
    """
    Create a grouped bar chart comparing benchmark methods across metrics.

    Args:
        input_results: One result JSON file path, or an iterable of result JSON file paths.
            Each file should contain top-level method names and metric dictionaries. Only enter the file name.
        description: Parameters used for the benchmark run, e.g.
            {"k": [10, 50], "batchsize": 500, "seed": 42, "documentsize": 10000}.

    Returns:
        Path to the saved visualization image.
    """
    description = description or {}
    result_files = _as_list(input_results)

    results_by_method = {}
    for result_file in result_files:
        path, result_data = _load_result_file(result_file)

        if not isinstance(result_data, dict):
            raise ValueError(f"Result file must contain a JSON object: {path}")

        for method_name, method_results in result_data.items():
            if not isinstance(method_results, dict):
                continue

            # Skip duplicates
            if not sameMethod and method_name in results_by_method:
                continue

            label = method_name
            if label in results_by_method:
                label = f"{method_name} ({path.stem})"

            results_by_method[label] = method_results

    if not results_by_method:
        raise ValueError("No method results found in the provided result files.")

    metrics = _selected_metrics(results_by_method, description)
    if not metrics:
        raise ValueError("No numeric '*_mean' metrics found to visualize.")

    output_dir = VISUALIZATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    method_names = list(results_by_method.keys())
    x_positions = list(range(len(metrics)))
    bar_width = min(0.8 / max(len(method_names), 1), 0.25)

    fig_width = max(10, len(metrics) * 1.1)
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for method_index, method_name in enumerate(method_names):
        offset = (method_index - (len(method_names) - 1) / 2) * bar_width
        values = [
            results_by_method[method_name].get(metric, 0)
            for metric in metrics
        ]
        ax.bar(
            [position + offset for position in x_positions],
            values,
            width=bar_width,
            label=method_name,
        )

    metric_labels = [metric.replace("_mean", "") for metric in metrics]
    fig.suptitle(
        "Benchmark Method Comparison",
        fontsize=16,
        y=0.98,
    )
    subtitle = _format_description(description)
    if subtitle:
        ax.set_title(
            subtitle,
            fontsize=10,
            pad=10,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_labels, rotation=35, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(title="Method")

    fig.tight_layout()

    output_path = output_dir / f"benchmark_{_description_slug(description)}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path

if __name__ == "__main__":
    description = {
        "k": [2,5,10],
        "batchsize": 500,
        "seed": 42,
        "documentsize": "all",
        "hierarchical": False,
        "categorical": False,
    }
    visualize_bar_chart(["method_results.json"],
                         description)