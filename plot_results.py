import csv
from argparse import ArgumentParser
from pathlib import Path


try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for plot_results.py. "
        "Install it with `python -m pip install matplotlib` or reinstall requirements-gpu.txt."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_summary",
        required=True,
        help="CSV from run_experiments.py for the main parameter sweep",
    )
    parser.add_argument(
        "--degradation_summary",
        required=True,
        help="CSV from run_degradation_benchmarks.py for degraded-video results",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "plots"),
        help="directory where figures and markdown summary will be written",
    )
    return parser.parse_args()


def load_rows(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(row, key):
    return float(row[key])


def ensure_output_dir(path):
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_parameter_sweep(rows, output_dir):
    rows = sorted(
        rows,
        key=lambda row: (
            row["motion_backend"],
            row["smoothing"],
            to_float(row, "smooth_radius"),
            to_float(row, "correction_strength"),
        ),
    )

    grouped = {}
    for row in rows:
        key = (row["motion_backend"], row["smoothing"], int(float(row["smooth_radius"])))
        grouped.setdefault(key, []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    metric_specs = [
        ("output_to_input_std_ratio_x", "Residual Motion Ratio X"),
        ("output_to_input_std_ratio_y", "Residual Motion Ratio Y"),
    ]

    for axis, (metric_key, title) in zip(axes, metric_specs):
        for (backend, smoothing, radius), group_rows in grouped.items():
            ordered = sorted(group_rows, key=lambda row: to_float(row, "correction_strength"))
            xs = [to_float(row, "correction_strength") for row in ordered]
            ys = [to_float(row, metric_key) for row in ordered]
            axis.plot(xs, ys, marker="o", linewidth=2, label=f"{backend}, {smoothing}, r={radius}")

        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Correction Strength")
        axis.set_ylabel("Output / Input Ratio")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)

    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Parameter Sweep On Main Benchmark Clip")
    output_path = output_dir / "parameter_sweep_ratios.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def degradation_sort_key(row):
    degradation_order = {"h264": 0, "blur": 1, "downscale": 2}
    parameter = row["parameter"]
    if "=" in parameter:
        _, value = parameter.split("=", 1)
        try:
            numeric_value = float(value)
        except ValueError:
            numeric_value = value
    else:
        numeric_value = parameter
    return degradation_order.get(row["degradation_type"], 99), numeric_value


def plot_degradation_bars(rows, output_dir):
    rows = sorted(rows, key=degradation_sort_key)
    labels = [row["variant_name"].replace("car_input_20s_", "") for row in rows]
    ratio_x = [to_float(row, "output_to_input_std_ratio_x") for row in rows]
    ratio_y = [to_float(row, "output_to_input_std_ratio_y") for row in rows]

    indices = list(range(len(rows)))
    width = 0.38

    fig, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    axis.bar([index - width / 2 for index in indices], ratio_x, width=width, label="Ratio X")
    axis.bar([index + width / 2 for index in indices], ratio_y, width=width, label="Ratio Y")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(indices)
    axis.set_xticklabels(labels, rotation=25, ha="right")
    axis.set_ylabel("Output / Input Ratio")
    axis.set_title("Stabilization Robustness Across Degradations")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()

    output_path = output_dir / "degradation_ratios.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_degradation_speed_tradeoff(rows, output_dir):
    rows = sorted(rows, key=degradation_sort_key)
    fps = [to_float(row, "motion_estimation_fps") for row in rows]
    avg_ratio = [
        (to_float(row, "output_to_input_std_ratio_x") + to_float(row, "output_to_input_std_ratio_y")) / 2.0
        for row in rows
    ]
    labels = [row["variant_name"].replace("car_input_20s_", "") for row in rows]

    fig, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    axis.scatter(fps, avg_ratio, s=70)
    for x_value, y_value, label in zip(fps, avg_ratio, labels):
        axis.annotate(label, (x_value, y_value), xytext=(5, 4), textcoords="offset points", fontsize=8)

    axis.set_xlabel("Motion Estimation FPS")
    axis.set_ylabel("Mean Residual Ratio ((x+y)/2)")
    axis.set_title("Speed / Accuracy Tradeoff Across Degradations")
    axis.grid(True, alpha=0.25)

    output_path = output_dir / "degradation_speed_tradeoff.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def best_experiment_row(rows):
    return min(
        rows,
        key=lambda row: (
            (to_float(row, "output_to_input_std_ratio_x") + to_float(row, "output_to_input_std_ratio_y")) / 2.0,
            -to_float(row, "motion_estimation_fps"),
        ),
    )


def write_markdown_summary(experiment_rows, degradation_rows, output_dir, figure_paths):
    best_row = best_experiment_row(experiment_rows)
    grouped_rows = {}
    for row in degradation_rows:
        grouped_rows.setdefault(row["degradation_type"], []).append(row)

    summary_path = output_dir / "report_summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Plot Summary\n\n")
        handle.write("## Best Main-Clip Setting\n\n")
        handle.write(
            f"- backend: `{best_row['motion_backend']}`\n"
            f"- smoothing: `{best_row['smoothing']}`\n"
            f"- smooth_radius: `{best_row['smooth_radius']}`\n"
            f"- correction_strength: `{best_row['correction_strength']}`\n"
            f"- ratio_xy=({float(best_row['output_to_input_std_ratio_x']):.3f}, "
            f"{float(best_row['output_to_input_std_ratio_y']):.3f})\n"
            f"- fps={float(best_row['motion_estimation_fps']):.2f}\n\n"
        )

        handle.write("## Worst Tested Case Per Degradation Type\n\n")
        for degradation_type in ["h264", "blur", "downscale"]:
            rows = grouped_rows.get(degradation_type, [])
            if not rows:
                continue
            worst_row = max(
                rows,
                key=lambda row: float(row["output_to_input_std_ratio_x"]) + float(row["output_to_input_std_ratio_y"]),
            )
            handle.write(
                f"- `{degradation_type}`: `{worst_row['variant_name']}` "
                f"with ratio_xy=({float(worst_row['output_to_input_std_ratio_x']):.3f}, "
                f"{float(worst_row['output_to_input_std_ratio_y']):.3f})\n"
            )

        handle.write("\n## Generated Figures\n\n")
        for figure_path in figure_paths:
            handle.write(f"- `{figure_path.name}`\n")

    return summary_path


def main():
    args = parse_args()
    experiment_rows = load_rows(args.experiment_summary)
    degradation_rows = load_rows(args.degradation_summary)
    output_dir = ensure_output_dir(args.output_dir)

    plt.style.use("ggplot")
    figure_paths = [
        plot_parameter_sweep(experiment_rows, output_dir),
        plot_degradation_bars(degradation_rows, output_dir),
        plot_degradation_speed_tradeoff(degradation_rows, output_dir),
    ]
    summary_path = write_markdown_summary(experiment_rows, degradation_rows, output_dir, figure_paths)

    print("Saved figures:")
    for figure_path in figure_paths:
        print(f"- {figure_path}")
    print(f"Saved markdown summary to: {summary_path}")


if __name__ == "__main__":
    main()
