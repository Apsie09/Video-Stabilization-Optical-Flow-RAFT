import csv
import itertools
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
STABILIZE_SCRIPT = PROJECT_ROOT / "stabilize.py"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--video", required=True, help="input video used for all experiment runs")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "experiments"),
        help="directory where stabilized videos and summary CSV will be written",
    )
    parser.add_argument(
        "--motion_backends",
        nargs="+",
        default=["lk"],
        choices=["lk", "raft"],
        help="motion-estimation backends to evaluate",
    )
    parser.add_argument("--model", help="RAFT checkpoint path; required when raft is included")
    parser.add_argument(
        "--smoothing_methods",
        nargs="+",
        default=["moving_average"],
        choices=["moving_average", "ema"],
        help="trajectory smoothing methods to evaluate",
    )
    parser.add_argument(
        "--smooth_radii",
        nargs="+",
        type=int,
        default=[15, 30],
        help="smoothing radii to evaluate",
    )
    parser.add_argument(
        "--correction_strengths",
        nargs="+",
        type=float,
        default=[0.6, 0.8, 1.0],
        help="correction strengths to evaluate",
    )
    parser.add_argument("--crop_margin", type=int, default=20, help="crop margin passed to stabilize.py")
    parser.add_argument(
        "--border_mode",
        choices=["constant", "reflect", "replicate"],
        default="reflect",
        help="warp border mode passed to stabilize.py",
    )
    parser.add_argument("--max_frames", type=int, help="optional cap on processed frames")
    parser.add_argument(
        "--skip_output_evaluation",
        action="store_true",
        help="skip output-video residual-motion evaluation for faster sweeps",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun experiments even if the expected metrics CSV already exists",
    )
    parser.add_argument(
        "--stabilize_quiet",
        action="store_true",
        help="pass --quiet to stabilize.py for less terminal output during sweeps",
    )
    args = parser.parse_args()

    if "raft" in args.motion_backends and not args.model:
        parser.error("--model is required when --motion_backends includes raft")
    if any(radius < 0 for radius in args.smooth_radii):
        parser.error("--smooth_radii values must be non-negative")
    if any(not 0.0 <= strength <= 1.0 for strength in args.correction_strengths):
        parser.error("--correction_strengths values must be in the range [0, 1]")
    return args


def format_float_token(value):
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") if text else "0"


def load_metrics(metrics_path):
    metrics = {}
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metrics[row["metric"]] = row["value"]
    return metrics


def build_run_name(video_stem, backend, smoothing, smooth_radius, correction_strength):
    return (
        f"{video_stem}_{backend}_{smoothing}"
        f"_r{smooth_radius}"
        f"_cs{format_float_token(correction_strength)}"
    )


def build_command(args, output_video, backend, smoothing, smooth_radius, correction_strength):
    command = [
        sys.executable,
        str(STABILIZE_SCRIPT),
        "--motion_backend",
        backend,
        "--video",
        args.video,
        "--output",
        str(output_video),
        "--smoothing",
        smoothing,
        "--smooth_radius",
        str(smooth_radius),
        "--correction_strength",
        str(correction_strength),
        "--crop_margin",
        str(args.crop_margin),
        "--border_mode",
        args.border_mode,
    ]

    if backend == "raft":
        command.extend(["--model", args.model])
    if args.max_frames is not None:
        command.extend(["--max_frames", str(args.max_frames)])
    if args.skip_output_evaluation:
        command.append("--skip_output_evaluation")
    if args.stabilize_quiet:
        command.append("--quiet")
    return command


def append_summary_row(summary_rows, run_name, output_video, metrics, backend, smoothing, smooth_radius, correction_strength):
    row = {
        "run_name": run_name,
        "output_video": str(output_video),
        "motion_backend": backend,
        "smoothing": smoothing,
        "smooth_radius": smooth_radius,
        "correction_strength": correction_strength,
    }
    row.update(metrics)
    summary_rows.append(row)


def write_summary(summary_path, rows):
    if not rows:
        return

    preferred_keys = [
        "run_name",
        "output_video",
        "motion_backend",
        "smoothing",
        "smooth_radius",
        "correction_strength",
        "frames_processed",
        "avg_motion_estimation_ms",
        "motion_estimation_fps",
        "input_motion_std_x",
        "input_motion_std_y",
        "output_motion_std_x",
        "output_motion_std_y",
        "output_to_input_std_ratio_x",
        "output_to_input_std_ratio_y",
        "output_to_input_mean_abs_ratio_x",
        "output_to_input_mean_abs_ratio_y",
        "valid_fov_ratio_after_crop",
    ]
    dynamic_keys = []
    for row in rows:
        for key in row.keys():
            if key not in preferred_keys and key not in dynamic_keys:
                dynamic_keys.append(key)

    fieldnames = preferred_keys + dynamic_keys
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    combinations = list(
        itertools.product(
            args.motion_backends,
            args.smoothing_methods,
            args.smooth_radii,
            args.correction_strengths,
        )
    )

    print(f"Running {len(combinations)} experiment(s) for {video_path.name}")
    for index, (backend, smoothing, smooth_radius, correction_strength) in enumerate(combinations, start=1):
        run_name = build_run_name(video_path.stem, backend, smoothing, smooth_radius, correction_strength)
        output_video = output_dir / f"{run_name}.mp4"
        metrics_path = output_dir / f"{run_name}_metrics.csv"

        print(f"[{index}/{len(combinations)}] {run_name}")
        if metrics_path.exists() and not args.overwrite:
            print(f"  skipping existing run, reusing {metrics_path.name}")
            metrics = load_metrics(metrics_path)
            append_summary_row(
                summary_rows,
                run_name,
                output_video,
                metrics,
                backend,
                smoothing,
                smooth_radius,
                correction_strength,
            )
            continue

        command = build_command(
            args,
            output_video,
            backend,
            smoothing,
            smooth_radius,
            correction_strength,
        )
        print("  " + " ".join(command))
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)

        if not metrics_path.exists():
            raise FileNotFoundError(f"Expected metrics CSV was not created: {metrics_path}")

        metrics = load_metrics(metrics_path)
        append_summary_row(
            summary_rows,
            run_name,
            output_video,
            metrics,
            backend,
            smoothing,
            smooth_radius,
            correction_strength,
        )

    summary_path = output_dir / f"{video_path.stem}_experiment_summary.csv"
    write_summary(summary_path, summary_rows)
    print(f"Saved summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
