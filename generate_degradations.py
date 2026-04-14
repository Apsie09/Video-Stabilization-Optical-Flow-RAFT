import csv
import subprocess
from argparse import ArgumentParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--video", required=True, help="source video used to generate degraded variants")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "degraded_videos"),
        help="directory where degraded videos and summary CSV will be written",
    )
    parser.add_argument(
        "--h264_crfs",
        nargs="+",
        type=int,
        default=[23, 28, 35],
        help="H.264 CRF values to generate",
    )
    parser.add_argument(
        "--downscale_factors",
        nargs="+",
        type=float,
        default=[0.75, 0.5],
        help="spatial downscale factors to generate",
    )
    parser.add_argument(
        "--blur_sigmas",
        nargs="+",
        type=float,
        default=[1.5, 3.0],
        help="Gaussian blur sigma values to generate",
    )
    parser.add_argument("--preset", default="medium", help="x264 preset used for generated variants")
    parser.add_argument("--quality_crf", type=int, default=18, help="x264 CRF for non-H.264 degradation outputs")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate files even if the expected output already exists",
    )
    args = parser.parse_args()

    if any(crf < 0 for crf in args.h264_crfs):
        parser.error("--h264_crfs values must be non-negative")
    if any(factor <= 0.0 or factor > 1.0 for factor in args.downscale_factors):
        parser.error("--downscale_factors values must be in the range (0, 1]")
    if any(sigma <= 0.0 for sigma in args.blur_sigmas):
        parser.error("--blur_sigmas values must be positive")
    return args


def format_float_token(value):
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") if text else "0"


def run_ffmpeg(command):
    print("  " + " ".join(command))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def write_summary(summary_path, rows):
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant_name", "degradation_type", "parameter", "output_video"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_base_command(input_video, output_video, preset, crf):
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        str(output_video),
    ]


def main():
    args = parse_args()
    input_video = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")

    summary_rows = []

    print(f"Generating degraded variants for {input_video.name}")

    for crf in args.h264_crfs:
        variant_name = f"{input_video.stem}_h264_crf{crf}"
        output_video = output_dir / f"{variant_name}.mp4"
        summary_rows.append(
            {
                "variant_name": variant_name,
                "degradation_type": "h264",
                "parameter": f"crf={crf}",
                "output_video": str(output_video),
            }
        )
        print(variant_name)
        if output_video.exists() and not args.overwrite:
            print(f"  skipping existing file: {output_video.name}")
            continue
        command = build_base_command(input_video, output_video, args.preset, crf)
        run_ffmpeg(command)

    for factor in args.downscale_factors:
        token = format_float_token(factor)
        variant_name = f"{input_video.stem}_downscale_{token}x"
        output_video = output_dir / f"{variant_name}.mp4"
        summary_rows.append(
            {
                "variant_name": variant_name,
                "degradation_type": "downscale",
                "parameter": f"factor={factor}",
                "output_video": str(output_video),
            }
        )
        print(variant_name)
        if output_video.exists() and not args.overwrite:
            print(f"  skipping existing file: {output_video.name}")
            continue
        scale_filter = f"scale=trunc(iw*{factor}/2)*2:trunc(ih*{factor}/2)*2"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            scale_filter,
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.quality_crf),
            "-c:a",
            "aac",
            str(output_video),
        ]
        run_ffmpeg(command)

    for sigma in args.blur_sigmas:
        token = format_float_token(sigma)
        variant_name = f"{input_video.stem}_blur_sigma{token}"
        output_video = output_dir / f"{variant_name}.mp4"
        summary_rows.append(
            {
                "variant_name": variant_name,
                "degradation_type": "blur",
                "parameter": f"sigma={sigma}",
                "output_video": str(output_video),
            }
        )
        print(variant_name)
        if output_video.exists() and not args.overwrite:
            print(f"  skipping existing file: {output_video.name}")
            continue
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            f"gblur=sigma={sigma}",
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.quality_crf),
            "-c:a",
            "aac",
            str(output_video),
        ]
        run_ffmpeg(command)

    summary_path = output_dir / f"{input_video.stem}_degradations.csv"
    write_summary(summary_path, summary_rows)
    print(f"Saved degradation summary CSV to: {summary_path}")


if __name__ == "__main__":
    main()
