import csv
import math
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "RAFT" / "core"))

from raft import RAFT
from utils.utils import InputPadder


IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


class ProgressTracker:
    def __init__(self, total_steps, label, enabled=True, bar_width=28):
        self.total_steps = max(int(total_steps), 0)
        self.label = label
        self.enabled = enabled
        self.bar_width = bar_width
        self.current = 0

    def update(self, step=None, extra_text=""):
        if not self.enabled:
            return

        if step is None:
            self.current += 1
        else:
            self.current = max(0, min(int(step), self.total_steps))

        ratio = (self.current / self.total_steps) if self.total_steps else 1.0
        filled = int(round(self.bar_width * ratio))
        filled = min(filled, self.bar_width)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        suffix = f" {extra_text}" if extra_text else ""
        print(
            f"\r{self.label}: [{bar}] {self.current}/{self.total_steps}{suffix}",
            end="",
            flush=True,
        )

    def finish(self, extra_text="done"):
        if not self.enabled:
            return
        self.current = self.total_steps
        self.update(self.total_steps, extra_text)
        print(flush=True)


def frame_to_tensor(frame, device):
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
    return tensor.to(device)


def normalize_state_dict(state_dict):
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def load_raft_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAFT(args)

    state_dict = torch.load(args.model, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(normalize_state_dict(state_dict))
    model.to(device)
    model.eval()
    return model, device


def build_motion_estimator(args):
    if args.motion_backend == "raft":
        model, device = load_raft_model(args)
        return {
            "backend": "raft",
            "model": model,
            "device": device,
        }

    if args.motion_backend == "lk":
        return {"backend": "lk"}

    raise ValueError(f"Unsupported motion backend: {args.motion_backend}")


def infer_flow(model, device, frame_1, frame_2, iters):
    image_1 = frame_to_tensor(frame_1, device)
    image_2 = frame_to_tensor(frame_2, device)
    padder = InputPadder(image_1.shape)
    image_1, image_2 = padder.pad(image_1, image_2)

    start = time.perf_counter()
    with torch.no_grad():
        _, flow_up = model(image_1, image_2, iters=iters, test_mode=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
    return flow.astype(np.float32), elapsed_ms


def estimate_affine_from_points(src, dst, ransac_threshold):
    if len(src) < 3:
        return IDENTITY_AFFINE.copy(), 0, len(src)

    matrix, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )

    if matrix is None:
        return IDENTITY_AFFINE.copy(), 0, len(src)

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return matrix.astype(np.float32), inlier_count, len(src)


def dominant_motion_mask(sampled_flow, consistency_threshold, consistency_percentile):
    if len(sampled_flow) == 0:
        return np.zeros((0,), dtype=bool)

    dominant_flow = np.median(sampled_flow, axis=0)
    deviations = np.linalg.norm(sampled_flow - dominant_flow, axis=1)

    adaptive_threshold = np.percentile(
        deviations,
        np.clip(consistency_percentile, 0.0, 100.0),
    )
    threshold = max(float(consistency_threshold), float(adaptive_threshold))
    mask = deviations <= threshold

    if mask.sum() >= 3:
        return mask

    fallback_count = min(len(sampled_flow), max(3, len(sampled_flow) // 2))
    keep_indices = np.argsort(deviations)[:fallback_count]
    fallback_mask = np.zeros(len(sampled_flow), dtype=bool)
    fallback_mask[keep_indices] = True
    return fallback_mask


def flow_to_correspondences(
    flow,
    sample_step,
    max_magnitude,
    consistency_threshold,
    consistency_percentile,
):
    height, width = flow.shape[:2]
    start_y = sample_step // 2
    start_x = sample_step // 2

    ys = np.arange(start_y, height, sample_step, dtype=np.int32)
    xs = np.arange(start_x, width, sample_step, dtype=np.int32)
    if ys.size == 0 or xs.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    grid_x, grid_y = np.meshgrid(xs, ys)
    src = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2).astype(np.float32)
    sampled_flow = flow[grid_y, grid_x].reshape(-1, 2).astype(np.float32)
    dst = src + sampled_flow

    valid = np.isfinite(dst).all(axis=1)
    if max_magnitude > 0:
        magnitudes = np.linalg.norm(sampled_flow, axis=1)
        valid &= magnitudes <= max_magnitude

    src = src[valid]
    dst = dst[valid]
    sampled_flow = sampled_flow[valid]

    consistency_mask = dominant_motion_mask(
        sampled_flow,
        consistency_threshold,
        consistency_percentile,
    )
    return src[consistency_mask], dst[consistency_mask]


def estimate_affine_from_flow(
    flow,
    sample_step,
    ransac_threshold,
    max_magnitude,
    consistency_threshold,
    consistency_percentile,
):
    src, dst = flow_to_correspondences(
        flow,
        sample_step,
        max_magnitude,
        consistency_threshold,
        consistency_percentile,
    )
    return estimate_affine_from_points(src, dst, ransac_threshold)


def lk_correspondences(frame_1, frame_2, args):
    gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    features = cv2.goodFeaturesToTrack(
        gray_1,
        maxCorners=args.lk_max_corners,
        qualityLevel=args.lk_quality_level,
        minDistance=args.lk_min_distance,
        blockSize=args.lk_block_size,
    )
    if features is None or len(features) < 3:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_1,
        gray_2,
        features,
        None,
        winSize=(args.lk_window_size, args.lk_window_size),
        maxLevel=args.lk_max_level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            args.lk_criteria_count,
            args.lk_criteria_eps,
        ),
    )

    if tracked is None or status is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    src = features.reshape(-1, 2).astype(np.float32)
    dst = tracked.reshape(-1, 2).astype(np.float32)
    valid = status.reshape(-1).astype(bool)
    valid &= np.isfinite(src).all(axis=1)
    valid &= np.isfinite(dst).all(axis=1)

    src = src[valid]
    dst = dst[valid]
    if len(src) == 0:
        return src, dst

    displacement = dst - src
    if args.max_flow_magnitude > 0:
        magnitudes = np.linalg.norm(displacement, axis=1)
        keep = magnitudes <= args.max_flow_magnitude
        src = src[keep]
        dst = dst[keep]
        displacement = displacement[keep]

    consistency_mask = dominant_motion_mask(
        displacement,
        args.flow_consistency_threshold,
        args.flow_consistency_percentile,
    )
    return src[consistency_mask], dst[consistency_mask]


def estimate_affine_lk(frame_1, frame_2, args):
    start = time.perf_counter()
    src, dst = lk_correspondences(frame_1, frame_2, args)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    matrix, inliers, samples = estimate_affine_from_points(
        src,
        dst,
        args.ransac_threshold,
    )
    return matrix, elapsed_ms, inliers, samples


def estimate_motion_between_frames(estimator, frame_1, frame_2, args):
    if estimator["backend"] == "raft":
        flow, elapsed_ms = infer_flow(
            estimator["model"],
            estimator["device"],
            frame_1,
            frame_2,
            args.iters,
        )
        matrix, inliers, samples = estimate_affine_from_flow(
            flow,
            args.sample_step,
            args.ransac_threshold,
            args.max_flow_magnitude,
            args.flow_consistency_threshold,
            args.flow_consistency_percentile,
        )
        return matrix, elapsed_ms, inliers, samples

    if estimator["backend"] == "lk":
        return estimate_affine_lk(frame_1, frame_2, args)

    raise ValueError(f"Unsupported motion backend: {estimator['backend']}")


def affine_to_params(matrix):
    dx = float(matrix[0, 2])
    dy = float(matrix[1, 2])
    angle = math.atan2(matrix[1, 0], matrix[0, 0])
    scale = math.sqrt(float(matrix[0, 0] ** 2 + matrix[1, 0] ** 2))
    scale = max(scale, 1e-6)
    log_scale = math.log(scale)
    return np.array([dx, dy, angle, log_scale], dtype=np.float32)


def params_to_affine(params):
    dx, dy, angle, log_scale = [float(value) for value in params]
    scale = math.exp(log_scale)
    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale

    return np.array(
        [[cos_a, -sin_a, dx], [sin_a, cos_a, dy]],
        dtype=np.float32,
    )


def moving_average_curve(curve, radius):
    if radius <= 0:
        return curve.copy()

    window = 2 * radius + 1
    padded = np.pad(curve, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(padded, kernel, mode="valid")


def ema_curve(curve, alpha):
    if len(curve) == 0:
        return curve.copy()

    forward = np.empty_like(curve)
    forward[0] = curve[0]
    for index in range(1, len(curve)):
        forward[index] = alpha * curve[index] + (1.0 - alpha) * forward[index - 1]

    backward = np.empty_like(curve)
    backward[-1] = forward[-1]
    for index in range(len(curve) - 2, -1, -1):
        backward[index] = alpha * forward[index] + (1.0 - alpha) * backward[index + 1]

    return backward


def smooth_trajectory(trajectory, method, radius, alpha):
    smoothed = trajectory.copy()
    for column in range(trajectory.shape[1]):
        curve = trajectory[:, column]
        if method == "moving_average":
            smoothed[:, column] = moving_average_curve(curve, radius)
        elif method == "ema":
            smoothed[:, column] = ema_curve(curve, alpha)
        else:
            raise ValueError(f"Unsupported smoothing method: {method}")
    return smoothed


def trajectory_to_transforms(trajectory):
    if len(trajectory) == 0:
        return trajectory.copy()

    transforms = trajectory.copy()
    transforms[1:] = trajectory[1:] - trajectory[:-1]
    return transforms


def border_mode_from_name(name):
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT,
        "replicate": cv2.BORDER_REPLICATE,
    }
    return border_modes[name]


def apply_crop(frame, crop_margin):
    if crop_margin <= 0:
        return frame

    height, width = frame.shape[:2]
    if crop_margin * 2 >= height or crop_margin * 2 >= width:
        raise ValueError("crop_margin is too large for the input frame size")

    cropped = frame[crop_margin : height - crop_margin, crop_margin : width - crop_margin]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def warp_frame(frame, matrix, border_mode, crop_margin):
    height, width = frame.shape[:2]
    stabilized = cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
    )
    return apply_crop(stabilized, crop_margin)


def motion_stat_summary(transforms):
    motion = transforms[:, :3] if len(transforms) else np.zeros((0, 3), dtype=np.float32)
    if len(motion) == 0:
        return {
            "std_x": 0.0,
            "std_y": 0.0,
            "std_rot": 0.0,
            "mean_abs_x": 0.0,
            "mean_abs_y": 0.0,
            "mean_abs_rot": 0.0,
        }

    return {
        "std_x": float(np.std(motion[:, 0])),
        "std_y": float(np.std(motion[:, 1])),
        "std_rot": float(np.std(motion[:, 2])),
        "mean_abs_x": float(np.mean(np.abs(motion[:, 0]))),
        "mean_abs_y": float(np.mean(np.abs(motion[:, 1]))),
        "mean_abs_rot": float(np.mean(np.abs(motion[:, 2]))),
    }


def safe_ratio(numerator, denominator):
    if abs(denominator) < 1e-8:
        return 0.0
    return float(numerator / denominator)


def compute_metrics(
    motion_backend,
    correction_strength,
    input_transforms,
    correction_transforms,
    corrected_transforms,
    output_transforms,
    timings_ms,
    evaluation_timings_ms,
    crop_margin,
    width,
    height,
):
    input_motion = motion_stat_summary(input_transforms)
    planned_motion = motion_stat_summary(correction_transforms)
    target_motion = motion_stat_summary(corrected_transforms)
    output_motion = motion_stat_summary(output_transforms)

    valid_width = max(width - 2 * crop_margin, 0)
    valid_height = max(height - 2 * crop_margin, 0)
    valid_ratio = (valid_width * valid_height) / float(width * height) if width and height else 0.0

    return {
        "motion_backend": motion_backend,
        "correction_strength": float(correction_strength),
        "frames_processed": int(len(input_transforms) + 1 if len(input_transforms) else 0),
        "avg_motion_estimation_ms": float(np.mean(timings_ms)) if timings_ms else 0.0,
        "motion_estimation_fps": float(1000.0 / np.mean(timings_ms)) if timings_ms else 0.0,
        "avg_output_eval_estimation_ms": float(np.mean(evaluation_timings_ms)) if evaluation_timings_ms else 0.0,
        "output_eval_estimation_fps": float(1000.0 / np.mean(evaluation_timings_ms))
        if evaluation_timings_ms
        else 0.0,
        "input_motion_std_x": input_motion["std_x"],
        "input_motion_std_y": input_motion["std_y"],
        "input_motion_std_rot_rad": input_motion["std_rot"],
        "input_motion_mean_abs_x": input_motion["mean_abs_x"],
        "input_motion_mean_abs_y": input_motion["mean_abs_y"],
        "input_motion_mean_abs_rot_rad": input_motion["mean_abs_rot"],
        "planned_correction_std_x": planned_motion["std_x"],
        "planned_correction_std_y": planned_motion["std_y"],
        "planned_correction_std_rot_rad": planned_motion["std_rot"],
        "planned_correction_mean_abs_x": planned_motion["mean_abs_x"],
        "planned_correction_mean_abs_y": planned_motion["mean_abs_y"],
        "planned_correction_mean_abs_rot_rad": planned_motion["mean_abs_rot"],
        "target_motion_std_x": target_motion["std_x"],
        "target_motion_std_y": target_motion["std_y"],
        "target_motion_std_rot_rad": target_motion["std_rot"],
        "target_motion_mean_abs_x": target_motion["mean_abs_x"],
        "target_motion_mean_abs_y": target_motion["mean_abs_y"],
        "target_motion_mean_abs_rot_rad": target_motion["mean_abs_rot"],
        "output_motion_std_x": output_motion["std_x"],
        "output_motion_std_y": output_motion["std_y"],
        "output_motion_std_rot_rad": output_motion["std_rot"],
        "output_motion_mean_abs_x": output_motion["mean_abs_x"],
        "output_motion_mean_abs_y": output_motion["mean_abs_y"],
        "output_motion_mean_abs_rot_rad": output_motion["mean_abs_rot"],
        "output_to_input_std_ratio_x": safe_ratio(output_motion["std_x"], input_motion["std_x"]),
        "output_to_input_std_ratio_y": safe_ratio(output_motion["std_y"], input_motion["std_y"]),
        "output_to_input_std_ratio_rot": safe_ratio(output_motion["std_rot"], input_motion["std_rot"]),
        "output_to_input_mean_abs_ratio_x": safe_ratio(output_motion["mean_abs_x"], input_motion["mean_abs_x"]),
        "output_to_input_mean_abs_ratio_y": safe_ratio(output_motion["mean_abs_y"], input_motion["mean_abs_y"]),
        "output_to_input_mean_abs_ratio_rot": safe_ratio(
            output_motion["mean_abs_rot"],
            input_motion["mean_abs_rot"],
        ),
        "valid_fov_ratio_after_crop": float(valid_ratio),
    }


def write_metrics(metrics_path, metrics):
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


def write_trajectory_csv(
    csv_path,
    raw_transforms,
    trajectory,
    smoothed_trajectory,
    corrected_transforms,
    inlier_counts,
    sample_counts,
):
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_idx",
                "dx",
                "dy",
                "angle_rad",
                "log_scale",
                "traj_x",
                "traj_y",
                "traj_angle_rad",
                "traj_log_scale",
                "smooth_x",
                "smooth_y",
                "smooth_angle_rad",
                "smooth_log_scale",
                "corrected_dx",
                "corrected_dy",
                "corrected_angle_rad",
                "corrected_log_scale",
                "ransac_inliers",
                "sample_points",
            ]
        )

        for index in range(len(raw_transforms)):
            writer.writerow(
                [
                    index + 1,
                    *raw_transforms[index].tolist(),
                    *trajectory[index].tolist(),
                    *smoothed_trajectory[index].tolist(),
                    *corrected_transforms[index].tolist(),
                    inlier_counts[index],
                    sample_counts[index],
                ]
            )


def estimate_motion_sequence(estimator, capture, args, video_path, progress_label):
    ret, previous_frame = capture.read()
    if not ret or previous_frame is None:
        raise RuntimeError(f"Failed to read first frame from video: {video_path}")

    transforms = []
    inference_timings = []
    inlier_counts = []
    sample_counts = []

    available_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames and args.max_frames > 0 and available_frames > 0:
        effective_frames = min(available_frames, args.max_frames)
    else:
        effective_frames = available_frames if available_frames > 0 else 0

    expected_pairs = max(effective_frames - 1, 0)
    progress = ProgressTracker(
        expected_pairs,
        progress_label,
        enabled=not args.quiet,
    )

    frame_limit = args.max_frames - 1 if args.max_frames and args.max_frames > 0 else None
    processed_pairs = 0

    while True:
        if frame_limit is not None and processed_pairs >= frame_limit:
            break

        ret, current_frame = capture.read()
        if not ret or current_frame is None:
            break

        matrix, elapsed_ms, inliers, samples = estimate_motion_between_frames(
            estimator,
            previous_frame,
            current_frame,
            args,
        )

        transforms.append(affine_to_params(matrix))
        inference_timings.append(elapsed_ms)
        inlier_counts.append(inliers)
        sample_counts.append(samples)
        progress.update(
            processed_pairs + 1,
            extra_text=f"{elapsed_ms:.1f} ms/frame",
        )

        previous_frame = current_frame
        processed_pairs += 1

    if not transforms:
        raise RuntimeError("Need at least two frames to estimate stabilization transforms")

    progress.finish(
        extra_text=(
            f"avg {np.mean(inference_timings):.1f} ms/frame"
            if inference_timings
            else "done"
        )
    )

    return (
        np.asarray(transforms, dtype=np.float32),
        inference_timings,
        inlier_counts,
        sample_counts,
    )


def analyze_video_motion(estimator, video_path, args, progress_label):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    try:
        return estimate_motion_sequence(
            estimator,
            capture,
            args,
            video_path,
            progress_label,
        )
    finally:
        capture.release()


def stabilize_video(input_path, output_path, warp_transforms, args):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.max_frames and args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    progress = ProgressTracker(
        total_frames,
        "Writing output",
        enabled=not args.quiet,
    )

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    ret, frame = capture.read()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read first frame from video: {input_path}")

    border_mode = border_mode_from_name(args.border_mode)
    writer.write(apply_crop(frame, args.crop_margin))
    progress.update(1)

    frame_index = 1
    while frame_index < total_frames:
        ret, frame = capture.read()
        if not ret or frame is None:
            break

        matrix = params_to_affine(warp_transforms[frame_index - 1])
        stabilized = warp_frame(frame, matrix, border_mode, args.crop_margin)
        writer.write(stabilized)

        if args.preview:
            preview_frame = np.hstack([frame, stabilized])
            cv2.imshow("Stabilization Preview", preview_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

        frame_index += 1
        progress.update(frame_index)

    capture.release()
    writer.release()
    if args.preview:
        cv2.destroyAllWindows()
    progress.finish()


def build_output_paths(video_path, output_path):
    video_path = Path(video_path)
    if output_path:
        output_video = Path(output_path)
    else:
        output_video = video_path.with_name(f"{video_path.stem}_stabilized.mp4")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_video.with_name(f"{output_video.stem}_metrics.csv")
    trajectory_path = output_video.with_name(f"{output_video.stem}_trajectory.csv")
    return output_video, metrics_path, trajectory_path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--motion_backend",
        choices=["raft", "lk"],
        default="raft",
        help="camera-motion estimation backend",
    )
    parser.add_argument("--model", help="path to pretrained RAFT checkpoint")
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--output", help="output stabilized video path")
    parser.add_argument("--iters", type=int, default=12, help="RAFT update iterations")
    parser.add_argument("--sample_step", type=int, default=16, help="grid step for flow sampling")
    parser.add_argument(
        "--max_flow_magnitude",
        type=float,
        default=80.0,
        help="discard sampled flow vectors above this magnitude in pixels; <=0 disables the filter",
    )
    parser.add_argument(
        "--ransac_threshold",
        type=float,
        default=3.0,
        help="RANSAC reprojection threshold in pixels",
    )
    parser.add_argument(
        "--flow_consistency_threshold",
        type=float,
        default=1.5,
        help="minimum flow-deviation threshold in pixels when keeping points near the dominant motion",
    )
    parser.add_argument(
        "--flow_consistency_percentile",
        type=float,
        default=65.0,
        help="keep points whose flow deviation is within this percentile of the dominant-motion deviation distribution",
    )
    parser.add_argument(
        "--lk_max_corners",
        type=int,
        default=600,
        help="maximum number of corners for LK feature detection",
    )
    parser.add_argument(
        "--lk_quality_level",
        type=float,
        default=0.01,
        help="quality threshold for LK feature detection",
    )
    parser.add_argument(
        "--lk_min_distance",
        type=float,
        default=15.0,
        help="minimum distance between LK features",
    )
    parser.add_argument(
        "--lk_block_size",
        type=int,
        default=7,
        help="block size for LK feature detection",
    )
    parser.add_argument(
        "--lk_window_size",
        type=int,
        default=21,
        help="window size for pyramidal Lucas-Kanade tracking",
    )
    parser.add_argument(
        "--lk_max_level",
        type=int,
        default=3,
        help="number of pyramid levels for Lucas-Kanade tracking",
    )
    parser.add_argument(
        "--lk_criteria_count",
        type=int,
        default=30,
        help="maximum iterations for Lucas-Kanade termination criteria",
    )
    parser.add_argument(
        "--lk_criteria_eps",
        type=float,
        default=0.01,
        help="epsilon for Lucas-Kanade termination criteria",
    )
    parser.add_argument(
        "--smoothing",
        choices=["moving_average", "ema"],
        default="moving_average",
        help="trajectory smoothing method",
    )
    parser.add_argument(
        "--smooth_radius",
        type=int,
        default=15,
        help="window radius for moving-average smoothing",
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.2,
        help="smoothing factor for EMA smoothing",
    )
    parser.add_argument(
        "--correction_strength",
        type=float,
        default=1.0,
        help="blend factor for applying the smoothed correction; 0 disables correction, 1 applies it fully",
    )
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=20,
        help="crop pixels from each border after warping and resize back to original size",
    )
    parser.add_argument(
        "--border_mode",
        choices=["constant", "reflect", "replicate"],
        default="reflect",
        help="OpenCV border mode used during warping",
    )
    parser.add_argument("--codec", default="mp4v", help="fourcc codec for output video")
    parser.add_argument("--max_frames", type=int, help="optional cap on processed frames")
    parser.add_argument("--preview", action="store_true", help="show original vs stabilized preview")
    parser.add_argument("--quiet", action="store_true", help="disable console progress output")
    parser.add_argument(
        "--skip_output_evaluation",
        action="store_true",
        help="skip the extra pass that measures residual motion on the stabilized output video",
    )
    parser.add_argument("--small", action="store_true", help="use RAFT small model")
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="enable mixed precision inference when CUDA is available",
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use alternate correlation implementation if available",
    )
    args = parser.parse_args()
    if args.motion_backend == "raft" and not args.model:
        parser.error("--model is required when --motion_backend=raft")
    if args.sample_step <= 0:
        parser.error("--sample_step must be positive")
    if not 0.0 < args.ema_alpha <= 1.0:
        parser.error("--ema_alpha must be in the range (0, 1]")
    if not 0.0 <= args.correction_strength <= 1.0:
        parser.error("--correction_strength must be in the range [0, 1]")
    if args.crop_margin < 0:
        parser.error("--crop_margin cannot be negative")
    if args.flow_consistency_threshold < 0:
        parser.error("--flow_consistency_threshold cannot be negative")
    if not 0.0 <= args.flow_consistency_percentile <= 100.0:
        parser.error("--flow_consistency_percentile must be in the range [0, 100]")
    if args.lk_max_corners <= 0:
        parser.error("--lk_max_corners must be positive")
    if not 0.0 < args.lk_quality_level <= 1.0:
        parser.error("--lk_quality_level must be in the range (0, 1]")
    if args.lk_min_distance < 0:
        parser.error("--lk_min_distance cannot be negative")
    if args.lk_block_size <= 0:
        parser.error("--lk_block_size must be positive")
    if args.lk_window_size <= 0:
        parser.error("--lk_window_size must be positive")
    if args.lk_max_level < 0:
        parser.error("--lk_max_level cannot be negative")
    if args.lk_criteria_count <= 0:
        parser.error("--lk_criteria_count must be positive")
    if args.lk_criteria_eps <= 0:
        parser.error("--lk_criteria_eps must be positive")
    return args


def main():
    args = parse_args()
    output_video, metrics_path, trajectory_path = build_output_paths(args.video, args.output)

    estimator = build_motion_estimator(args)

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {args.video}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    raw_transforms, inference_timings, inlier_counts, sample_counts = analyze_video_motion(
        estimator,
        args.video,
        args,
        "Estimating input motion",
    )

    trajectory = np.cumsum(raw_transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(
        trajectory,
        args.smoothing,
        args.smooth_radius,
        args.ema_alpha,
    )
    correction = (smoothed_trajectory - trajectory) * args.correction_strength
    target_trajectory = trajectory + correction
    corrected_transforms = trajectory_to_transforms(target_trajectory)

    stabilize_video(args.video, output_video, correction, args)

    output_transforms = np.zeros((0, 4), dtype=np.float32)
    evaluation_timings = []
    if not args.skip_output_evaluation:
        output_transforms, evaluation_timings, _, _ = analyze_video_motion(
            estimator,
            output_video,
            args,
            "Evaluating output motion",
        )

    metrics = compute_metrics(
        args.motion_backend,
        args.correction_strength,
        raw_transforms,
        correction,
        corrected_transforms,
        output_transforms,
        inference_timings,
        evaluation_timings,
        args.crop_margin,
        width,
        height,
    )
    write_metrics(metrics_path, metrics)
    write_trajectory_csv(
        trajectory_path,
        raw_transforms,
        trajectory,
        smoothed_trajectory,
        corrected_transforms,
        inlier_counts,
        sample_counts,
    )

    print(f"Saved stabilized video to: {output_video}")
    print(f"Saved metrics CSV to: {metrics_path}")
    print(f"Saved trajectory CSV to: {trajectory_path}")
    print(
        "Summary:"
        f" backend={metrics['motion_backend']},"
        f" avg_estimation_ms={metrics['avg_motion_estimation_ms']:.2f},"
        f" fps_estimate={metrics['motion_estimation_fps']:.2f},"
        f" valid_fov_ratio={metrics['valid_fov_ratio_after_crop']:.4f}"
    )
    if not args.skip_output_evaluation:
        print(
            "Residual motion:"
            f" input_std_xy=({metrics['input_motion_std_x']:.4f}, {metrics['input_motion_std_y']:.4f}),"
            f" output_std_xy=({metrics['output_motion_std_x']:.4f}, {metrics['output_motion_std_y']:.4f}),"
            f" ratio_xy=({metrics['output_to_input_std_ratio_x']:.3f}, {metrics['output_to_input_std_ratio_y']:.3f})"
        )


if __name__ == "__main__":
    main()
