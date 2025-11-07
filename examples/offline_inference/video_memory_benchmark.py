"""
Video Memory Orchestrator Micro-Benchmark

Measures end-to-end latency and token throughput for repeated memory snapshot
reasoning on a fixed video segment using a VLM (e.g., Qwen2.5-VL) via vLLM.

Example:
  python examples/offline_inference/video_memory_benchmark.py \
      --video /path/to/video.mp4 \
      --model Qwen/Qwen2.5-VL-3B-Instruct \
      --fps 2 --window-size 16 --runs 10 --max-tokens 256
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Iterator, List, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import decord  # type: ignore
except Exception:  # pragma: no cover
    decord = None  # type: ignore


def _ensure_video_backend() -> str:
    if decord is not None:
        return "decord"
    if cv2 is not None:
        return "opencv"
    raise RuntimeError(
        "No video backend available. Please install decord or opencv-python."
    )


def _video_fps_with_opencv(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def _iter_frames_opencv(path: str, every_n: int) -> Iterator[Tuple[np.ndarray, int]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    index = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if index % every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                yield frame_rgb, index
            index += 1
    finally:
        cap.release()


def _iter_frames_decord(path: str, sample_stride: int) -> Iterator[Tuple[np.ndarray, int]]:
    vr = decord.VideoReader(path)
    for idx in range(0, len(vr), sample_stride):
        frame_nd = vr[idx]
        frame = frame_nd.asnumpy()
        yield frame, idx


def sample_segment(path: str, target_fps: float | None, window_size: int) -> Sequence[np.ndarray]:
    backend = _ensure_video_backend()
    if backend == "opencv":
        base_fps = _video_fps_with_opencv(path)
    else:
        base_fps = 30.0
    target = target_fps or base_fps
    stride = max(1, int(round(base_fps / target)))

    frames: List[np.ndarray] = []
    if backend == "decord":
        iterator = _iter_frames_decord(path, stride)
    else:
        iterator = _iter_frames_opencv(path, stride)
    for frame, _ in iterator:
        frames.append(frame)
        if len(frames) >= window_size:
            break
    if not frames:
        raise RuntimeError("No frames sampled from video.")
    return frames


def run_benchmark(args: argparse.Namespace) -> None:
    # Prepare fixed segment and prompt
    segment = sample_segment(args.video, args.fps, args.window_size)
    prompt = (
        "SYSTEM:\nPeriodic memory snapshot benchmark.\n\n"
        "USER: Describe what just happened in the recent video segment concisely. ASSISTANT:"
    )

    # vLLM setup
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, enforce_eager=True, limit_mm_per_prompt={"video": 1})
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)

    latencies: List[float] = []
    token_counts: List[int] = []

    for _ in range(args.runs):
        inputs = {"prompt": prompt, "multi_modal_data": {"video": segment}}
        t0 = time.perf_counter()
        outputs = llm.generate([inputs], sampling_params=sampling)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        toks = 0
        try:
            toks = len(outputs[0].outputs[0].token_ids)
        except Exception:
            pass
        token_counts.append(toks)

    total_time = sum(latencies)
    total_tokens = sum(token_counts)
    metrics = {
        "runs": args.runs,
        "avg_latency_s": (total_time / args.runs) if args.runs > 0 else None,
        "p50_latency_s": statistics.median(latencies) if latencies else None,
        "p90_latency_s": statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else None,
        "total_generated_tokens": total_tokens,
        "tokens_per_second": (total_tokens / total_time) if total_time > 0 else None,
        "snapshots_per_second": (args.runs / total_time) if total_time > 0 else None,
    }

    print(json.dumps(metrics, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Video memory micro-benchmark with vLLM Qwen-VL")
    p.add_argument("--video", type=str, required=True, help="Path to input video")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="VLM model id/path for vLLM")
    p.add_argument("--fps", type=float, default=None, help="Target sampling FPS (default: native fps)")
    p.add_argument("--window-size", type=int, default=16, help="Frames per segment")
    p.add_argument("--runs", type=int, default=10, help="Number of repeated snapshot inferences")
    p.add_argument("--max-tokens", type=int, default=256, help="Max new tokens for responses")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_benchmark(args)


if __name__ == "__main__":  # pragma: no cover
    main()


