#!/usr/bin/env python3
"""
Run focused KV cache experiments for per-frame memory scaling (resolution/dtype sweep)
and prompt-vs-generation breakdowns.

This mirrors the VLM/vLLM instrumentation in the repo while:

- measuring VRAM/CPU deltas per resized frame + dtype, separating a model footing
  from the KV cache line items, emitting a CSV + plot, and narrating why prompt vs
  generated tokens own how much KV.
- inspecting the scheduler's kv_cache_manager after a single run to report prompt/
  generation token counts, bytes attributed to each and the estimated bytes per block,
  and emitting compact JSON/CSV outputs plus a short text table.

Usage examples:

    python examples/kv_memory_profile.py          # runs both modes, writes CSV/plot
    python examples/kv_memory_profile.py --kv-dtype bfloat16 --run-kv-breakdown
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - optional plot dependency
    HAS_MATPLOTLIB = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:  # pragma: no cover - optional psutil
    HAS_PSUTIL = False
    psutil = None  # type: ignore[assignment]

try:
    import tracemalloc

    HAS_TRACEMALLOC = True
except ImportError:  # pragma: no cover - optional tracemalloc
    HAS_TRACEMALLOC = False
    tracemalloc = None  # type: ignore[assignment]

MB = 1024**2
VIDEO_NAME = "baby_reading"
DTYPE_ALIASES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32", "half": "float16"}


def capture_memory_snapshot() -> dict[str, float]:
    """Snapshot GPU, process-level CPU, and Python memory in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    snapshot = {}
    if torch.cuda.is_available():
        snapshot["gpu_allocated"] = torch.cuda.memory_allocated() / MB
        snapshot["gpu_reserved"] = torch.cuda.memory_reserved() / MB
        snapshot["gpu_max_allocated"] = torch.cuda.max_memory_allocated() / MB
    else:
        snapshot["gpu_allocated"] = 0.0
        snapshot["gpu_reserved"] = 0.0
        snapshot["gpu_max_allocated"] = 0.0
    if HAS_PSUTIL:
        proc = psutil.Process()
        mem_info = proc.memory_info()
        snapshot["cpu_rss"] = mem_info.rss / MB
        snapshot["cpu_vms"] = mem_info.vms / MB
    else:
        snapshot["cpu_rss"] = 0.0
        snapshot["cpu_vms"] = 0.0
    if HAS_TRACEMALLOC and tracemalloc is not None:
        current, peak = tracemalloc.get_traced_memory()
        snapshot["python_current"] = current / MB
        snapshot["python_peak"] = peak / MB
    else:
        snapshot["python_current"] = 0.0
        snapshot["python_peak"] = 0.0
    return snapshot


def memory_delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return max(0.0, after.get(key, 0.0) - before.get(key, 0.0))


def normalize_dtype(dtype: str) -> str:
    return DTYPE_ALIASES.get(dtype.lower(), dtype)


def resize_frame(base_frame: np.ndarray, resolution: int) -> np.ndarray:
    """Resize a single RGB frame to the requested square resolution."""
    pil_img = Image.fromarray(base_frame)
    resized = pil_img.resize((resolution, resolution), resample=Image.BILINEAR)
    return np.array(resized)


def build_prompt(resolution: int | None = None, dtype: str | None = None) -> str:
    details = []
    if resolution is not None:
        details.append(f"{resolution}p")
    if dtype is not None:
        details.append(dtype)
    detail_str = f" at {' / '.join(details)}" if details else ""
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|video_pad|><|vision_end|>"
        f"Describe what is happening in this single frame{detail_str}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def gather_kv_cache_info(llm: LLM) -> dict[str, float]:
    """Extract block-level stats from the scheduler's KV cache manager if available."""
    try:
        scheduler = llm.llm_engine.engine_core.scheduler
    except AttributeError:
        return {}
    kv_mgr = scheduler.kv_cache_manager
    if not kv_mgr.kv_cache_config.kv_cache_groups:
        return {}
    spec = kv_mgr.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    block_bytes = spec.page_size_bytes
    total_blocks = max(0, kv_mgr.block_pool.num_gpu_blocks - 1)
    usage = kv_mgr.usage
    estimated_used = usage * total_blocks
    return {
        "block_bytes": block_bytes,
        "total_blocks": total_blocks,
        "usage_ratio": usage,
        "estimated_used_blocks": estimated_used,
        "estimated_bytes_used": estimated_used * block_bytes,
    }


def run_generation(
    llm_kwargs: dict[str, Any],
    video_input: np.ndarray,
    metadata: dict[str, Any],
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    """Destroy and reset the LLM around a single request to measure memory deltas."""
    baseline = capture_memory_snapshot()
    torch.cuda.empty_cache()
    llm = LLM(**llm_kwargs)
    after_init = capture_memory_snapshot()
    start = time.perf_counter()
    outputs = llm.generate(
        {
            "prompt": build_prompt(),
            "multi_modal_data": {"video": [(video_input, metadata)]},
        },
        sampling_params=sampling_params,
    )
    end = time.perf_counter()
    final_snapshot = capture_memory_snapshot()
    try:
        request_output = outputs[0]
    except IndexError:
        prompt_tokens = 0
        generated_tokens = 0
    else:
        prompt_tokens = len(request_output.prompt_token_ids or [])
        generated_tokens = sum(len(c.token_ids) for c in request_output.outputs)
    total_tokens = max(1, prompt_tokens + generated_tokens)

    kv_info = gather_kv_cache_info(llm)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gpu_model_mb = memory_delta(after_init, baseline, "gpu_allocated")
    gpu_kv_mb = memory_delta(final_snapshot, after_init, "gpu_allocated")
    cpu_model_mb = memory_delta(after_init, baseline, "cpu_rss")
    cpu_kv_mb = memory_delta(final_snapshot, after_init, "cpu_rss")

    def split(value: float, count: int) -> float:
        if count == 0:
            return 0.0
        return value * count / total_tokens

    gpu_prompt_mb = split(gpu_kv_mb, prompt_tokens)
    gpu_generated_mb = split(gpu_kv_mb, generated_tokens)
    cpu_prompt_mb = split(cpu_kv_mb, prompt_tokens)
    cpu_generated_mb = split(cpu_kv_mb, generated_tokens)

    summary = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "gpu_model_mb": gpu_model_mb,
        "gpu_kv_mb": gpu_kv_mb,
        "gpu_kv_prompt_mb": gpu_prompt_mb,
        "gpu_kv_generated_mb": gpu_generated_mb,
        "cpu_model_mb": cpu_model_mb,
        "cpu_kv_mb": cpu_kv_mb,
        "cpu_kv_prompt_mb": cpu_prompt_mb,
        "cpu_kv_generated_mb": cpu_generated_mb,
        "kv_info": kv_info,
        "run_secs": end - start,
    }
    return summary


def run_per_frame_sweep(
    args: argparse.Namespace,
    base_frame: np.ndarray,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    records: list[dict[str, Any]] = []
    for resolution in sorted(set(args.resolutions)):
        video_input = np.expand_dims(resize_frame(base_frame, resolution), 0)
        for dtype in args.dtypes:
            normalized_dtype = normalize_dtype(dtype)
            llm_kwargs = {
                "model": args.model,
                "dtype": normalized_dtype,
                "max_model_len": 8192,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 1,
                "limit_mm_per_prompt": {"image": 10, "video": 10},
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": True,
                "compilation_config": {"level": 0},
            }
            summary = run_generation(llm_kwargs, video_input, metadata, sampling)
            kv_info = summary["kv_info"]
            record = {
                "resolution": resolution,
                "dtype": normalized_dtype,
                "prompt_tokens": summary["prompt_tokens"],
                "generated_tokens": summary["generated_tokens"],
                "gpu_model_mb": summary["gpu_model_mb"],
                "gpu_kv_mb": summary["gpu_kv_mb"],
                "gpu_kv_prompt_mb": summary["gpu_kv_prompt_mb"],
                "gpu_kv_generated_mb": summary["gpu_kv_generated_mb"],
                "cpu_model_mb": summary["cpu_model_mb"],
                "cpu_kv_mb": summary["cpu_kv_mb"],
                "cpu_kv_prompt_mb": summary["cpu_kv_prompt_mb"],
                "cpu_kv_generated_mb": summary["cpu_kv_generated_mb"],
                "kv_block_bytes": kv_info.get("block_bytes", 0),
                "kv_total_blocks": kv_info.get("total_blocks", 0),
                "kv_usage_ratio": kv_info.get("usage_ratio", 0),
                "kv_estimated_used_blocks": kv_info.get("estimated_used_blocks", 0),
                "kv_estimated_bytes_used": kv_info.get("estimated_bytes_used", 0),
                "run_secs": summary["run_secs"],
            }
            records.append(record)
            print(
                f"[Frame sweep] res={resolution} dtype={normalized_dtype} | "
                f"GPU model={record['gpu_model_mb']:.1f}MB KV={record['gpu_kv_mb']:.1f}MB "
                f"(prompt {record['gpu_kv_prompt_mb']:.1f}MB, gen {record['gpu_kv_generated_mb']:.1f}MB); "
                f"CPU KV={record['cpu_kv_mb']:.1f}MB"
            )

    if not records:
        print("No resolution/dtype combinations were provided for the sweep.")
        return

    frame_csv = output_dir / "per_frame_memory.csv"
    frame_fields = [
        "resolution",
        "dtype",
        "prompt_tokens",
        "generated_tokens",
        "gpu_model_mb",
        "gpu_kv_mb",
        "gpu_kv_prompt_mb",
        "gpu_kv_generated_mb",
        "cpu_model_mb",
        "cpu_kv_mb",
        "cpu_kv_prompt_mb",
        "cpu_kv_generated_mb",
        "kv_block_bytes",
        "kv_total_blocks",
        "kv_usage_ratio",
        "kv_estimated_used_blocks",
        "kv_estimated_bytes_used",
        "run_secs",
    ]
    with frame_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(record for record in records)

    if HAS_MATPLOTLIB:
        plot_frame_records(records, output_dir / "per_frame_memory.png")
    else:
        print("matplotlib not installed; skipping per-frame plot.")

    print(f"Per-frame sweep data written to {frame_csv}")
    if HAS_MATPLOTLIB:
        print(f"Plot saved to {output_dir / 'per_frame_memory.png'}")


def plot_frame_records(records: list[dict[str, Any]], plot_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    for dtype in sorted({record["dtype"] for record in records}):
        subset = sorted(
            (r for r in records if r["dtype"] == dtype),
            key=lambda item: item["resolution"],
        )
        xs = [r["resolution"] for r in subset]
        ys = [r["gpu_kv_mb"] for r in subset]
        plt.plot(xs, ys, marker="o", label=dtype)
    plt.xlabel("Resolution (pixels)")
    plt.ylabel("GPU KV (MB)")
    plt.title("Per-frame KV memory by resolution and dtype")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def run_kv_breakdown(
    args: argparse.Namespace,
    base_frame: np.ndarray,
    metadata: dict[str, Any],
    output_dir: Path,
) -> None:
    dtype = normalize_dtype(args.kv_dtype or args.dtypes[0])
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    video_input = np.expand_dims(base_frame, 0)
    llm_kwargs = {
        "model": args.model,
        "dtype": dtype,
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 1,
        "limit_mm_per_prompt": {"image": 10, "video": 10},
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "compilation_config": {"level": 0},
    }
    summary = run_generation(llm_kwargs, video_input, metadata, sampling)
    kv_info = summary["kv_info"]
    kv_total_bytes = summary["gpu_kv_mb"] * MB
    kv_prompt_bytes = summary["gpu_kv_prompt_mb"] * MB
    kv_generated_bytes = summary["gpu_kv_generated_mb"] * MB
    record = {
        "dtype": dtype,
        "prompt_tokens": summary["prompt_tokens"],
        "generated_tokens": summary["generated_tokens"],
        "kv_total_bytes": kv_total_bytes,
        "kv_prompt_bytes": kv_prompt_bytes,
        "kv_generated_bytes": kv_generated_bytes,
        "kv_block_bytes": kv_info.get("block_bytes", 0),
        "kv_total_blocks": kv_info.get("total_blocks", 0),
        "kv_usage_ratio": kv_info.get("usage_ratio", 0),
        "kv_estimated_used_blocks": kv_info.get("estimated_used_blocks", 0),
        "kv_estimated_bytes_used": kv_info.get("estimated_bytes_used", 0),
        "kv_bytes_per_prompt_token": kv_prompt_bytes / summary["prompt_tokens"]
        if summary["prompt_tokens"]
        else None,
        "kv_bytes_per_generated_token": kv_generated_bytes / summary["generated_tokens"]
        if summary["generated_tokens"]
        else None,
        "run_secs": summary["run_secs"],
    }

    json_path = output_dir / "kv_breakdown.json"
    csv_path = output_dir / "kv_breakdown.csv"
    json_path.write_text(json.dumps(record, indent=2))
    csv_fields = [
        "dtype",
        "prompt_tokens",
        "generated_tokens",
        "kv_total_bytes",
        "kv_prompt_bytes",
        "kv_generated_bytes",
        "kv_block_bytes",
        "kv_total_blocks",
        "kv_usage_ratio",
        "kv_estimated_used_blocks",
        "kv_estimated_bytes_used",
        "kv_bytes_per_prompt_token",
        "kv_bytes_per_generated_token",
        "run_secs",
    ]
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerow(record)

    print("KV breakdown summary")
    print(
        f"Tokens: prompt={summary['prompt_tokens']} generated={summary['generated_tokens']}"
    )
    print(
        f"KV bytes: total={kv_total_bytes:.0f}B "
        f"(prompt {kv_prompt_bytes:.0f}B, gen {kv_generated_bytes:.0f}B)"
    )
    print(
        f"Blocks: {record['kv_estimated_used_blocks']:.1f}/{record['kv_total_blocks']} "
        f"({record['kv_usage_ratio']:.2%} usage)"
    )
    prompt_per = (
        record["kv_bytes_per_prompt_token"]
        if record["kv_bytes_per_prompt_token"] is not None
        else float("nan")
    )
    gen_per = (
        record["kv_bytes_per_generated_token"]
        if record["kv_bytes_per_generated_token"] is not None
        else float("nan")
    )
    print(f"Bytes/token: prompt={prompt_per:.0f}B gen={gen_per:.0f}B")
    print(f"KV block size: {record['kv_block_bytes']} bytes")
    print(f"Run duration: {record['run_secs']:.2f}s")
    print(f"JSON output: {json_path}")
    print(f"CSV output: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frame-wise memory/precision sweep + KV breakdown reporting."
    )
    parser.add_argument(
        "--run-per-frame",
        action="store_true",
        help="Enable the resolution/dtype sweep experiment.",
    )
    parser.add_argument(
        "--run-kv-breakdown",
        action="store_true",
        help="Enable the single-run KV prompt/autoregressive breakdown.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[256, 320, 448],
        help="Square resolutions to test per frame.",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=["float32", "bfloat16", "float16"],
        help="Model dtypes to sweep (ModelDType strings or shortcuts like fp32/bf16).",
    )
    parser.add_argument(
        "--kv-dtype",
        default=None,
        help="If set, overrides the dtype used by the KV breakdown run.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Number of output tokens to generate when profiling.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model to load for the experiments.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory vLLM attempts to reserve for model + KV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/kv_memory_profiles"),
        help="Directory where CSV/plot/JSON artifacts are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.run_per_frame or args.run_kv_breakdown):
        args.run_per_frame = args.run_kv_breakdown = True
    args.dtypes = [normalize_dtype(str(dtype)) for dtype in args.dtypes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asset = VideoAsset(name=VIDEO_NAME, num_frames=1)
    base_frame = asset.np_ndarrays[0]
    metadata = asset.metadata.copy()

    tracemalloc_started = False
    if HAS_TRACEMALLOC and tracemalloc is not None:
        tracemalloc.start()
        tracemalloc_started = True

    try:
        if args.run_per_frame:
            run_per_frame_sweep(args, base_frame, metadata, args.output_dir)
        if args.run_kv_breakdown:
            run_kv_breakdown(args, base_frame, metadata, args.output_dir)
    finally:
        if tracemalloc_started and tracemalloc is not None:
            tracemalloc.stop()


if __name__ == "__main__":
    main()
