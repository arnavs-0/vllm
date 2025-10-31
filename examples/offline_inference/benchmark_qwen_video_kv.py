#!/usr/bin/env python3
"""
Benchmark latency and KV cache usage for a video sent to Qwen-VL with vLLM.

This script:
- Loads a Qwen-VL model via vLLM
- Builds one video input (frames + metadata) using get_multi_modal_input
- Runs a single generate() and collects Prometheus metrics before/after
- Computes prefill/decode/e2e times and KV cache usage
- Plots simple graphs for latency breakdown and KV cache usage

Examples:
  python examples/offline_inference/benchmark_qwen_video_kv.py \
    --model Qwen/Qwen3-VL-4B-Instruct --num-frames 16 --max-tokens 64 --fps 1
"""

import argparse
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Metric

# Reuse helpers from examples/offline_inference/vision_language.py
try:
    from examples.offline_inference.vision_language import (
        get_multi_modal_input,
        run_qwen3_vl,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import helpers from examples.offline_inference.vision_language."
    ) from e


@dataclass
class RunStats:
    e2e_time: float
    prefill_time: float
    decode_time: float
    ttft: float
    kv_usage_perc: float
    prompt_tokens: int
    generation_tokens: int
    cache_block_size: Optional[int]
    cache_num_gpu_blocks: Optional[int]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark video latency and KV cache usage for Qwen-VL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="HF model id for Qwen-VL",
    )
    parser.add_argument(
        "--num-frames", type=int, default=16, help="Number of frames from the video"
    )
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Sampling fps to pass to processor"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64, help="Max new tokens to generate"
    )
    parser.add_argument(
        "--mm-cache-gb",
        type=float,
        default=2.0,
        help="MM processor cache size in GB (0 to disable)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help="Set vLLM max_model_len to accommodate text + video tokens",
    )
    parser.add_argument(
        "--save-fig",
        type=str,
        default=None,
        help="Optional path to save the generated figure (png)",
    )
    return parser.parse_args()


def metrics_by_name(metrics: List[Metric]) -> Dict[str, List[Metric]]:
    by_name: Dict[str, List[Metric]] = {}
    for m in metrics:
        by_name.setdefault(m.name, []).append(m)
    return by_name


def get_hist_value_delta(
    pre: List[Metric], post: List[Metric], metric_name: str
) -> float:
    pre_map = metrics_by_name(pre)
    post_map = metrics_by_name(post)
    pre_list = pre_map.get(metric_name, [])
    post_list = post_map.get(metric_name, [])
    pre_sum = sum(m.sum for m in pre_list if isinstance(m, Histogram))
    post_sum = sum(m.sum for m in post_list if isinstance(m, Histogram))
    return max(0.0, post_sum - pre_sum)


def get_counter_delta(pre: List[Metric], post: List[Metric], metric_name: str) -> int:
    pre_map = metrics_by_name(pre)
    post_map = metrics_by_name(post)
    pre_val = sum(m.value for m in pre_map.get(metric_name, []) if isinstance(m, Counter))
    post_val = sum(
        m.value for m in post_map.get(metric_name, []) if isinstance(m, Counter)
    )
    return max(0, post_val - pre_val)


def get_latest_gauge_value(metrics: List[Metric], metric_name: str) -> Optional[float]:
    vals = [m.value for m in metrics_by_name(metrics).get(metric_name, []) if isinstance(m, Gauge)]
    return vals[-1] if vals else None


def extract_cache_config_info(metrics: List[Metric]) -> Tuple[Optional[int], Optional[int]]:
    # vllm:cache_config_info is a Gauge with labels including block_size and num_gpu_blocks
    for m in metrics:
        if isinstance(m, Gauge) and m.name == "vllm:cache_config_info":
            labels = m.labels
            block_size = labels.get("block_size")
            num_gpu_blocks = labels.get("num_gpu_blocks")
            try:
                return (int(block_size) if block_size is not None else None,
                        int(num_gpu_blocks) if num_gpu_blocks is not None else None)
            except Exception:
                return (None, None)
    return (None, None)


def build_input_for_qwen_video(model_name: str, num_frames: int, fps: float):
    # Build EngineArgs + prompts via run_qwen3_vl
    req_data = run_qwen3_vl(["Describe this video."], "video")
    # Override to use the requested model/fps if needed
    req_data.engine_args.model = model_name
    req_data.engine_args.mm_processor_kwargs = req_data.engine_args.mm_processor_kwargs or {}
    req_data.engine_args.mm_processor_kwargs["fps"] = fps

    # Prepare multi-modal input
    class _Args:
        def __init__(self, model_type: str, modality: str, num_frames: int):
            self.model_type = model_type
            self.modality = modality
            self.num_frames = num_frames

    mm = get_multi_modal_input(_Args("qwen3_vl", "video", num_frames))
    data = mm["data"]
    prompt = req_data.prompts[0]
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"video": data},
        # UUID helps MM processor cache across repeats if any
        "multi_modal_uuids": {"video": "vid_uuid_0"},
    }
    return req_data.engine_args, inputs


def run_once(
    model_name: str,
    num_frames: int,
    fps: float,
    max_tokens: int,
    mm_cache_gb: float,
    max_model_len: int,
) -> RunStats:
    engine_args, inputs = build_input_for_qwen_video(model_name, num_frames, fps)

    # Limit to a single video per prompt
    engine_args.limit_mm_per_prompt = {"video": 1}
    # Enable/disable MM processor cache
    engine_args.mm_processor_cache_gb = max(0.0, mm_cache_gb)
    # Ensure context window can fit text + multimodal tokens
    try:
        current = getattr(engine_args, "max_model_len", None) or 0
        engine_args.max_model_len = max(current, int(max_model_len))
    except Exception:
        engine_args.max_model_len = int(max_model_len)

    llm = LLM(**engine_args.__dict__)
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    # Snapshot metrics BEFORE
    pre_metrics = llm.get_metrics()

    t0 = time.time()
    outputs = llm.generate([inputs], sampling_params=sampling_params)
    e2e_time = time.time() - t0

    # Snapshot metrics AFTER
    post_metrics = llm.get_metrics()

    # Extract latency components from histograms (delta for this request)
    ttft = get_hist_value_delta(pre_metrics, post_metrics, "vllm:time_to_first_token_seconds")
    prefill_time = get_hist_value_delta(pre_metrics, post_metrics, "vllm:request_prefill_time_seconds")
    decode_time = get_hist_value_delta(pre_metrics, post_metrics, "vllm:request_decode_time_seconds")

    # KV cache usage percentage (engine-level gauge, last recorded)
    kv_usage_perc = get_latest_gauge_value(post_metrics, "vllm:kv_cache_usage_perc") or 0.0

    # Counters deltas
    prompt_tokens = get_counter_delta(pre_metrics, post_metrics, "vllm:prompt_tokens")
    generation_tokens = get_counter_delta(pre_metrics, post_metrics, "vllm:generation_tokens")

    block_size, num_gpu_blocks = extract_cache_config_info(post_metrics)

    # Cleanup engine
    del llm

    return RunStats(
        e2e_time=e2e_time,
        prefill_time=prefill_time,
        decode_time=decode_time,
        ttft=ttft,
        kv_usage_perc=kv_usage_perc,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        cache_block_size=block_size,
        cache_num_gpu_blocks=num_gpu_blocks,
    )


def plot_results(stats: RunStats, save_path: Optional[str] = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Latency breakdown
    axes[0].bar(["TTFT", "Prefill", "Decode", "E2E"], [stats.ttft, stats.prefill_time, stats.decode_time, stats.e2e_time], color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    axes[0].set_title("Latency Breakdown (s)")
    axes[0].set_ylabel("seconds")

    # KV usage and tokens
    kv_pct = stats.kv_usage_perc * 100.0
    axes[1].bar(["KV Usage %"], [kv_pct], color="#E45756")
    axes[1].set_ylim(0, max(100, kv_pct * 1.2 if kv_pct > 0 else 100))
    axes[1].set_title("KV Cache Usage (%)")
    axes[1].text(0, kv_pct, f"{kv_pct:.1f}%", ha="center", va="bottom")

    # Annotate tokens and cache config
    note_lines = [
        f"Prompt tokens: {stats.prompt_tokens}",
        f"Generation tokens: {stats.generation_tokens}",
    ]
    if stats.cache_block_size is not None and stats.cache_num_gpu_blocks is not None:
        note_lines.append(
            f"KV blocks: {stats.cache_num_gpu_blocks}, block_size: {stats.cache_block_size}"
        )
    fig.suptitle("Qwen-VL Video: Latency & KV Cache Usage")
    fig.text(0.5, 0.01, "\n".join(note_lines), ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def main():
    args = parse_args()
    stats = run_once(
        model_name=args.model,
        num_frames=args.num_frames,
        fps=args.fps,
        max_tokens=args.max_tokens,
        mm_cache_gb=args.mm_cache_gb,
        max_model_len=args.max_model_len,
    )

    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"E2E time: {stats.e2e_time:.3f}s")
    print(f"TTFT: {stats.ttft:.3f}s, Prefill: {stats.prefill_time:.3f}s, Decode: {stats.decode_time:.3f}s")
    print(f"Prompt tokens: {stats.prompt_tokens}, Generation tokens: {stats.generation_tokens}")
    print(f"KV usage: {stats.kv_usage_perc*100:.1f}%")
    if stats.cache_block_size is not None and stats.cache_num_gpu_blocks is not None:
        print(
            f"Cache config: blocks={stats.cache_num_gpu_blocks}, block_size={stats.cache_block_size}"
        )

    plot_results(stats, save_path=args.save_fig)


if __name__ == "__main__":
    main()


