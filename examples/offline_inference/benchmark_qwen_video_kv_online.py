#!/usr/bin/env python3
"""
Benchmark streaming (online) KV cache usage for a video sent to Qwen-VL with vLLM.

This script uses AsyncLLM streaming (DELTA mode) to:
- Build one video input (frames + metadata)
- Stream tokens and align KV cache usage per generated token
- Poll KV usage during prefill to plot usage vs. frame (approximate)
- Report latency components and peak KV usage (overall and decode-only)

Examples:
  python examples/offline_inference/benchmark_qwen_video_kv_online.py \
    --model Qwen/Qwen3-VL-4B-Instruct --num-frames 16 --max-tokens 64 --fps 1
"""

import argparse
import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm import SamplingParams
from vllm.v1.metrics.reader import Counter, Gauge, Histogram, Metric, get_metrics_snapshot

# Reuse helpers from examples/offline_inference/vision_language.py
try:
    from examples.offline_inference.vision_language import (
        get_multi_modal_input,
        run_qwen3_vl,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import helpers from examples/offline_inference.vision_language."
    ) from e


@dataclass
class RunStats:
    e2e_time: float
    prefill_time: float
    decode_time: float
    ttft: float
    kv_usage_perc_last: float
    peak_kv_usage_perc: float
    peak_kv_usage_perc_decode: float
    prompt_tokens: int
    generation_tokens: int
    cache_block_size: Optional[int]
    cache_num_gpu_blocks: Optional[int]
    kv_usage_samples: List[Tuple[float, float]]  # (t_rel_sec, usage_frac)
    kv_usage_per_token: List[float]  # usage % per generated token index
    fps: float
    num_frames: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Streaming benchmark of video latency and KV cache usage for Qwen-VL"
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


def get_hist_value_delta(pre: List[Metric], post: List[Metric], metric_name: str) -> float:
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
                return (
                    int(block_size) if block_size is not None else None,
                    int(num_gpu_blocks) if num_gpu_blocks is not None else None,
                )
            except Exception:
                return (None, None)
    return (None, None)


def get_latest_kv_usage(metrics: List[Metric]) -> Optional[float]:
    return get_latest_gauge_value(metrics, "vllm:kv_cache_usage_perc")


async def build_input_for_qwen_video_async(model_name: str, num_frames: int, fps: float):
    # Build EngineArgs + prompts via run_qwen3_vl, then adapt to AsyncEngineArgs
    req_data = run_qwen3_vl(["Describe this video."], "video")
    # Override model/fps
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
        "multi_modal_uuids": {"video": "vid_uuid_0"},
    }

    # Convert EngineArgs -> AsyncEngineArgs (they share fields)
    engine_args = req_data.engine_args
    # Ensure required overrides
    engine_args.limit_mm_per_prompt = {"video": 1}
    aengine_args = AsyncEngineArgs(**engine_args.__dict__)
    return aengine_args, inputs


def monitor_kv_usage_sync(stop_evt: threading.Event, start_time: float, samples: List[Tuple[float, float]], interval_s: float = 0.1) -> None:
    while not stop_evt.is_set():
        try:
            ms = get_metrics_snapshot()
            val = get_latest_kv_usage(ms)
            if val is not None:
                samples.append((time.time() - start_time, float(val)))
        except Exception:
            pass
        time.sleep(interval_s)


async def run_once_streaming(
    model_name: str,
    num_frames: int,
    fps: float,
    max_tokens: int,
    mm_cache_gb: float,
    max_model_len: int,
) -> RunStats:
    aengine_args, inputs = await build_input_for_qwen_video_async(model_name, num_frames, fps)

    # Enable/disable MM processor cache and max_model_len
    aengine_args.mm_processor_cache_gb = max(0.0, mm_cache_gb)
    try:
        current = getattr(aengine_args, "max_model_len", None) or 0
        aengine_args.max_model_len = max(current, int(max_model_len))
    except Exception:
        aengine_args.max_model_len = int(max_model_len)

    engine = AsyncLLM.from_engine_args(aengine_args)
    tokenizer = await engine.get_tokenizer()

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        output_kind=RequestOutputKind.DELTA,
    )

    # Metrics snapshot BEFORE
    pre_metrics = get_metrics_snapshot()

    # Prefill sampling thread (poll kv usage)
    samples: List[Tuple[float, float]] = []
    t0 = time.time()
    stop_evt = threading.Event()
    sampler = threading.Thread(target=monitor_kv_usage_sync, args=(stop_evt, t0, samples), daemon=True)
    sampler.start()

    # Streaming loop
    request_id = "stream-req-0"
    ttft = None
    gen_token_usage: List[float] = []  # % usage per token index (decode)
    gen_token_count = 0

    async for output in engine.generate(inputs, sampling_params, request_id):
        if ttft is None:
            # First time we receive any output
            ttft = time.time() - t0
            # Stop prefill-only sampling? Keep sampling overall; we still want decode samples
            # We'll keep the sampler running until end

        # For each new delta, grab KV usage now and attribute to tokens
        ms_now = get_metrics_snapshot()
        usage_now = get_latest_kv_usage(ms_now) or 0.0
        for completion in output.outputs:
            delta_text = completion.text or ""
            if not delta_text:
                continue
            # Estimate token count using tokenizer
            try:
                delta_tokens = tokenizer.encode(delta_text, add_special_tokens=False)  # type: ignore[attr-defined]
                num_new = len(delta_tokens)
            except Exception:
                # Fallback: count as 1 token
                num_new = 1
            gen_token_usage.extend([usage_now * 100.0] * num_new)
            gen_token_count += num_new

        if output.finished:
            break

    e2e_time = time.time() - t0

    # Stop sampler and take AFTER metrics
    stop_evt.set()
    sampler.join(timeout=1.0)
    post_metrics = get_metrics_snapshot()

    # Latency components from histograms
    ttft_hist = get_hist_value_delta(pre_metrics, post_metrics, "vllm:time_to_first_token_seconds")
    prefill_time = get_hist_value_delta(pre_metrics, post_metrics, "vllm:request_prefill_time_seconds")
    decode_time = get_hist_value_delta(pre_metrics, post_metrics, "vllm:request_decode_time_seconds")

    # Use observed ttft if present, else histogram value
    if ttft is None:
        ttft = ttft_hist

    # Token counters
    prompt_tokens = get_counter_delta(pre_metrics, post_metrics, "vllm:prompt_tokens")
    generation_tokens = get_counter_delta(pre_metrics, post_metrics, "vllm:generation_tokens")

    # KV usage last and peaks
    kv_usage_last = get_latest_kv_usage(post_metrics) or 0.0
    peak_overall = max([v for _, v in samples], default=kv_usage_last)
    peak_decode = max([v for t, v in samples if t > prefill_time], default=peak_overall)

    block_size, num_gpu_blocks = extract_cache_config_info(post_metrics)

    # Cleanup engine
    engine.shutdown()

    return RunStats(
        e2e_time=e2e_time,
        prefill_time=prefill_time,
        decode_time=decode_time,
        ttft=ttft,
        kv_usage_perc_last=kv_usage_last,
        peak_kv_usage_perc=peak_overall,
        peak_kv_usage_perc_decode=peak_decode,
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        cache_block_size=block_size,
        cache_num_gpu_blocks=num_gpu_blocks,
        kv_usage_samples=samples,
        kv_usage_per_token=gen_token_usage,
        fps=fps,
        num_frames=num_frames,
    )


def plot_results(stats: RunStats, save_path: Optional[str] = None) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Latency breakdown
    axes[0].bar(["TTFT", "Prefill", "Decode", "E2E"], [stats.ttft, stats.prefill_time, stats.decode_time, stats.e2e_time], color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    axes[0].set_title("Latency Breakdown (s)")
    axes[0].set_ylabel("seconds")

    # KV usage last vs peak
    last_pct = stats.kv_usage_perc_last * 100.0
    peak_pct = stats.peak_kv_usage_perc * 100.0
    axes[1].bar(["Last", "Peak", "Peak-Decode"], [last_pct, peak_pct, stats.peak_kv_usage_perc_decode * 100.0], color=["#E45756", "#F1A340", "#2CA02C"])
    axes[1].set_ylim(0, max(100, max(last_pct, peak_pct, stats.peak_kv_usage_perc_decode * 100.0) * 1.2))
    axes[1].set_title("KV Cache Usage (%)")
    for i, v in enumerate([last_pct, peak_pct, stats.peak_kv_usage_perc_decode * 100.0]):
        axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom")

    # KV usage vs frame (prefill)
    if stats.num_frames > 0 and stats.fps > 0 and stats.prefill_time > 0 and stats.kv_usage_samples:
        per_frame: List[float] = [0.0] * stats.num_frames
        has_sample: List[bool] = [False] * stats.num_frames
        for t_rel, frac in stats.kv_usage_samples:
            if t_rel < 0 or t_rel > stats.prefill_time:
                continue
            fi = int(t_rel * stats.fps)
            if fi >= stats.num_frames:
                fi = stats.num_frames - 1
            val = frac * 100.0
            if not has_sample[fi] or val > per_frame[fi]:
                per_frame[fi] = val
                has_sample[fi] = True
        xs = list(range(stats.num_frames))
        axes[2].plot(xs, per_frame, marker="o", color="#4C78A8")
        axes[2].set_xlabel("Frame index")
        axes[2].set_ylabel("KV Usage (%)")
        axes[2].set_title("KV Usage vs Frame (prefill phase)")
        axes[2].set_xlim(-0.5, stats.num_frames - 0.5)
        axes[2].set_ylim(0, max(100, (max(per_frame) if per_frame else 0) * 1.2))
    else:
        axes[2].axis("off")

    # KV usage vs generated token (decode)
    if stats.kv_usage_per_token:
        xs = list(range(len(stats.kv_usage_per_token)))
        axes[3].plot(xs, stats.kv_usage_per_token, marker=".", linestyle="-", color="#2CA02C")
        axes[3].set_xlabel("Generated token index")
        axes[3].set_ylabel("KV Usage (%)")
        axes[3].set_title("KV Usage vs Token (decode phase)")
        axes[3].set_xlim(-0.5, len(xs) - 0.5 if xs else 0.5)
        axes[3].set_ylim(0, max(100, (max(stats.kv_usage_per_token) if stats.kv_usage_per_token else 0) * 1.2))
    else:
        axes[3].axis("off")

    # Notes
    note_lines = [
        f"Prompt tokens: {stats.prompt_tokens}",
        f"Generation tokens: {stats.generation_tokens}",
    ]
    if stats.cache_block_size is not None and stats.cache_num_gpu_blocks is not None:
        note_lines.append(
            f"KV blocks: {stats.cache_num_gpu_blocks}, block_size: {stats.cache_block_size}"
        )
    fig.suptitle("Qwen-VL Video (Streaming): Latency & KV Cache Usage")
    fig.text(0.5, 0.01, "\n".join(note_lines), ha="center", fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


async def amain():
    args = parse_args()
    stats = await run_once_streaming(
        model_name=args.model,
        num_frames=args.num_frames,
        fps=args.fps,
        max_tokens=args.max_tokens,
        mm_cache_gb=args.mm_cache_gb,
        max_model_len=args.max_model_len,
    )

    print("-" * 60)
    print("RESULTS (Streaming)")
    print("-" * 60)
    print(f"E2E time: {stats.e2e_time:.3f}s")
    print(f"TTFT: {stats.ttft:.3f}s, Prefill: {stats.prefill_time:.3f}s, Decode: {stats.decode_time:.3f}s")
    print(f"Prompt tokens: {stats.prompt_tokens}, Generation tokens: {stats.generation_tokens}")
    print(f"KV usage last: {stats.kv_usage_perc_last*100:.1f}% | peak: {stats.peak_kv_usage_perc*100:.1f}% | peak(decode): {stats.peak_kv_usage_perc_decode*100:.1f}%")
    if stats.cache_block_size is not None and stats.cache_num_gpu_blocks is not None:
        print(
            f"Cache config: blocks={stats.cache_num_gpu_blocks}, block_size={stats.cache_block_size}"
        )

    plot_results(stats, save_path=args.save_fig)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()


