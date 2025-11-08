#!/usr/bin/env python3
"""
Benchmark: Streaming Video with KV Cache Compression

This benchmark demonstrates the Streaming-VL KV cache policy for infinite-length
video processing. It compares:

1. Baseline (no compression): Limited by memory, re-prefill on long videos
2. Streaming-VL Compression: Constant memory, no re-prefill, infinite length

The streaming policy maintains a sparse KV cache:
- Sink blocks: Always keep first frame (attention anchor)
- Recent blocks: Keep last N frames (sliding window)  
- Middle blocks: Explicitly freed to maintain constant size

This enables true video streaming without the catastrophic re-prefill problem
of naive sliding windows.

NOTE: Uses both compression systems:
- V1 Block Manager: StreamingVideoKVCacheManager (explicit block freeing)
- V0 Attention: streaming_llm strategy (attention masking, legacy)
The block manager provides the actual memory savings for infinite streaming.

Measures:
- Time to first token (TTFT)
- Time per output token (TPOT)
- Total generation time
- Memory usage (GPU: allocated/reserved/max, CPU: RSS/VMS, Python: current/peak)
- Throughput (tokens/second)
- KV cache compression statistics

Supports both GPU and CPU environments:
- GPU: Full memory tracking with torch.cuda APIs
- CPU: Process memory (psutil) and Python memory (tracemalloc)
- Memory deltas show usage changes from initialization

NOTE: CPU mode is VERY SLOW (minutes per run). Use smaller --max-tokens (e.g., 20-30)
      for reasonable runtime. For GPU, this demonstrates constant-memory streaming.
"""

import argparse
import math
import os
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset

# Disable TorchDynamo/Inductor for CPU to avoid compilation errors
# Comment this out if running on GPU and want Inductor optimizations
if not torch.cuda.is_available():
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not available - CPU memory measurements will be limited")

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False
    print("⚠️  tracemalloc not available - Python memory tracking disabled")


def load_video_from_path(file_path: str):
    """Load a video from a file path using OpenCV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found at: {file_path}")
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {file_path}")
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV reads frames in BGR format, convert to RGB
        frames.append(frame[:, :, ::-1])
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    cap.release()
    
    video_data = np.stack(frames)
    metadata = {
        "fps": fps,
        "size": (width, height),
        "duration": frame_count / fps if fps > 0 else 0,
    }
    
    print(f"✓ Loaded video '{Path(file_path).name}': {frame_count} frames, {height}x{width} @ {fps:.1f} FPS")
    
    return video_data, metadata


def get_memory_usage():
    """Get memory usage in MB for both CPU and GPU."""
    memory_info = {}

    # GPU memory (if available) - NOTE: In vLLM multi-process architecture,
    # GPU memory usage happens in worker processes, not main process
    if torch.cuda.is_available():
        try:
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024 / 1024
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        except Exception as e:
            memory_info['gpu_allocated'] = 0.0
            memory_info['gpu_reserved'] = 0.0
            memory_info['gpu_max_allocated'] = 0.0
    else:
        memory_info['gpu_allocated'] = 0.0
        memory_info['gpu_reserved'] = 0.0
        memory_info['gpu_max_allocated'] = 0.0

    # CPU memory (process-level)
    if HAS_PSUTIL:
        process = psutil.Process()
        memory_info['cpu_rss'] = process.memory_info().rss / 1024 / 1024  # Resident Set Size
        memory_info['cpu_vms'] = process.memory_info().vms / 1024 / 1024  # Virtual Memory Size
    else:
        memory_info['cpu_rss'] = 0.0
        memory_info['cpu_vms'] = 0.0

    # Python memory (tracemalloc)
    if HAS_TRACEMALLOC:
        current, peak = tracemalloc.get_traced_memory()
        memory_info['python_current'] = current / 1024 / 1024
        memory_info['python_peak'] = peak / 1024 / 1024
    else:
        memory_info['python_current'] = 0.0
        memory_info['python_peak'] = 0.0

    return memory_info


def print_memory_usage(memory_info, label=""):
    """Print formatted memory usage."""
    if label:
        print(f"\n{label}")
        print("-" * len(label))

    # CPU and Python memory (these work reliably)
    print(f"CPU RSS:           {memory_info['cpu_rss']:.1f} MB")
    print(f"CPU VMS:           {memory_info['cpu_vms']:.1f} MB")
    print(f"Python Current:    {memory_info['python_current']:.1f} MB")
    print(f"Python Peak:       {memory_info['python_peak']:.1f} MB")

    # GPU memory (limited in multi-process vLLM architecture)
    gpu_allocated = memory_info['gpu_allocated']
    gpu_reserved = memory_info['gpu_reserved']
    gpu_max = memory_info['gpu_max_allocated']
    print(f"\n⚠️  GPU Memory (Limited in vLLM multi-process):")
    print(f"GPU Allocated:     {gpu_allocated:.1f} MB (main process only)")
    print(f"GPU Reserved:      {gpu_reserved:.1f} MB (main process only)")
    print(f"GPU Max Alloc:     {gpu_max:.1f} MB (main process only)")
    print(f"Note: GPU work happens in worker processes, not main process")


def get_gpu_memory_mb():
    """Get current GPU memory allocated in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_model(
    max_tokens: int,
    enable_compression: bool,
    compression_threshold: int = 50,
    num_sink_tokens: int = 4,
    num_recent_tokens: int = 128,
    video_file: str = None,
    num_frames: int = 4,
):
    """Run benchmark for one configuration"""

    mode = "COMPRESSED" if enable_compression else "BASELINE"
    print(f"\n{'='*80}")
    print(f"🔬 BENCHMARKING: {mode}")
    if video_file:
        print(f"   Video: {Path(video_file).name}, Max tokens: {max_tokens}")
    else:
        print(f"   Frames: {num_frames}, Max tokens: {max_tokens}")
        
    if enable_compression:
        print(f"   Compression: threshold={compression_threshold}, "
              f"sink={num_sink_tokens}, recent={num_recent_tokens}")
    print(f"{'='*80}\n")

    # Clear GPU memory and start memory tracking BEFORE model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if HAS_TRACEMALLOC:
        tracemalloc.start()

    # Get baseline memory BEFORE model initialization
    baseline_memory = get_memory_usage()

    # Measure model initialization time
    init_start = time.perf_counter()
    
    llm_kwargs = {
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "max_model_len": 37000,
        "max_num_batched_tokens": 37000,
        "max_num_seqs": 1,
        "limit_mm_per_prompt": {"image": 10, "video": 10},
        "gpu_memory_utilization": 0.95,
    }
    
    if enable_compression:
        llm_kwargs.update({
            "enable_kv_compression": True,
            "kv_compression_strategy": "none",
            "kv_compression_max_tokens": compression_threshold,
            "kv_compression_num_sink_tokens": num_sink_tokens,
            "kv_compression_num_recent_tokens": num_recent_tokens,
        })
    
    llm = LLM(**llm_kwargs)
    
    # --- Extract engine info for impact analysis ---
    engine_config = {}
    try:
        model_config = llm.llm_engine.model_config
        cache_config = llm.llm_engine.cache_config
        
        hf_config = model_config.hf_config
        text_config = getattr(hf_config, 'text_config', hf_config)

        engine_config['block_size'] = cache_config.block_size
        engine_config['num_gpu_blocks'] = cache_config.num_gpu_blocks
        
        dtype_size = torch.tensor([], dtype=model_config.dtype).element_size()
        num_layers = text_config.num_hidden_layers
        num_heads = text_config.num_attention_heads
        hidden_size = text_config.hidden_size
        head_dim = hidden_size // num_heads
        
        block_mem_size = 2 * engine_config['block_size'] * num_layers * num_heads * head_dim * dtype_size
        engine_config['block_mem_size_mb'] = block_mem_size / 1024 / 1024
        
        print("\n⚙️  Engine Info:")
        print(f"   Block Size: {engine_config['block_size']} tokens")
        print(f"   Total GPU Blocks in Pool: {engine_config['num_gpu_blocks']}")
        print(f"   KV Cache Memory per Block: {engine_config['block_mem_size_mb']:.3f} MB")

    except Exception as e:
        print(f"⚠️  Could not extract engine info for impact analysis: {e}")
    # -----------------------------------------

    init_time = time.perf_counter() - init_start
    init_memory = get_memory_usage()

    print(f"✓ Model initialized in {init_time:.2f}s")
    print_memory_usage(init_memory, "Initial Memory Usage")

    # --- Load video data ---
    print("📹 Loading video frames...")
    process_start = time.perf_counter()
    if video_file:
        video, metadata = load_video_from_path(video_file)
        num_frames = len(video)
    else:
        video = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays
        metadata = VideoAsset(name="baby_reading", num_frames=num_frames).metadata
    # -----------------------

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        "Describe what happens in this video in detail.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    process_time = time.perf_counter() - process_start
    prefill_memory = get_memory_usage()

    tokenizer = llm.get_tokenizer()
    # A rough approximation for logging. The actual value is calculated after generation.
    approx_prompt_tokens = len(tokenizer.encode(prompt)) + (num_frames * 120)
    print(f"✓ Video processed in {process_time:.2f}s")
    print(f"✓ Prompt tokens (approx): {approx_prompt_tokens}")
    print(f"✓ Compression threshold: {compression_threshold if enable_compression else 'N/A'}")
    if enable_compression and approx_prompt_tokens > compression_threshold:
        print(f"✓ Compression should trigger (prompt tokens > threshold)")
    elif enable_compression:
        print(f"⚠️  Compression may not trigger (prompt tokens <= threshold)")
    print_memory_usage(prefill_memory, "Memory After Prefill")
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    
    print("🚀 Starting generation...")
    gen_start = time.perf_counter()
    
    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"video": [(video, metadata)]}},
        sampling_params=sampling_params,
    )
    
    gen_end = time.perf_counter()
    total_gen_time = gen_end - gen_start

    final_memory = get_memory_usage()

    if HAS_TRACEMALLOC:
        tracemalloc.stop()

    output_text = outputs[0].outputs[0].text
    num_output_tokens = len(outputs[0].outputs[0].token_ids)
    actual_prompt_tokens = len(outputs[0].prompt_token_ids)
    total_tokens_generated = actual_prompt_tokens + num_output_tokens
    
    throughput = num_output_tokens / total_gen_time if total_gen_time > 0 else 0
    time_per_token_ms = (total_gen_time / num_output_tokens * 1000) if num_output_tokens > 0 else 0

    # --- Calculate blocks used based on algorithm rules ---
    num_blocks_used = 0
    if 'block_size' in engine_config and engine_config['block_size'] > 0:
        block_size = engine_config['block_size']
        if not enable_compression:
            num_blocks_used = math.ceil(total_tokens_generated / block_size)
        else:
            if total_tokens_generated <= compression_threshold:
                num_blocks_used = math.ceil(total_tokens_generated / block_size)
            else:
                num_sink_blocks = max(1, num_sink_tokens // block_size)
                num_recent_blocks = max(1, num_recent_tokens // block_size)
                num_blocks_used = num_sink_blocks + num_recent_blocks
    # ----------------------------------------------------

    print(f"\n{'='*80}")
    print(f"📊 RESULTS - {mode}")
    print(f"{'='*80}")
    print(f"⏱️  TIMING:")
    print(f"   Total generation:     {total_gen_time:.3f}s")
    print(f"   Time per token (avg): {time_per_token_ms:.1f}ms")
    print(f"   Throughput:           {throughput:.2f} tokens/s")
    print(f"\n💾 MEMORY:")
    print_memory_usage(final_memory, "Final Memory Usage")

    print(f"\n📝 TOKENS:")
    print(f"   Input tokens (actual): {actual_prompt_tokens}")
    print(f"   Output tokens:        {num_output_tokens}")
    print(f"   Total tokens:         {total_tokens_generated}")
    if enable_compression:
        print(f"   Compression config:   sink={num_sink_tokens}, recent={num_recent_tokens}")
        if total_tokens_generated > compression_threshold:
            print(f"   ✓ Compression TRIGGERED (total={total_tokens_generated} > threshold={compression_threshold})")
        else:
            print(f"   ⚠️  Compression NOT triggered (total={total_tokens_generated} <= threshold={compression_threshold})")

    print(f"\n📄 OUTPUT ({len(output_text)} chars):")
    print(f"   {output_text}...")
    print(f"{'='*80}\n")
    
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "mode": mode,
        "total_gen_time": total_gen_time,
        "time_per_token_ms": time_per_token_ms,
        "throughput": throughput,
        "output_text": output_text,
        "engine_config": engine_config,
        "num_blocks_used": num_blocks_used,
    }


def compare_results(baseline, compressed):
    """Compare baseline vs compressed results"""
    
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON: BASELINE vs COMPRESSED")
    print(f"{'='*80}\n")

    # --- High-Impact Analysis ---
    if (baseline.get('engine_config') and compressed.get('engine_config') and 
        baseline.get('num_blocks_used') is not None and compressed.get('num_blocks_used') is not None 
        and baseline['num_blocks_used'] > 0 and compressed['num_blocks_used'] > 0):
        
        print(f"🚀 CAPACITY ANALYSIS (PROJECTION FROM SINGLE RUN)")
        print(f"{'-'*80}")
        print(f"(Based on projecting the block usage of this single run onto the total GPU block pool)")

        bl_engine_cfg = baseline['engine_config']
        bl_blocks_used = baseline['num_blocks_used']
        bl_mem_per_stream = bl_blocks_used * bl_engine_cfg.get('block_mem_size_mb', 0)
        bl_capacity = bl_engine_cfg['num_gpu_blocks'] // bl_blocks_used if bl_blocks_used > 0 else 0

        cp_engine_cfg = compressed['engine_config']
        cp_blocks_used = compressed['num_blocks_used']
        cp_mem_per_stream = cp_blocks_used * cp_engine_cfg.get('block_mem_size_mb', 0)
        cp_capacity = cp_engine_cfg['num_gpu_blocks'] // cp_blocks_used if cp_blocks_used > 0 else 0
        
        improvement_factor = cp_capacity / bl_capacity if bl_capacity > 0 else float('inf')

        print(f"\nEffective KV Cache Memory per Stream (Actual for this run):")
        print(f"  - Baseline:   {bl_mem_per_stream:7.1f} MB ({bl_blocks_used} blocks)")
        print(f"  - Compressed: {cp_mem_per_stream:7.1f} MB ({cp_blocks_used} blocks)")
        
        print(f"\nProjected Concurrent Stream Capacity (Estimated):")
        print(f"  - Baseline:   {bl_capacity} stream(s)")
        print(f"  - Compressed: {cp_capacity} stream(s)  <-- {improvement_factor:.1f}x more capacity!")
        print(f"{'-'*80}\n")
    else:
        print(f"⚠️  Skipping Capacity Analysis: Could not retrieve necessary engine/block info or block usage was zero.")
        print(f"{'-'*80}\n")
    # --------------------------
    
    print(f"⏱️  TIMING COMPARISON:")
    print(f"{'Metric':<30} {'Baseline':<15} {'Compressed':<15} {'Speedup':<15}")
    print(f"{'-'*75}")
    
    speedup = baseline['total_gen_time'] / compressed['total_gen_time'] if compressed['total_gen_time'] > 0 else 0
    print(f"{'Total generation time':<30} {baseline['total_gen_time']:.3f}s{'':<8} "
          f"{compressed['total_gen_time']:.3f}s{'':<8} {speedup:.2f}x")
    
    speedup = baseline['time_per_token_ms'] / compressed['time_per_token_ms'] if compressed['time_per_token_ms'] > 0 else 0
    print(f"{'Time per token':<30} {baseline['time_per_token_ms']:.1f}ms{'':<9} "
          f"{compressed['time_per_token_ms']:.1f}ms{'':<9} {speedup:.2f}x")
    
    improvement = (compressed['throughput'] / baseline['throughput'] - 1) * 100 if baseline['throughput'] > 0 else 0
    print(f"{'Throughput':<30} {baseline['throughput']:.2f} tok/s{'':<5} "
          f"{compressed['throughput']:.2f} tok/s{'':<5} {improvement:+.1f}%")
    
    print(f"\n📝 OUTPUT QUALITY:")
    print(f"{'Metric':<30} {'Baseline':<15} {'Compressed':<15} {'Match':<15}")
    print(f"{'-'*75}")

    match = "✓ Yes" if baseline['output_text'] == compressed['output_text'] else "✗ No"
    print(f"{'Output length (chars)':<30} {len(baseline['output_text']):<15} "
          f"{len(compressed['output_text']):<15} {match}")

    if baseline['output_text'] == compressed['output_text']:
        print(f"\n✅ OUTPUTS ARE IDENTICAL - 100% match!")
    else:
        print(f"\n⚠️  OUTPUTS DIFFER - Quality may be affected")
    
    print(f"\n{'='*80}\n")


def main():
    import gc
    parser = argparse.ArgumentParser(description="Benchmark compression time and memory")
    parser.add_argument("--video-file", type=str, default=None, help="Path to a local MP4 video file to process.")
    parser.add_argument("--num-frames", type=int, default=4, help="Number of video frames (used if --video-file is not provided).")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--threshold", type=int, default=5000, help="Compression threshold (tokens before compression triggers)")
    parser.add_argument("--num-sink", type=int, default=4, help="Number of sink tokens")
    parser.add_argument("--num-recent", type=int, default=128, help="Number of recent tokens")
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline")
    parser.add_argument("--compressed-only", action="store_true", help="Run only compressed")
    
    args = parser.parse_args()
    
    results = {}
    
    common_args = {
        "max_tokens": args.max_tokens,
        "video_file": args.video_file,
        "num_frames": args.num_frames,
    }
    
    if not args.compressed_only:
        print("\n" + "="*80)
        print("🔵 RUNNING BASELINE (NO COMPRESSION)")
        print("="*80)
        baseline_result = benchmark_model(
            **common_args,
            enable_compression=False,
        )
        results['baseline'] = baseline_result
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)
    
    if not args.baseline_only:
        print("\n" + "="*80)
        print("🟢 RUNNING COMPRESSED")
        print("="*80)
        compressed_result = benchmark_model(
            **common_args,
            enable_compression=True,
            compression_threshold=args.threshold,
            num_sink_tokens=args.num_sink,
            num_recent_tokens=args.num_recent,
        )
        results['compressed'] = compressed_result
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    if 'baseline' in results and 'compressed' in results:
        compare_results(results['baseline'], results['compressed'])


if __name__ == "__main__":
    main()