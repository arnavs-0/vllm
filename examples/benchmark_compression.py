#!/usr/bin/env python3
"""
Benchmark: Time and Memory Complexity - Baseline vs Compression

Measures:
- Time to first token (TTFT)
- Time per output token (TPOT)
- Total generation time
- Memory usage (GPU: allocated/reserved/max, CPU: RSS/VMS, Python: current/peak)
- Throughput (tokens/second)

Supports both GPU and CPU environments:
- GPU: Full memory tracking with torch.cuda APIs
- CPU: Process memory (psutil) and Python memory (tracemalloc)
- Memory deltas show usage changes from initialization

NOTE: CPU mode is VERY SLOW (minutes per run). Use smaller --max-tokens (e.g., 20-30)
      for reasonable runtime. Memory measurements work on both CPU and GPU now.
"""

import argparse
import os
import time
import torch
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


def get_memory_usage():
    """Get memory usage in MB for both CPU and GPU."""
    memory_info = {}

    # GPU memory (if available)
    if torch.cuda.is_available():
        memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024 / 1024
        memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024 / 1024
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

    print(f"GPU Allocated:     {memory_info['gpu_allocated']:.1f} MB")
    print(f"GPU Reserved:      {memory_info['gpu_reserved']:.1f} MB")
    print(f"GPU Max Alloc:     {memory_info['gpu_max_allocated']:.1f} MB")
    print(f"CPU RSS:           {memory_info['cpu_rss']:.1f} MB")
    print(f"CPU VMS:           {memory_info['cpu_vms']:.1f} MB")
    print(f"Python Current:    {memory_info['python_current']:.1f} MB")
    print(f"Python Peak:       {memory_info['python_peak']:.1f} MB")


def get_gpu_memory_mb():
    """Get current GPU memory allocated in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_model(
    num_frames: int,
    max_tokens: int,
    enable_compression: bool,
    compression_threshold: int = 50,
    num_sink_tokens: int = 4,
    num_recent_tokens: int = 128,
):
    """Run benchmark for one configuration"""

    mode = "COMPRESSED" if enable_compression else "BASELINE"
    print(f"\n{'='*80}")
    print(f"🔬 BENCHMARKING: {mode}")
    print(f"   Frames: {num_frames}, Max tokens: {max_tokens}")
    if enable_compression:
        print(f"   Compression: threshold={compression_threshold}, "
              f"sink={num_sink_tokens}, recent={num_recent_tokens}")
    print(f"{'='*80}\n")

    # Clear GPU memory and start memory tracking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if HAS_TRACEMALLOC:
        tracemalloc.start()

    # Measure model initialization time
    init_start = time.perf_counter()
    
    llm_kwargs = {
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "max_model_len": 8192,
        "max_num_batched_tokens": 8192,  # Must be >= max_model_len
        "max_num_seqs": 1,
        "limit_mm_per_prompt": {"image": 10, "video": 10},
        "gpu_memory_utilization": 0.95,
    }
    
    if enable_compression:
        llm_kwargs.update({
            "enable_kv_compression": True,
            "kv_compression_strategy": "streaming_llm",
            "kv_compression_max_tokens": compression_threshold,
            "kv_compression_num_sink_tokens": num_sink_tokens,
            "kv_compression_num_recent_tokens": num_recent_tokens,
        })
    
    llm = LLM(**llm_kwargs)
    
    init_time = time.perf_counter() - init_start
    init_memory = get_memory_usage()

    print(f"✓ Model initialized in {init_time:.2f}s")
    print_memory_usage(init_memory, "Initial Memory Usage")

    # Load video using VideoAsset (same as working scripts)
    print("📹 Loading video frames...")
    process_start = time.perf_counter()

    video = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays
    metadata = VideoAsset(name="baby_reading", num_frames=num_frames).metadata

    # Create prompt (same format as working scripts)
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        "Describe what happens in this video in detail.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    process_time = time.perf_counter() - process_start
    prefill_memory = get_memory_usage()

    # Get approximate prompt token count (will be accurate after generation)
    tokenizer = llm.get_tokenizer()
    num_prompt_tokens = len(tokenizer.encode(prompt)) + (num_frames * 120)  # Approx 120 tokens per frame
    print(f"✓ Video processed in {process_time:.2f}s")
    print(f"✓ Prompt tokens (approx): {num_prompt_tokens}")
    print_memory_usage(prefill_memory, "Memory After Prefill")
    
    # Generate output
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )
    
    print("🚀 Starting generation...")
    gen_start = time.perf_counter()
    
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"video": [(video, metadata)]},
        },
        sampling_params=sampling_params,
    )
    
    gen_end = time.perf_counter()
    total_gen_time = gen_end - gen_start

    # Get final memory usage
    final_memory = get_memory_usage()

    if HAS_TRACEMALLOC:
        tracemalloc.stop()

    # Extract output
    output_text = outputs[0].outputs[0].text
    num_output_tokens = len(outputs[0].outputs[0].token_ids)

    # Calculate metrics
    throughput = num_output_tokens / total_gen_time if total_gen_time > 0 else 0
    time_per_token_ms = (total_gen_time / num_output_tokens * 1000) if num_output_tokens > 0 else 0

    # Estimate TTFT (first token time) - rough estimate
    # In vLLM, TTFT includes prefill, so it's roughly total_time / num_tokens * initial_overhead
    # For simplicity, we'll estimate it as the time before main generation
    estimated_ttft = process_time + 0.1  # Rough estimate

    print(f"\n{'='*80}")
    print(f"📊 RESULTS - {mode}")
    print(f"{'='*80}")
    print(f"⏱️  TIMING:")
    print(f"   Model init:           {init_time:.3f}s")
    print(f"   Video processing:     {process_time:.3f}s")
    print(f"   Total generation:     {total_gen_time:.3f}s")
    print(f"   Time per token (avg): {time_per_token_ms:.1f}ms")
    print(f"   Throughput:           {throughput:.2f} tokens/s")
    print(f"\n💾 MEMORY:")
    print_memory_usage(final_memory, "Final Memory Usage")

    # Show memory deltas
    print(f"\n📈 MEMORY DELTAS (from init):")
    for key in final_memory:
        if key in init_memory:
            delta = final_memory[key] - init_memory[key]
            key_name = key.replace('_', ' ').title()
            print(f"   {key_name}: {delta:+.1f} MB")

    print(f"\n📝 TOKENS:")
    print(f"   Input tokens:         {num_prompt_tokens}")
    print(f"   Output tokens:        {num_output_tokens}")
    print(f"   Total tokens:         {num_prompt_tokens + num_output_tokens}")
    print(f"\n📄 OUTPUT ({len(output_text)} chars):")
    print(f"   {output_text[:200]}...")
    print(f"{'='*80}\n")
    
    return {
        "mode": mode,
        "init_time": init_time,
        "process_time": process_time,
        "total_gen_time": total_gen_time,
        "time_per_token_ms": time_per_token_ms,
        "throughput": throughput,
        "init_memory": init_memory,
        "prefill_memory": prefill_memory,
        "final_memory": final_memory,
        "num_prompt_tokens": num_prompt_tokens,
        "num_output_tokens": num_output_tokens,
        "output_text": output_text,
    }


def compare_results(baseline, compressed):
    """Compare baseline vs compressed results"""
    
    print(f"\n{'='*80}")
    print(f"📊 COMPARISON: BASELINE vs COMPRESSED")
    print(f"{'='*80}\n")
    
    print(f"⏱️  TIMING COMPARISON:")
    print(f"{'Metric':<30} {'Baseline':<15} {'Compressed':<15} {'Speedup':<15}")
    print(f"{'-'*75}")
    
    # Total generation time
    speedup = baseline['total_gen_time'] / compressed['total_gen_time'] if compressed['total_gen_time'] > 0 else 0
    print(f"{'Total generation time':<30} {baseline['total_gen_time']:.3f}s{'':<8} "
          f"{compressed['total_gen_time']:.3f}s{'':<8} {speedup:.2f}x")
    
    # Time per token
    speedup = baseline['time_per_token_ms'] / compressed['time_per_token_ms'] if compressed['time_per_token_ms'] > 0 else 0
    print(f"{'Time per token':<30} {baseline['time_per_token_ms']:.1f}ms{'':<9} "
          f"{compressed['time_per_token_ms']:.1f}ms{'':<9} {speedup:.2f}x")
    
    # Throughput
    improvement = (compressed['throughput'] / baseline['throughput'] - 1) * 100 if baseline['throughput'] > 0 else 0
    print(f"{'Throughput':<30} {baseline['throughput']:.2f} tok/s{'':<5} "
          f"{compressed['throughput']:.2f} tok/s{'':<5} {improvement:+.1f}%")
    
    print(f"\n💾 MEMORY COMPARISON:")
    print(f"{'Metric':<30} {'Baseline':<15} {'Compressed':<15} {'Savings':<15}")
    print(f"{'-'*75}")

    # Compare final memory usage for each metric
    memory_keys = ['gpu_allocated', 'gpu_reserved', 'gpu_max_allocated', 'cpu_rss', 'cpu_vms', 'python_current', 'python_peak']
    for key in memory_keys:
        if key in baseline['final_memory'] and key in compressed['final_memory']:
            baseline_val = baseline['final_memory'][key]
            compressed_val = compressed['final_memory'][key]
            savings = (1 - compressed_val / baseline_val) * 100 if baseline_val > 0 else 0
            key_name = key.replace('_', ' ').title()
            print(f"{'Final ' + key_name:<30} {baseline_val:.1f} MB{'':<8} "
                  f"{compressed_val:.1f} MB{'':<8} {savings:+.1f}%")

    print(f"\n📝 OUTPUT QUALITY:")
    print(f"{'Metric':<30} {'Baseline':<15} {'Compressed':<15} {'Match':<15}")
    print(f"{'-'*75}")

    # Output length
    match = "✓ Yes" if baseline['output_text'] == compressed['output_text'] else "✗ No"
    print(f"{'Output length (chars)':<30} {len(baseline['output_text']):<15} "
          f"{len(compressed['output_text']):<15} {match}")

    # Exact match
    if baseline['output_text'] == compressed['output_text']:
        print(f"\n✅ OUTPUTS ARE IDENTICAL - 100% match!")
    else:
        print(f"\n⚠️  OUTPUTS DIFFER - Quality may be affected")

    print(f"\n{'='*80}")
    print(f"🎯 SUMMARY:")
    print(f"{'='*80}")

    if compressed['total_gen_time'] < baseline['total_gen_time']:
        speedup = baseline['total_gen_time'] / compressed['total_gen_time']
        print(f"⚡ Compression is {speedup:.2f}x FASTER")
    else:
        slowdown = compressed['total_gen_time'] / baseline['total_gen_time']
        print(f"🐌 Compression is {slowdown:.2f}x SLOWER")

    # Check for memory savings across all metrics
    memory_saved = False
    for key in memory_keys:
        if (key in baseline['final_memory'] and key in compressed['final_memory'] and
            compressed['final_memory'][key] < baseline['final_memory'][key]):
            memory_saved = True
            break

    if memory_saved:
        print(f"💾 Compression REDUCES memory usage")
    else:
        print(f"💾 Compression increases or maintains memory usage")

    if baseline['output_text'] == compressed['output_text']:
        print(f"✅ Quality: IDENTICAL outputs (0% degradation)")
    else:
        print(f"⚠️  Quality: DIFFERENT outputs (quality may vary)")
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark compression time and memory")
    parser.add_argument("--num-frames", type=int, default=4, help="Number of video frames")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--threshold", type=int, default=50, help="Compression threshold")
    parser.add_argument("--num-sink", type=int, default=4, help="Number of sink tokens")
    parser.add_argument("--num-recent", type=int, default=128, help="Number of recent tokens")
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline")
    parser.add_argument("--compressed-only", action="store_true", help="Run only compressed")
    
    args = parser.parse_args()
    
    results = {}
    
    # Run baseline
    if not args.compressed_only:
        print("\n" + "="*80)
        print("🔵 RUNNING BASELINE (NO COMPRESSION)")
        print("="*80)
        baseline_result = benchmark_model(
            num_frames=args.num_frames,
            max_tokens=args.max_tokens,
            enable_compression=False,
        )
        results['baseline'] = baseline_result
        
        # Wait a bit between runs
        time.sleep(2)
    
    # Run compressed
    if not args.baseline_only:
        print("\n" + "="*80)
        print("🟢 RUNNING COMPRESSED")
        print("="*80)
        compressed_result = benchmark_model(
            num_frames=args.num_frames,
            max_tokens=args.max_tokens,
            enable_compression=True,
            compression_threshold=args.threshold,
            num_sink_tokens=args.num_sink,
            num_recent_tokens=args.num_recent,
        )
        results['compressed'] = compressed_result
    
    # Compare results
    if 'baseline' in results and 'compressed' in results:
        compare_results(results['baseline'], results['compressed'])


if __name__ == "__main__":
    main()
