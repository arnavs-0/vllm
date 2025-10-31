#!/usr/bin/env python3
"""
Test script for temporal frame batching hypothesis
Tests VLLM single vs batch processing on CUDA/MPS/CPU
"""

import argparse
import gc
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.multimodal.image import convert_image_mode


@dataclass
class BenchmarkResult:
    scenario: str
    num_requests: int
    total_time: float
    avg_latency: float
    throughput_tokens_per_sec: float
    total_tokens: int
    
    def __str__(self):
        return f"""
{'='*60}
Scenario: {self.scenario}
{'='*60}
Requests: {self.num_requests}
Total time: {self.total_time:.2f}s
Avg latency: {self.avg_latency:.2f}s/request
Throughput: {self.throughput_tokens_per_sec:.2f} tokens/sec
Total tokens: {self.total_tokens}
{'='*60}
"""


def create_temporal_frames(base_image: Image.Image, num_frames: int) -> List[Image.Image]:
    """Create variations of base image to simulate temporal frames"""
    frames = []
    base_array = np.array(base_image)
    
    for i in range(num_frames):
        # Add slight variations
        noise = np.random.normal(0, 5, base_array.shape)
        frame_array = np.clip(base_array + noise, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame_array))
    
    return frames


def benchmark_vllm_single(num_requests: int, max_tokens: int = 32, model_name: str = "Qwen/Qwen3-VL-4B-Instruct") -> BenchmarkResult:
    """VLLM with single requests (no batching)"""
    print("\nRunning VLLM (single request mode)...")
    print(f"  Model: {model_name}")
    
    # Configure based on model
    if "gemma" in model_name.lower():
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            max_num_seqs=1,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
            mm_processor_cache_gb=0,
        )
    else:
        llm = LLM(
            model=model_name,
            max_model_len=4096,
            max_num_seqs=1,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
            mm_processor_cache_gb=0,
        )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    
    base_image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    frames = create_temporal_frames(base_image, num_requests)
    
    question = "What is in this image?"
    
    # Use model-specific prompt format
    if "gemma" in model_name.lower():
        prompt = (
            "<start_of_turn>user\n"
            f"<image_soft_token>{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    else:
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n<|im_start|>assistant\n"
        )
    
    total_tokens = 0
    request_times = []
    
    start_time = time.time()
    
    for i, frame in enumerate(frames):
        req_start = time.time()
        
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": frame},
            "multi_modal_uuids": {"image": f"uuid_{i}"},
        }
        
        outputs = llm.generate([inputs], sampling_params=sampling_params)
        total_tokens += len(outputs[0].outputs[0].token_ids)
        
        req_end = time.time()
        request_times.append(req_end - req_start)
        print(f"  Request {i+1}/{num_requests}: {req_end - req_start:.2f}s")
    
    total_time = time.time() - start_time
    
    del llm
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return BenchmarkResult(
        scenario="VLLM (single)",
        num_requests=num_requests,
        total_time=total_time,
        avg_latency=np.mean(request_times),
        throughput_tokens_per_sec=total_tokens / total_time,
        total_tokens=total_tokens,
    )


def benchmark_vllm_batch(num_requests: int, max_tokens: int = 32, model_name: str = "Qwen/Qwen3-VL-4B-Instruct") -> BenchmarkResult:
    """VLLM with batch processing (THE HYPOTHESIS)"""
    print("\nRunning VLLM (batch mode - THE HYPOTHESIS)...")
    print(f"  Model: {model_name}")
    
    # Configure based on model
    if "gemma" in model_name.lower():
        llm = LLM(
            model=model_name,
            max_model_len=2048,
            max_num_seqs=num_requests,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
            mm_processor_cache_gb=0,
        )
    else:
        llm = LLM(
            model=model_name,
            max_model_len=4096,
            max_num_seqs=num_requests,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
            mm_processor_cache_gb=0,
        )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    
    base_image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    frames = create_temporal_frames(base_image, num_requests)
    
    question = "What is in this image?"
    
    # Use model-specific prompt format
    if "gemma" in model_name.lower():
        prompt_template = (
            "<start_of_turn>user\n"
            f"<image_soft_token>{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    else:
        prompt_template = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question}<|im_end|>\n<|im_start|>assistant\n"
        )
    
    # Create batch of inputs
    inputs = []
    for i, frame in enumerate(frames):
        inputs.append({
            "prompt": prompt_template,
            "multi_modal_data": {"image": frame},
            "multi_modal_uuids": {"image": f"uuid_{i}"},
        })
    
    print(f"  Processing {len(inputs)} frames in batch...")
    start_time = time.time()
    
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    total_time = time.time() - start_time
    
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    
    del llm
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return BenchmarkResult(
        scenario="VLLM (batch)",
        num_requests=num_requests,
        total_time=total_time,
        avg_latency=total_time / num_requests,
        throughput_tokens_per_sec=total_tokens / total_time,
        total_tokens=total_tokens,
    )


def main():
    parser = argparse.ArgumentParser(description="Temporal frame batching hypothesis test")
    parser.add_argument("--num-frames", type=int, default=4, help="Number of frames (default: 4)")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens (default: 32)")
    parser.add_argument("--model", type=str, default="google/gemma-3n-E2B-it", 
                       help="Model to use (default: google/gemma-3n-E2B-it)")
    
    args = parser.parse_args()
    
    # Detect device
    if torch.cuda.is_available():
        device_info = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device_info = "MPS (Apple Silicon)"
    else:
        device_info = "CPU"
    
    print("="*60)
    print("TEMPORAL FRAME BATCHING HYPOTHESIS TEST")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {device_info}")
    print(f"Frames: {args.num_frames}, Max tokens: {args.max_tokens}")
    print("="*60)
    
    results = []
    
    # Test single request mode
    single_result = benchmark_vllm_single(
        args.num_frames, args.max_tokens, args.model
    )
    results.append(single_result)
    
    # Test batch mode
    batch_result = benchmark_vllm_batch(
        args.num_frames, args.max_tokens, args.model
    )
    results.append(batch_result)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for result in results:
        print(result)
    
    # Calculate speedups
    print("\n" + "="*60)
    print("SPEEDUP ANALYSIS")
    print("="*60)
    
    vllm_single = results[0]
    vllm_batch = results[1]
    
    batch_speedup = vllm_batch.throughput_tokens_per_sec / vllm_single.throughput_tokens_per_sec
    time_speedup = vllm_single.total_time / vllm_batch.total_time
    
    print(f"\nVLLM (batch) vs VLLM (single):")
    print(f"  Throughput speedup: {batch_speedup:.2f}x")
    print(f"  Total time speedup: {time_speedup:.2f}x")
    print(f"\n  ✓ Hypothesis {'VALIDATED' if batch_speedup > 1.5 else 'PARTIAL'}: {batch_speedup:.2f}x speedup")
    print(f"\n  The hypothesis: Temporal frames with shared prompt tokens benefit from VLLM batching")
    
    print("="*60)


if __name__ == "__main__":
    main()
