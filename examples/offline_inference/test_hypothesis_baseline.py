#!/usr/bin/env python3
"""Run only baseline (transformers) benchmark - separate from VLLM"""

import argparse
import gc
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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


def create_temporal_frames(base_image: Image.Image, num_frames: int):
    frames = []
    base_array = np.array(base_image)
    for i in range(num_frames):
        noise = np.random.normal(0, 5, base_array.shape)
        frame_array = np.clip(base_array + noise, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(frame_array))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--model", type=str, default="google/gemma-3n-E2B-it")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"BASELINE BENCHMARK")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Frames: {args.num_frames}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    
    base_image = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")
    frames = create_temporal_frames(base_image, args.num_frames)
    question = "What is in this image?"
    
    total_tokens = 0
    request_times = []
    start_time = time.time()
    
    for i, frame in enumerate(frames):
        req_start = time.time()
        
        if "gemma" in args.model.lower():
            prompt = f"<start_of_turn>user\n<image_soft_token>{question}<end_of_turn>\n<start_of_turn>model\n"
            inputs = processor(text=prompt, images=frame, return_tensors="pt").to(device)
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": frame},
                    {"type": "text", "text": question}
                ]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[frame], return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        
        num_tokens = output.shape[1] - inputs['input_ids'].shape[1]
        total_tokens += num_tokens
        request_times.append(time.time() - req_start)
        print(f"Frame {i+1}/{args.num_frames}: {request_times[-1]:.2f}s ({num_tokens} tokens)")
        
        del inputs, output
        gc.collect()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg latency: {np.mean(request_times):.2f}s/request")
    print(f"Throughput: {total_tokens / total_time:.2f} tokens/sec")
    print(f"Total tokens: {total_tokens}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
