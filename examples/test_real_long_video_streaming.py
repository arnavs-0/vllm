"""
Real Video Streaming Test
Usage: python examples/test_real_long_video_streaming.py --video-path /path/to/video.mp4 --chunk-size 16

This script:
1. Loads a video file incrementally (does not load full video to RAM at start).
2. Streams it to vLLM in chunks (simulating live streaming).
3. Uses Prefix Caching + KV Cache Compression.
"""

import argparse
import time
import torch
import math
import numpy as np
import cv2
import os
from vllm import LLM, SamplingParams

def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def get_kv_stats(llm_instance, output):
    try:
        model_config = llm_instance.llm_engine.model_config
        cache_config = llm_instance.llm_engine.cache_config
        hf_config = model_config.hf_config
        text_config = getattr(hf_config, 'text_config', hf_config)
        
        block_size = cache_config.block_size
        dtype_size = torch.tensor([], dtype=model_config.dtype).element_size()
        num_layers = text_config.num_hidden_layers
        num_heads = text_config.num_attention_heads
        hidden_size = text_config.hidden_size
        head_dim = hidden_size // num_heads
        block_mem_size_mb = (2 * block_size * num_layers * num_heads * head_dim * dtype_size) / 1024 / 1024

        # For compression, we assume bounded memory (sink + recent)
        # This is an approximation since we can't easily query the exact compressed state from here
        # But we can check the total blocks used by the scheduler if we had access
        
        # Just return the calculated blocks based on output for now, 
        # or try to peek into the engine if possible.
        # For this script, we'll rely on the "time" metric to prove it works.
        return 0, 0.0
    except:
        return 0, 0.0

def frame_generator(video_path, max_frames=None):
    """Yields frames from video file one by one"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
        
        count += 1
        if max_frames and count >= max_frames:
            break
            
    cap.release()

def run_streaming_test(args):
    print(f"Initializing LLM with model: {args.model}")
    print(f"Video path: {args.video_path}")
    print(f"Chunk size: {args.chunk_size} frames")
    
    # Configuration
    llm = LLM(
        model=args.model,
        # Context Limit:
        # Your video seems to have high resolution (~800 tokens/frame).
        # 48 frames exceeded 32k tokens.
        # We increase limit to 64k to accommodate the rolling window safely.
        max_model_len=65536,
        max_num_batched_tokens=32768,
        enable_chunked_prefill=True, # Enable chunking for large updates
        enable_prefix_caching=True,
        enable_kv_compression=True,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 10, "video": 10},
        # Large recent window to enable prefix caching for long context
        kv_compression_num_sink_tokens=256,
        kv_compression_num_recent_tokens=16384, 
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    
    # Prepare prompt template
    prompt_template = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        "Describe what is happening in the video so far.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    # Streaming Loop
    history_frames = []
    step = 0
    total_time = 0
    
    # Rolling Window Configuration
    # We keep the first 'sink_size' frames (context anchor)
    # And the last 'window_size' frames (recent context)
    sink_size = 4
    window_size = args.chunk_size * 4 # Keep last 4 chunks as context
    
    print("\nStarting Stream...")
    print(f"Strategy: Rolling Window (Sink: {sink_size} frames, Window: {window_size} frames)")
    print(f"{'Step':<6} | {'Total Frames':<12} | {'Input Frames':<12} | {'Time (s)':<10} | {'FPS':<10} | {'Output':<40}")
    print("-" * 110)
    
    current_chunk = []
    all_frames_count = 0
    
    # Iterate through video frames
    for frame in frame_generator(args.video_path, args.max_frames):
        current_chunk.append(frame)
        all_frames_count += 1
        
        if len(current_chunk) >= args.chunk_size:
            step += 1
            
            # Add chunk to history
            history_frames.extend(current_chunk)
            current_chunk = [] # Reset chunk
            
            # Construct Input with Rolling Window
            # 1. Sink Frames (Start of video)
            if len(history_frames) > (sink_size + window_size):
                input_frames = history_frames[:sink_size] + history_frames[-window_size:]
            else:
                input_frames = history_frames
            
            # vLLM expects numpy array for video
            video_input = np.array(input_frames)
            
            # Generate
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.perf_counter()
            
            outputs = llm.generate(
                {
                    "prompt": prompt_template,
                    "multi_modal_data": {"video": video_input}
                },
                sampling_params=sampling_params
            )
            
            duration = time.perf_counter() - start_time
            total_time += duration
            
            output_text = outputs[0].outputs[0].text.replace("\n", " ").strip()
            fps = args.chunk_size / duration
            
            print(f"{step:<6} | {all_frames_count:<12} | {len(input_frames):<12} | {duration:<10.2f} | {fps:<10.2f} | {output_text[:40]}...")
            
            # Optional: Force garbage collection if memory is tight
            # import gc; gc.collect()
            
    print("-" * 110)
    print(f"Total Streaming Time: {total_time:.2f}s")
    if 'output_text' in locals():
        print("\nFinal Step Output:")
        print(output_text)
    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, required=True, help="Path to video file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--chunk-size", type=int, default=16, help="Number of frames per streaming step")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (optional)")
    
    args = parser.parse_args()
    run_streaming_test(args)
