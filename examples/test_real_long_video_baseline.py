"""
Real Video Baseline Test (No Streaming)
Usage: python examples/test_real_long_video_baseline.py --video-path /path/to/video.mp4

This script:
1. Loads the video file (or max_frames of it).
2. Processes it ALL AT ONCE in a single request (Standard Batch Inference).
3. Measures total time for comparison with streaming.
"""

import argparse
import time
import torch
import numpy as np
import cv2
from vllm import LLM, SamplingParams

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

def run_baseline_test(args):
    print(f"Initializing LLM with model: {args.model}")
    print(f"Video path: {args.video_path}")
    
    # Configuration
    # We need a large context to fit the whole video if it's long
    llm = LLM(
        model=args.model,
        max_model_len=32768, # Adjust if video is longer
        max_num_batched_tokens=32768,
        enforce_eager=True,
        gpu_memory_utilization=0.90,
        limit_mm_per_prompt={"image": 10, "video": 10},
    )
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    
    # Prepare prompt template
    prompt_template = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        "Describe what is happening in the video.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    print("\nLoading frames...")
    frames = []
    for frame in frame_generator(args.video_path, args.max_frames):
        frames.append(frame)
    
    print(f"Loaded {len(frames)} frames.")
    
    # vLLM expects numpy array for video
    video_input = np.array(frames)
    
    print("Starting Inference (Single Batch)...")
    
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
    
    output_text = outputs[0].outputs[0].text.replace("\n", " ").strip()
    fps = len(frames) / duration
    
    print("-" * 50)
    print(f"Total Frames: {len(frames)}")
    print(f"Total Time:   {duration:.2f}s")
    print(f"FPS:          {fps:.2f}")
    print("-" * 50)
    print(f"Output: {output_text}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str, required=True, help="Path to video file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (optional)")
    
    args = parser.parse_args()
    run_baseline_test(args)
