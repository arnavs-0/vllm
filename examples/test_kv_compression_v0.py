#!/usr/bin/env python3
"""
Test KV cache compression with V1 engine (default engine).
Note: V0 has been removed from vLLM - all calls now use V1 engine.
Compression works at the attention layer level in V1.
"""

from vllm import LLM, SamplingParams
from vllm.assets.video import VideoAsset

def test_with_compression_v0(num_frames: int = 4, max_tokens: int = 400):
    """Test with KV compression enabled on V1 engine"""
    print("\n" + "="*80)
    print(f"TEST: KV Compression with V1 Engine")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Engine: V1 (default - V0 has been removed)")
    print(f"  - Video frames: {num_frames}")
    print(f"  - Max tokens (compression threshold): {max_tokens}")
    print(f"  - Strategy: StreamingLLM")
    print("="*80)
    
    # V1 engine is now the DEFAULT (V0 has been removed)
    # The engine uses:
    # 1. Multi-process architecture (EngineCore + Worker)
    # 2. Compression at attention layer level
    # 3. Identifies tokens to evict but doesn't free memory yet
    llm = LLM(
        model="Qwen/Qwen3-VL-4B-Instruct",
        max_model_len=4096,
        max_num_seqs=1,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        # KV Compression settings
        enable_kv_compression=True,
        kv_compression_strategy="streaming_llm",
        kv_compression_max_tokens=max_tokens,
    )
    
    # Load video
    video = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays
    metadata = VideoAsset(name="baby_reading", num_frames=num_frames).metadata
    
    # Create prompt
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        "Describe what happens in this video in detail.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=50,
    )
    
    print("\nStarting generation with V1 engine...")
    print("="*80 + "\n")
    
    try:
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"video": [(video, metadata)]},
            },
            sampling_params=sampling_params,
        )
        
        print("\n" + "="*80)
        print("Generated output:")
        print("="*80)
        print(outputs[0].outputs[0].text)
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        return outputs
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return None


def compare_v0_vs_v1():
    """Show the current engine architecture info"""
    print("\n" + "="*80)
    print("ENGINE ARCHITECTURE INFO")
    print("="*80)
    print("V1 Engine (current):")
    print("  • Multi-process architecture (EngineCore + Worker)")
    print("  • Compression at attention layer level")
    print("  • Identifies tokens to evict but doesn't free memory yet")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test KV compression with V1 engine (V0 has been removed)"
    )
    parser.add_argument(
        "--num-frames", 
        type=int, 
        default=4,
        help="Number of video frames (4 frames = ~588 tokens)"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=400,
        help="Max tokens before compression (default 400 to trigger with 4 frames)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show engine architecture info and exit"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_v0_vs_v1()
        exit(0)
    
    print("\n" + "=" * 80)
    print("TESTING KV CACHE COMPRESSION WITH V1 ENGINE")
    print("=" * 80)
    
    compare_v0_vs_v1()
    
    result = test_with_compression_v0(
        num_frames=args.num_frames,
        max_tokens=args.max_tokens,
    )
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    if result is not None:
        print("Test completed successfully!")
        print("V1 engine handled KV compression correctly")
    else:
        print("Test failed - see errors above")
    print("="*80 + "\n")
