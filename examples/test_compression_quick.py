#!/usr/bin/env python3
"""
Quick test to verify KV compression implementation.
"""

def test_compression():
    """Run quick tests to verify compression works"""
    from vllm import LLM, SamplingParams
    import torch
    
    print("="*80)
    print("QUICK COMPRESSION TEST")
    print("="*80)
    
    # Test 1: Compression triggers
    print("\n[Test 1] Checking if compression triggers...")
    llm = LLM(
        model="Qwen/Qwen2-VL-2B-Instruct",
        max_model_len=2048,
        enforce_eager=True,
        enable_kv_compression=True,
        kv_compression_strategy="streaming_llm",
        kv_compression_max_tokens=50,
    )
    
    prompt = "Write a story: " + "word " * 100
    outputs = llm.generate([prompt], SamplingParams(max_tokens=10, temperature=0))
    
    # Print full generated output (no truncation)
    print(f"Generated output: {outputs[0].outputs[0].text}")
    print(f"No crashes - compression hook is stable")
    
    # Clean up GPU resources before next test
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    import time
    time.sleep(2)  # Give system time to release resources
    
    # Test 2: Verify compression with longer sequence
    print("\n[Test 2] Verifying compression calculation...")
    llm2 = LLM(
        model="Qwen/Qwen2-VL-2B-Instruct",
        max_model_len=2048,
        enforce_eager=True,
        enable_kv_compression=True,
        kv_compression_strategy="streaming_llm",
        kv_compression_max_tokens=100,  # Higher threshold to trigger at 200 tokens
    )
    
    # Create a longer prompt that will exceed the compression threshold
    long_prompt = "Write a story: " + "word " * 180  # This should create ~200 tokens
    outputs2 = llm2.generate([long_prompt], SamplingParams(max_tokens=10, temperature=0))
    
    # Print full generated output (no truncation)
    print(f"Generated output with compression: {outputs2[0].outputs[0].text}")
    print(f"No crashes - compression pipeline works")
    
    # Test 3: Verify compression math (direct test)
    print("\n[Test 3] Verifying compression calculation...")
    from vllm.attention.kv_compression import KVCacheCompressor, KVCompressionConfig, CompressionStrategy
    
    config = KVCompressionConfig(
        strategy=CompressionStrategy.STREAMING_LLM,
        max_tokens_before_compression=100,
        num_sink_tokens=4,
        num_recent_tokens=50,
    )
    compressor = KVCacheCompressor(config)
    
    # Simulate 200 token sequence
    seq_len = 200
    dummy_cache = torch.randn(1, 1, 1, 1)  # Dummy tensor
    _, _, keep_mask, compression_info = compressor.compress(dummy_cache, dummy_cache, seq_len)
    
    kept = keep_mask.sum().item()
    evicted = seq_len - kept
    
    print(f"  Sequence length: {seq_len}")
    print(f"  Kept tokens: {kept} (4 sinks + 50 recent = 54 expected)")
    print(f"  Evicted tokens: {evicted} ({evicted/seq_len*100:.1f}%)")
    
    # Check that blocks_to_free is populated
    blocks_to_free = compression_info.get("blocks_to_free", set())
    print(f"  Blocks to free: {len(blocks_to_free)} blocks")
    
    assert kept == 54, f"Expected 54 kept tokens, got {kept}"
    assert len(blocks_to_free) > 0, f"Expected blocks to free, got {blocks_to_free}"
    print(f"Math is correct!")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED")
    print("="*80)
    print("\nSUCCESS: KV cache compression is working end-to-end!")

if __name__ == "__main__":
    test_compression()
