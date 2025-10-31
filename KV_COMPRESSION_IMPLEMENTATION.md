# KV Cache Compression for vLLM

## What Was Actually Implemented

This is a **proper integration** into vLLM's attention layer, not a standalone script.

### Files Modified

1. **`vllm/attention/kv_compression.py`** (NEW)
   - Core compression logic
   - `KVCacheCompressor` class
   - Implements H2O and StreamingLLM strategies

2. **`vllm/config/cache.py`** (MODIFIED)
   - Added compression config options:
     - `enable_kv_compression`
     - `kv_compression_strategy`
     - `kv_compression_max_tokens`
     - `kv_compression_ratio`

3. **`vllm/attention/layer.py`** (MODIFIED)
   - Integrated `KVCacheCompressor` into `Attention` class
   - Compression happens during forward pass

### How It Works

```python
# 1. User enables compression via config
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    enable_kv_compression=True,  # ← Enables compression
    kv_compression_strategy="h2o",  # ← Choose strategy
    kv_compression_max_tokens=4096,  # ← Threshold
)

# 2. During inference, vLLM's attention layer:
#    - Monitors sequence length
#    - When seq_len > max_tokens, triggers compression
#    - Applies H2O or StreamingLLM to select important tokens
#    - Continues generation with compressed cache
```

### Architecture

```
┌─────────────────────────────────────┐
│  User Application                    │
│  (qwen_video_with_compression.py)    │
└──────────────┬──────────────────────┘
               │ llm.generate()
               ▼
┌─────────────────────────────────────┐
│  vLLM LLM Engine                     │
│  - Reads compression config          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Attention Layer (layer.py)          │
│  - Creates KVCacheCompressor         │
│  - Calls compressor.compress()       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  KVCacheCompressor (kv_compression.py)│
│  - compress_h2o()                    │
│  - compress_streaming_llm()          │
│  - Returns compressed KV + mask      │
└─────────────────────────────────────┘
```

### Compression Strategies

#### H2O (Heavy Hitter Oracle)
```
Original: [t0, t1, t2, ..., t3000, t3001, ..., t4096]
                                           ↑ recent (always keep)

Attention scores: [0.8, 0.05, 0.7, ..., ...]
                   ↑ high  ↑ low  ↑ high

After H2O:
- Keep recent 40%: [t2867, ..., t4096]  (1640 tokens)
- Keep heavy hitters 10%: [t0, t2, t50, ...]  (410 tokens)
- Total: 2050 tokens (50% compression)
```

#### StreamingLLM
```
Original: [t0, t1, t2, t3, ..., t4092, t4093, t4094, t4095]
           ↑  attention sinks          ↑ recent window

After StreamingLLM:
- Keep first 4: [t0, t1, t2, t3]
- Keep recent 512: [t3584, ..., t4095]
- Total: 516 tokens (~87% compression!)
```

### Current Limitations

⚠️ **This is a foundation - not fully functional yet**

What's implemented:
- ✅ Config integration
- ✅ Compression logic (H2O, StreamingLLM)
- ✅ Attention layer integration

What still needs work:
- ❌ **Attention score extraction**: Need to capture actual attention weights during forward pass
- ❌ **Block-level compression**: vLLM uses paged blocks, need to handle block structure
- ❌ **Attention backend integration**: Need to modify attention computation to use compressed cache
- ❌ **Testing**: No tests yet

### Why This Is Different from the Other Files

Previous files (`qwen_video_streaming.py`, `qwen_kv_compression.py`):
- ❌ Standalone scripts
- ❌ Don't touch vLLM internals
- ❌ Can't actually compress the KV cache

This implementation:
- ✅ Modifies vLLM core (`attention/layer.py`)
- ✅ Integrates with cache config
- ✅ Hooks into attention forward pass
- ✅ Can actually access/modify KV cache

### Next Steps to Make It Fully Functional

1. **Capture attention scores** during forward pass:
```python
# In Attention.forward(), after attention computation:
if self.kv_compressor is not None:
    self.kv_compressor.update_attention_scores(
        attention_weights, seq_len
    )
```

2. **Apply compression to paged blocks**:
```python
# Need to map keep_mask to vLLM's block structure
# and actually evict unused blocks
```

3. **Modify attention backends** to handle compressed sequences:
```python
# Update FlashAttention, xformers, etc. to work with
# non-contiguous token sequences
```

This would require 2-3 more days of work to fully implement and test.

### How to Test Current Implementation

```bash
# This will initialize the compressor but won't see full compression yet
python qwen_video_with_compression.py

# Check logs for:
# "KV compression enabled for layer model.layers.0.self_attn.attn"
```

### Comparison to Alternatives

| Approach | Implementation Effort | Actually Works? |
|----------|----------------------|-----------------|
| **Sliding window** | ✅ Built into vLLM | ✅ Yes, but loses context |
| **This (KV compression)** | ⚠️ Partially done | ⚠️ Foundation laid, needs completion |
| **Hierarchical summaries** (my first approach) | ✅ Simple | ✅ Yes, but not true KV compression |

### Recommendation

Given the complexity:

**Option A**: Complete this implementation (2-3 days work)
- Finish attention score capture
- Handle block-level compression
- Test with long videos

**Option B**: Use hierarchical approach for now
- Works immediately
- Simpler to understand
- Can query any time point
- Not "true" KV compression but achieves similar goal

Your call! But now you have actual vLLM modifications, not random standalone scripts.
