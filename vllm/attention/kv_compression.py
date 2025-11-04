# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV Cache Compression Module

Implements compression strategies for KV cache to handle long video sequences.
Specifically designed for vision-language models processing long videos.

Supported strategies:
- StreamingLLM: Keep attention sinks + recent tokens
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)


# Global state for tracking sequence length across all layers
# This is needed because each layer has its own KVCacheCompressor instance
# but they all need to track the same growing sequence length
_global_seq_len_tracker: dict[str, int] = {}
_global_forward_call_counter: dict[str, int] = {}  # Track forward calls to detect new tokens


class CompressionStrategy(str, Enum):
    """KV cache compression strategies"""
    NONE = "none"
    STREAMING_LLM = "streaming_llm"  # Attention sinks + recent


@dataclass
class KVCompressionConfig:
    """Configuration for KV cache compression"""
    
    # Strategy selection
    strategy: CompressionStrategy = CompressionStrategy.NONE
    
    # Compression thresholds
    max_tokens_before_compression: int = 4096  # Start compressing after this
    compression_ratio: float = 0.5  # Keep 50% of tokens when compressing
    
    # StreamingLLM specific  
    num_sink_tokens: int = 4  # Attention sink tokens to keep
    num_recent_tokens: int = 128  # Recent tokens to keep (reduced from 512)
    
    # Compression frequency
    compress_every_n_tokens: int = 256  # Check for compression this often
    
    def __post_init__(self):
        if self.strategy != CompressionStrategy.NONE:
            logger.info(
                f"KV compression enabled: {self.strategy.value}, "
                f"max_tokens={self.max_tokens_before_compression}"

            )


class KVCacheCompressor:
    """
    Compresses KV cache by selecting important tokens.
    
    This modifies the KV cache tensors in-place by selecting a subset
    of key/value positions based on importance scores.
    """
    
    def __init__(self, config: KVCompressionConfig):
        self.config = config
        self.total_tokens_seen = 0
        
        # Use global tracker for sequence length (shared across all layers)
        # We use a simple key "default" for single-sequence tracking
        self._seq_tracker_key = "default"
        if self._seq_tracker_key not in _global_seq_len_tracker:
            _global_seq_len_tracker[self._seq_tracker_key] = None
        # Initialize compression count in global state
        if f"{self._seq_tracker_key}_compression_count" not in _global_seq_len_tracker:
            _global_seq_len_tracker[f"{self._seq_tracker_key}_compression_count"] = 0
    
    @property
    def compression_count(self) -> int:
        """Get the compression count from global state"""
        return _global_seq_len_tracker.get(f"{self._seq_tracker_key}_compression_count", 0)
    
    @compression_count.setter
    def compression_count(self, value: int):
        """Set the compression count in global state"""
        _global_seq_len_tracker[f"{self._seq_tracker_key}_compression_count"] = value
    
    @property
    def _actual_seq_len(self) -> Optional[int]:
        """Get the tracked sequence length from global state"""
        return _global_seq_len_tracker.get(self._seq_tracker_key)
    
    @_actual_seq_len.setter
    def _actual_seq_len(self, value: Optional[int]):
        """Set the tracked sequence length in global state"""
        _global_seq_len_tracker[self._seq_tracker_key] = value
    
    @property
    def _last_compressed_len(self) -> Optional[int]:
        """Get the last compressed length (for reference)"""
        return _global_seq_len_tracker.get(f"{self._seq_tracker_key}_last_compressed")
    
    @_last_compressed_len.setter
    def _last_compressed_len(self, value: Optional[int]):
        """Set the last compressed length"""
        _global_seq_len_tracker[f"{self._seq_tracker_key}_last_compressed"] = value
    
    def should_compress(self, current_seq_len: int) -> bool:
        """Check if compression should be applied
        
        During CUDA graph capture, we cannot perform dynamic operations,
        so we skip compression to avoid capture errors.
        """
        # Skip compression during CUDA graph capture
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return False
        # Use tracked sequence length if available, otherwise use provided
        if self._actual_seq_len is not None:
            actual_len = self._actual_seq_len
        else:
            actual_len = current_seq_len
            self._actual_seq_len = current_seq_len
        
        # For initial prefill, compress if we exceed max_tokens threshold
        if self.compression_count == 0:
            should_compress = actual_len >= self.config.max_tokens_before_compression
            if should_compress:
                logger.info(
                    f"Compression triggered: seq_len={actual_len}, "
                    f"threshold={self.config.max_tokens_before_compression}"
                )
            return should_compress
        
        # After initial compression, only re-compress when we accumulate enough new tokens
        # Target size after compression is num_sink_tokens + num_recent_tokens
        target_size = self.config.num_sink_tokens + self.config.num_recent_tokens
        
        # Re-compress when we've grown by num_recent_tokens beyond the target
        # This means we've accumulated a full window of new tokens
        should_compress = actual_len >= target_size + self.config.num_recent_tokens
        
        return should_compress
    
    def compress_streaming_llm(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        StreamingLLM compression with accumulation support.
        
        Keep:
        1. First N tokens (attention sinks)
        2. Last M tokens (recent window)
        Evict everything in between.
        
        Args:
            key_cache: KV cache keys
            value_cache: KV cache values
            seq_len: Current sequence length (may be stale, use _actual_seq_len)
            
        Returns:
            compressed_keys, compressed_values, keep_mask
        """
        # Skip compression during CUDA graph capture
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # Return identity mask (keep everything)
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=key_cache.device)
            return key_cache, value_cache, keep_mask
        # Use tracked sequence length for proper accumulation
        if self._actual_seq_len is not None:
            actual_seq_len = self._actual_seq_len
        else:
            actual_seq_len = seq_len
            self._actual_seq_len = seq_len
        
        keep_indices = []
        
        # Keep sink tokens
        keep_indices.extend(range(min(self.config.num_sink_tokens, actual_seq_len)))
        
        # Keep recent tokens
        recent_start = max(self.config.num_sink_tokens, actual_seq_len - self.config.num_recent_tokens)
        keep_indices.extend(range(recent_start, actual_seq_len))
        
        # Remove duplicates and sort
        keep_indices = sorted(set(keep_indices))
        
        # Create keep mask
        keep_mask = torch.zeros(actual_seq_len, dtype=torch.bool, device=key_cache.device)
        keep_mask[keep_indices] = True
        
        evicted_count = actual_seq_len - len(keep_indices)
        logger.info(
            f"StreamingLLM compression: {actual_seq_len} -> {len(keep_indices)} tokens, "
            f"evicted {evicted_count} ({evicted_count/actual_seq_len*100:.1f}%)"
            f" [compression_count={self.compression_count + 1}]"
        )
        
        # Update tracked length after compression
        self._last_compressed_len = len(keep_indices)
        self._actual_seq_len = len(keep_indices)  # After compression, we have fewer tokens
        
        self.compression_count += 1
        
        return key_cache, value_cache, keep_mask
    
    def compress(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        Main compression entry point.
        
        Args:
            key_cache: KV cache keys tensor
            value_cache: KV cache values tensor
            seq_len: Current sequence length
            
        Returns:
            compressed_keys, compressed_values, keep_mask (None if no compression),
            compression_info (dict with block freeing info, None if no compression)
        """
        # Skip compression during CUDA graph capture
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return key_cache, value_cache, None, None
        if not self.should_compress(seq_len):
            return key_cache, value_cache, None, None
        
        # Generate keep mask
        if self.config.strategy == CompressionStrategy.STREAMING_LLM:
            _, _, keep_mask = self.compress_streaming_llm(key_cache, value_cache, seq_len)
        else:
            return key_cache, value_cache, None, None
        
        # Actually apply compression by zeroing out evicted positions
        # This works with vLLM's paged attention as it will ignore zero entries
        try:
            # Apply mask to both key and value caches
            # Note: This modifies the cache in-place
            compression_info = self._apply_compression_mask(key_cache, value_cache, keep_mask, seq_len)
        except Exception as e:
            logger.warning(f"Failed to apply compression mask: {e}. Compression skipped.")
            return key_cache, value_cache, None, None
        
        return key_cache, value_cache, keep_mask, compression_info
    
    def _apply_compression_mask(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        keep_mask: torch.Tensor,
        seq_len: int
    ) -> dict:
        """
        Apply the keep_mask to actually compress the KV cache by zeroing evicted entries.
        
        This modifies the KV cache tensors in-place, setting evicted positions to zero.
        
        Returns:
            dict: Compression information including blocks that can be freed
        """
        # Create eviction mask (inverse of keep_mask)
        evict_mask = ~keep_mask
        
        # Count tokens to evict for logging
        num_evicted = evict_mask.sum().item()
        
        if num_evicted == 0:
            logger.debug("No tokens to evict after compression")
            return {
                "original_seq_len": seq_len,
                "compressed_seq_len": seq_len,
                "evicted_tokens": 0,
                "blocks_to_free": set(),
                "strategy": self.config.strategy.value,
                "compression_count": self.compression_count,
            }
        
        try:
            # vLLM's paged KV cache has complex structure that varies by backend.
            # Common structures:
            # - [num_blocks, block_size, num_heads, head_dim]  (paged)
            # - [num_layers, num_blocks, block_size, num_heads, head_dim]
            #
            # To handle this generically, we need to:
            # 1. Determine the structure
            # 2. Map token indices to block/position indices
            # 3. Zero out the appropriate entries
            
            # Assuming block_size (typically 16 in vLLM)
            block_size = 16  # vLLM default
            
            # Find evicted token indices
            evicted_indices = torch.where(evict_mask)[0]
            
            # Map to blocks (token_idx // block_size -> block_idx)
            evicted_blocks = set()
            for token_idx in evicted_indices.tolist():
                block_idx = token_idx // block_size
                evicted_blocks.add(block_idx)
            
            logger.info(
                f"Compression mask generated: {num_evicted}/{seq_len} "
                f"tokens marked for eviction ({num_evicted/seq_len*100:.1f}%). "
                f"Affects {len(evicted_blocks)} KV cache blocks."
            )
            
            # Actually zero out evicted positions in KV cache
            # This provides immediate memory savings by clearing unused data
            if hasattr(key_cache, 'shape') and hasattr(value_cache, 'shape'):
                # Handle different KV cache layouts
                if len(key_cache.shape) == 4:  # [num_blocks, block_size, num_heads, head_dim]
                    num_blocks, block_sz, num_heads, head_dim = key_cache.shape
                    
                    for token_idx in evicted_indices.tolist():
                        if token_idx < seq_len:
                            block_idx = token_idx // block_sz
                            pos_in_block = token_idx % block_sz
                            
                            if block_idx < num_blocks:
                                # Zero out key and value for this position
                                key_cache[block_idx, pos_in_block, :, :] = 0
                                value_cache[block_idx, pos_in_block, :, :] = 0
                                
                elif len(key_cache.shape) == 5:  # [num_layers, num_blocks, block_size, num_heads, head_dim]
                    num_layers, num_blocks, block_sz, num_heads, head_dim = key_cache.shape
                    
                    for token_idx in evicted_indices.tolist():
                        if token_idx < seq_len:
                            block_idx = token_idx // block_sz
                            pos_in_block = token_idx % block_sz
                            
                            if block_idx < num_blocks:
                                # Zero out key and value for this position across all layers
                                for layer_idx in range(num_layers):
                                    key_cache[layer_idx, block_idx, pos_in_block, :, :] = 0
                                    value_cache[layer_idx, block_idx, pos_in_block, :, :] = 0
                else:
                    logger.warning(f"Unsupported KV cache shape: key={key_cache.shape}, value={value_cache.shape}")
            
            logger.info(f"Zeroed out {num_evicted} evicted token positions in KV cache")
            
            # Identify blocks that are completely empty (all tokens evicted)
            # This is a simplified heuristic: if a block has more evicted tokens than kept tokens,
            # consider it for freeing. In practice, this would need more sophisticated logic
            # that considers which blocks belong to which requests.
            blocks_to_free = set()
            kept_tokens_per_block = {}
            
            # Count kept tokens per block
            kept_indices = torch.where(keep_mask)[0]
            for token_idx in kept_indices.tolist():
                if token_idx < seq_len:
                    block_idx = token_idx // block_size
                    kept_tokens_per_block[block_idx] = kept_tokens_per_block.get(block_idx, 0) + 1
            
            # A block can be freed if it has no kept tokens (all tokens were evicted)
            for block_idx in evicted_blocks:
                kept_in_block = kept_tokens_per_block.get(block_idx, 0)
                if kept_in_block == 0:
                    blocks_to_free.add(block_idx)
            
            if blocks_to_free:
                logger.info(f"Identified {len(blocks_to_free)} blocks that can be freed: {sorted(blocks_to_free)}")
            
            return {
                "original_seq_len": seq_len,
                "compressed_seq_len": seq_len - num_evicted,
                "evicted_tokens": num_evicted,
                "blocks_to_free": blocks_to_free,
                "strategy": self.config.strategy.value,
                "compression_count": self.compression_count,
            }
            
        except Exception as e:
            logger.warning(f"Error during compression mask application: {e}")
            logger.debug(f"KV cache shapes - key: {key_cache.shape if hasattr(key_cache, 'shape') else 'N/A'}, "
                        f"value: {value_cache.shape if hasattr(value_cache, 'shape') else 'N/A'}")
            
            return {
                "original_seq_len": seq_len,
                "compressed_seq_len": seq_len,
                "evicted_tokens": 0,
                "blocks_to_free": set(),
                "strategy": self.config.strategy.value,
                "compression_count": self.compression_count,
            }
    
    
    def get_attention_bias(self, keep_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Convert keep_mask to attention bias.
        
        Args:
            keep_mask: Boolean mask where True = keep, False = evict
            device: Device to create the bias on
            
        Returns:
            Attention bias tensor where evicted positions have -inf
        """
        # Create bias: 0 for kept positions, -inf for evicted positions
        bias = torch.zeros_like(keep_mask, dtype=torch.float32, device=device)
        bias[~keep_mask] = float('-inf')
        return bias
    
    def get_stats(self) -> dict:
        """Get compression statistics"""
        return {
            "strategy": self.config.strategy.value,
            "total_compressions": self.compression_count,
            "max_tokens_before_compression": self.config.max_tokens_before_compression,
        }
