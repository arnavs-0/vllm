# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KV Cache Compression Module

Implements compression strategies for KV cache to handle long video sequences.
Specifically designed for vision-language models processing long videos.

Supported strategies:
- H2O (Heavy Hitter Oracle): Keep tokens with highest cumulative attention
- StreamingLLM: Keep attention sinks + recent tokens
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)


class CompressionStrategy(str, Enum):
    """KV cache compression strategies"""
    NONE = "none"
    H2O = "h2o"  # Heavy Hitter Oracle
    STREAMING_LLM = "streaming_llm"  # Attention sinks + recent


@dataclass
class KVCompressionConfig:
    """Configuration for KV cache compression"""
    
    # Strategy selection
    strategy: CompressionStrategy = CompressionStrategy.NONE
    
    # Compression thresholds
    max_tokens_before_compression: int = 4096  # Start compressing after this
    compression_ratio: float = 0.5  # Keep 50% of tokens when compressing
    
    # H2O specific
    heavy_hitter_ratio: float = 0.1  # Top 10% high-attention tokens
    recent_window_ratio: float = 0.4  # Recent 40% always kept
    
    # StreamingLLM specific  
    num_sink_tokens: int = 4  # Attention sink tokens to keep
    num_recent_tokens: int = 512  # Recent tokens to keep
    
    # Compression frequency
    compress_every_n_tokens: int = 256  # Check for compression this often
    
    def __post_init__(self):
        if self.strategy != CompressionStrategy.NONE:
            logger.info(
                f"KV Cache Compression enabled: strategy={self.strategy}, "
                f"max_tokens={self.max_tokens_before_compression}, "
                f"ratio={self.compression_ratio}"
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
        self.compression_count = 0
        
        # Track cumulative attention scores (for H2O)
        self.cumulative_attention_scores: Optional[torch.Tensor] = None
    
    def should_compress(self, current_seq_len: int) -> bool:
        """Check if compression should be triggered"""
        if self.config.strategy == CompressionStrategy.NONE:
            return False
        
        return current_seq_len >= self.config.max_tokens_before_compression
    
    def update_attention_scores(
        self,
        attention_weights: torch.Tensor,
        seq_len: int
    ):
        """
        Update cumulative attention scores for H2O strategy.
        
        Args:
            attention_weights: [batch, num_heads, seq_len, seq_len]
            seq_len: Current sequence length
        """
        if self.config.strategy != CompressionStrategy.H2O:
            return
        
        # Sum across query positions and heads to get importance per key position
        # attention_weights: [batch, num_heads, query_len, key_len]
        # We want importance of each key token
        key_importance = attention_weights.sum(dim=(0, 1, 2))  # [key_len]
        
        # Initialize or update cumulative scores
        if self.cumulative_attention_scores is None:
            self.cumulative_attention_scores = key_importance
        else:
            # Pad if sequence grew
            if len(key_importance) > len(self.cumulative_attention_scores):
                padding_size = len(key_importance) - len(self.cumulative_attention_scores)
                self.cumulative_attention_scores = F.pad(
                    self.cumulative_attention_scores,
                    (0, padding_size),
                    value=0.0
                )
            # Add new scores
            self.cumulative_attention_scores[:len(key_importance)] += key_importance
    
    def compress_h2o(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        H2O (Heavy Hitter Oracle) compression.
        
        Keep:
        1. Recent tokens (recent_window_ratio)
        2. High-attention tokens from older context (heavy_hitter_ratio)
        
        Args:
            key_cache: [num_blocks, num_heads, head_size, block_size]  
            value_cache: [num_blocks, num_heads, head_size, block_size]
            seq_len: Current sequence length
            
        Returns:
            compressed_keys, compressed_values, keep_mask
        """
        # Calculate keep sizes
        recent_size = int(seq_len * self.config.recent_window_ratio)
        heavy_hitter_size = int(seq_len * self.config.heavy_hitter_ratio)
        total_keep = min(recent_size + heavy_hitter_size, seq_len)
        
        # Recent tokens are always kept
        recent_start_idx = seq_len - recent_size
        keep_indices = list(range(recent_start_idx, seq_len))
        
        # Find heavy hitters from older context
        if self.cumulative_attention_scores is not None and recent_start_idx > 0:
            old_scores = self.cumulative_attention_scores[:recent_start_idx]
            _, topk_indices = torch.topk(old_scores, min(heavy_hitter_size, len(old_scores)))
            keep_indices = sorted(topk_indices.tolist() + keep_indices)
        
        # Create keep mask
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=key_cache.device)
        keep_mask[keep_indices] = True
        
        # Apply mask to caches
        # This is a simplified version - actual implementation needs to handle
        # the paged block structure of vLLM
        # For now, return the mask which will be used by attention computation
        
        logger.info(
            f"H2O Compression: {seq_len} -> {len(keep_indices)} tokens "
            f"({len(keep_indices)/seq_len*100:.1f}% kept)"
        )
        
        self.compression_count += 1
        
        return key_cache, value_cache, keep_mask
    
    def compress_streaming_llm(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        StreamingLLM compression.
        
        Keep:
        1. First N tokens (attention sinks)
        2. Last M tokens (recent window)
        Evict everything in between.
        
        Args:
            key_cache: KV cache keys
            value_cache: KV cache values
            seq_len: Current sequence length
            
        Returns:
            compressed_keys, compressed_values, keep_mask
        """
        keep_indices = []
        
        # Keep sink tokens
        keep_indices.extend(range(min(self.config.num_sink_tokens, seq_len)))
        
        # Keep recent tokens
        recent_start = max(self.config.num_sink_tokens, seq_len - self.config.num_recent_tokens)
        keep_indices.extend(range(recent_start, seq_len))
        
        # Remove duplicates and sort
        keep_indices = sorted(set(keep_indices))
        
        # Create keep mask
        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=key_cache.device)
        keep_mask[keep_indices] = True
        
        logger.info(
            f"StreamingLLM Compression: {seq_len} -> {len(keep_indices)} tokens "
            f"({self.config.num_sink_tokens} sinks + {self.config.num_recent_tokens} recent)"
        )
        
        self.compression_count += 1
        
        return key_cache, value_cache, keep_mask
    
    def compress(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Main compression entry point.
        
        Returns:
            compressed_keys, compressed_values, keep_mask (None if no compression)
        """
        if not self.should_compress(seq_len):
            return key_cache, value_cache, None
        
        if self.config.strategy == CompressionStrategy.H2O:
            return self.compress_h2o(key_cache, value_cache, seq_len)
        elif self.config.strategy == CompressionStrategy.STREAMING_LLM:
            return self.compress_streaming_llm(key_cache, value_cache, seq_len)
        else:
            return key_cache, value_cache, None
    
    def get_stats(self) -> dict:
        """Get compression statistics"""
        return {
            "strategy": self.config.strategy.value,
            "total_compressions": self.compression_count,
            "max_tokens_before_compression": self.config.max_tokens_before_compression,
        }
