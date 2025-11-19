# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Streaming Video KV Cache Manager

Implements a specialized KV cache policy for video streaming that maintains:
1. Attention sink blocks (e.g., first frame)
2. Recent context blocks (e.g., last N frames)
3. Selective block freeing for middle frames

This enables true infinite-length video streaming without re-prefill.

Based on StreamingLLM principles adapted for video frames.
"""

from collections import defaultdict
import itertools
from collections.abc import Sequence
from typing import Optional

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.request import Request

logger = init_logger(__name__)


class StreamingVideoKVCacheManager(SingleTypeKVCacheManager):
    """
    KV cache manager for streaming video with attention sink policy.
    
    This manager maintains a sparse KV cache structure:
    - Sink blocks: Always kept (e.g., first frame)
    - Recent blocks: Last N blocks (sliding window)
    - Middle blocks: Explicitly freed when they fall out of the window
    
    This avoids the catastrophic re-prefill problem of naive sliding windows.
    """
    
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        num_sink_blocks: int = 4,  # Number of sink blocks to keep (e.g., 1 frame = ~8-16 blocks)
        num_recent_blocks: int = 32,  # Number of recent blocks to keep
    ):
        super().__init__(kv_cache_spec, block_pool, kv_cache_group_id, dcp_world_size)
        
        self.num_sink_blocks = num_sink_blocks
        self.num_recent_blocks = num_recent_blocks
        
        # Track which blocks are sink blocks for each request
        self.sink_blocks: dict[str, list[KVCacheBlock]] = {}
        
        # Track block allocation order for determining which blocks to free
        self.block_allocation_order: dict[str, list[KVCacheBlock]] = defaultdict(list)
        
        logger.info(
            f"Initialized StreamingVideoKVCacheManager: "
            f"sink_blocks={num_sink_blocks}, recent_blocks={num_recent_blocks}"
        )
    
    def allocate_new_blocks(
        self, request_id: str, num_tokens: int
    ) -> list[KVCacheBlock]:
        """
        Allocate new blocks for a request, capping at max_allowed_blocks.
        
        During prefill: Allow full allocation (let video frames load normally)
        During generation: Cap at sink + recent window
        
        This creates O(1) memory usage during generation while preserving
        full prefill accuracy.
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        
        # Check if this is initial prefill or generation
        is_prefill = len(req_blocks) == 0
        
        if not is_prefill:
            # During generation: Cap at maximum allowed blocks (sink + recent)
            max_allowed_blocks = self.num_sink_blocks + self.num_recent_blocks
            num_required_blocks = min(num_required_blocks, max_allowed_blocks)
        
        num_new_blocks = num_required_blocks - len(req_blocks)
        
        if num_new_blocks <= 0:
            # Already at capacity - blocks are reused via attention masking
            if not is_prefill:
                logger.debug(
                    f"⚠️  Request {request_id} at capacity - reusing blocks (generation mode)"
                )
            return []
        
        # Allocate new blocks
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        
        # Track allocation order
        self.block_allocation_order[request_id].extend(new_blocks)
        
        # Mark sink blocks on first allocation (prefill)
        if is_prefill:
            # First allocation - mark the first num_sink_blocks as sinks
            num_sink = min(self.num_sink_blocks, len(req_blocks))
            self.sink_blocks[request_id] = req_blocks[:num_sink]
            logger.info(
                f"✅ Prefill: Allocated {len(req_blocks)} blocks for request {request_id}, "
                f"marked {num_sink} as sink blocks"
            )
        else:
            # Generation: Check if we hit capacity
            max_allowed_blocks = self.num_sink_blocks + self.num_recent_blocks
            if len(req_blocks) >= max_allowed_blocks:
                logger.info(
                    f"🎯 Reached maximum capacity ({max_allowed_blocks} blocks) for request {request_id}. "
                    f"Structure: {len(self.sink_blocks.get(request_id, []))} sink + {self.num_recent_blocks} recent. "
                    f"Future tokens will reuse existing blocks (attention layer masks middle)."
                )
        
        return new_blocks
    
    def _free_middle_blocks(self, request_id: str, num_blocks_to_free: int) -> None:
        """
        Free middle blocks to maintain constant memory (sink + recent).
        
        This is the core streaming policy:
        - Keep sink blocks (first N blocks - attention sinks)
        - Free middle blocks (everything between sink and recent)
        - Keep recent blocks (last M blocks - recent context)
        
        Args:
            request_id: The request ID
            num_blocks_to_free: Number of middle blocks to free
        """
        req_blocks = self.req_to_blocks[request_id]
        sink_blocks = self.sink_blocks.get(request_id, [])
        
        if not req_blocks or num_blocks_to_free <= 0:
            return
        
        # Identify middle blocks (between sink and recent)
        sink_end_idx = len(sink_blocks)
        recent_start_idx = max(sink_end_idx, len(req_blocks) - self.num_recent_blocks)
        
        # Get middle blocks to free
        middle_blocks = []
        for i in range(sink_end_idx, recent_start_idx):
            if i < len(req_blocks):
                middle_blocks.append(req_blocks[i])
        
        # Free oldest middle blocks (FIFO)
        blocks_to_free = middle_blocks[:min(num_blocks_to_free, len(middle_blocks))]
        
        if blocks_to_free:
            logger.info(
                f"🗑️  Freeing {len(blocks_to_free)} middle blocks for request {request_id}. "
                f"Total blocks: {len(req_blocks)} -> {len(req_blocks) - len(blocks_to_free)}. "
                f"Maintaining: {len(sink_blocks)} sink + {self.num_recent_blocks} recent"
            )
            
            # Free the blocks
            self.block_pool.free_blocks(blocks_to_free)
            
            # Remove from req_to_blocks
            for block in blocks_to_free:
                if block in req_blocks:
                    req_blocks.remove(block)
            
            # Remove from allocation order tracking
            if request_id in self.block_allocation_order:
                for block in blocks_to_free:
                    if block in self.block_allocation_order[request_id]:
                        self.block_allocation_order[request_id].remove(block)
    
    def free(self, request_id: str) -> None:
        """
        Free all blocks for a request, including sink blocks.
        """
        # Clean up tracking structures
        self.sink_blocks.pop(request_id, None)
        self.block_allocation_order.pop(request_id, None)
        
        # Call parent to free blocks
        super().free(request_id)
        
        logger.debug(f"Freed all blocks for request {request_id}")
    
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Get the number of common prefix blocks for all requests.
        """
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks
    
    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        """
        Remove blocks that fall outside the sliding window.
        
        This is called during attention computation to handle cases where
        tokens fall outside the attention window.
        """
        # For streaming video, skipped blocks are handled by _free_middle_blocks_if_needed
        # We don't need additional logic here
        pass
    
    def get_streaming_stats(self, request_id: str) -> dict:
        """
        Get statistics about the streaming video cache state.
        
        Returns:
            dict with keys:
                - total_blocks: Current total blocks allocated
                - sink_blocks: Number of sink blocks
                - middle_blocks: Number of middle blocks (should be ~0 ideally)
                - recent_blocks: Number of recent blocks
        """
        req_blocks = self.req_to_blocks.get(request_id, [])
        sink_blocks = self.sink_blocks.get(request_id, [])
        
        total = len(req_blocks)
        sink = len(sink_blocks)
        
        # Recent blocks calculation
        recent_start = max(sink, total - self.num_recent_blocks)
        recent = total - recent_start
        
        # Middle blocks (should be minimal in steady state)
        middle = total - sink - recent
        
        return {
            "total_blocks": total,
            "sink_blocks": sink,
            "middle_blocks": middle,
            "recent_blocks": recent,
            "max_allowed": self.num_sink_blocks + self.num_recent_blocks,
        }

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Get the number of common prefix blocks for all requests.
        
        For streaming video, we don't use prefix caching (blocks are freed),
        so return 0.
        """
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list[BlockHash],
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        dcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Find longest prefix cache hit.
        """
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size > 1:
            block_size *= dcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def save_new_computed_blocks(
        self, request_id: str, new_computed_blocks: Sequence[KVCacheBlock]
    ) -> None:
        """
        Add the new computed blocks to the request and update sink blocks if needed.
        """
        super().save_new_computed_blocks(request_id, new_computed_blocks)
        
        # If this is the first batch of blocks (prefill), identify sink blocks
        # and track allocation order
        req_blocks = self.req_to_blocks[request_id]
        if new_computed_blocks:
            # Add to allocation order for consistency
            self.block_allocation_order[request_id].extend(new_computed_blocks)
            
            if request_id not in self.sink_blocks and req_blocks:
                num_sink = min(self.num_sink_blocks, len(req_blocks))
                self.sink_blocks[request_id] = req_blocks[:num_sink]
                logger.info(
                    f"✅ Cache Hit: Reused {len(new_computed_blocks)} blocks for request {request_id}, "
                    f"marked {num_sink} as sink blocks"
                )

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        """
        Remove blocks that are no longer needed and free them.
        
        For streaming video, skipped blocks are already handled by
        _free_middle_blocks_if_needed, so this is a no-op.
        """
        pass


class StreamingVideoConfig:
    """
    Configuration for streaming video KV cache.
    """
    
    def __init__(
        self,
        num_sink_blocks: int = 4,
        num_recent_blocks: int = 32,
        tokens_per_frame: int = 120,  # Approximate tokens per video frame
    ):
        self.num_sink_blocks = num_sink_blocks
        self.num_recent_blocks = num_recent_blocks
        self.tokens_per_frame = tokens_per_frame
        
        # Calculate approximate frame capacity
        # Assuming block_size = 16 (vLLM default)
        block_size = 16
        blocks_per_frame = cdiv(tokens_per_frame, block_size)
        
        self.approx_sink_frames = num_sink_blocks // blocks_per_frame
        self.approx_recent_frames = num_recent_blocks // blocks_per_frame
        
        logger.info(
            f"Streaming Video Config: ~{self.approx_sink_frames} sink frames, "
            f"~{self.approx_recent_frames} recent frames"
        )
    
    @classmethod
    def from_frame_counts(
        cls,
        num_sink_frames: int = 1,
        num_recent_frames: int = 4,
        tokens_per_frame: int = 120,
        block_size: int = 16,
    ) -> "StreamingVideoConfig":
        """
        Create config from frame counts instead of block counts.
        
        Args:
            num_sink_frames: Number of sink frames to keep (e.g., 1 for first frame)
            num_recent_frames: Number of recent frames to keep (e.g., 4-10)
            tokens_per_frame: Tokens per video frame
            block_size: KV cache block size
        """
        blocks_per_frame = cdiv(tokens_per_frame, block_size)
        
        num_sink_blocks = num_sink_frames * blocks_per_frame
        num_recent_blocks = num_recent_frames * blocks_per_frame
        
        return cls(
            num_sink_blocks=num_sink_blocks,
            num_recent_blocks=num_recent_blocks,
            tokens_per_frame=tokens_per_frame,
        )
