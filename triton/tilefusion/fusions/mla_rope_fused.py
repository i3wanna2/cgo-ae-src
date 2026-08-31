"""
MLA RoPE Fused Kernel Implementation in Triton

This file implements the concat_and_cache_mla_rope_fused CUDA kernel in Triton,
decomposed into independent kernels that can be auto-fused by tilefusion.

Original CUDA kernel does:
1. RoPE on Q (multi-head)
2. RoPE on K (single head)  
3. Write K + KV_C to KV cache (with optional FP8 quantization)
"""

import math
import torch
import triton
import triton.language as tl
from tilefusion.utils.utils import DEVICE


# =============================================================================
# Independent Triton Kernels
# =============================================================================

@triton.jit
def rope_apply_q_kernel(
    q_pe_ptr,                    # [num_tokens, num_q_heads, rot_dim]
    cos_sin_cache_ptr,           # [max_position, rot_dim]
    positions_ptr,               # [num_tokens]
    num_tokens,
    num_q_heads,
    rot_dim,
    q_pe_stride_token,
    q_pe_stride_head,
    q_pe_stride_dim,
    cos_sin_stride_pos,
    cos_sin_stride_dim,
    IS_NEOX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply RoPE to Q tensor (in-place)
    
    Grid: (num_tokens, num_q_heads) - each program handles one token's one head
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Load position for this token
    pos = tl.load(positions_ptr + token_idx)
    
    embed_dim = rot_dim // 2
    cos_sin_base = cos_sin_cache_ptr + pos * cos_sin_stride_pos
    
    # Each program processes one head
    q_pe_head_ptr = q_pe_ptr + token_idx * q_pe_stride_token + head_idx * q_pe_stride_head
    
    for pair_start in range(0, embed_dim, BLOCK_SIZE):
        pair_offs = pair_start + tl.arange(0, BLOCK_SIZE)
        mask = pair_offs < embed_dim
        
        # Load cos and sin
        cos = tl.load(cos_sin_base + pair_offs * cos_sin_stride_dim, mask=mask, other=0.0)
        sin = tl.load(cos_sin_base + (pair_offs + embed_dim) * cos_sin_stride_dim, mask=mask, other=0.0)
        
        if IS_NEOX:
            # GPT-NeoX style: [x0, x1, ..., xd/2-1, xd/2, ..., xd-1]
            pair_idx_x = pair_offs
            pair_idx_y = embed_dim + pair_offs
        else:
            # GPT-J style: [x0, y0, x1, y1, ...]
            pair_idx_x = pair_offs * 2
            pair_idx_y = pair_offs * 2 + 1
        
        # Load x and y
        x_src = tl.load(q_pe_head_ptr + pair_idx_x * q_pe_stride_dim, mask=mask, other=0.0)
        y_src = tl.load(q_pe_head_ptr + pair_idx_y * q_pe_stride_dim, mask=mask, other=0.0)
        
        # Apply rotation
        x_dst = x_src * cos - y_src * sin
        y_dst = y_src * cos + x_src * sin
        
        # Store back
        tl.store(q_pe_head_ptr + pair_idx_x * q_pe_stride_dim, x_dst, mask=mask)
        tl.store(q_pe_head_ptr + pair_idx_y * q_pe_stride_dim, y_dst, mask=mask)


@triton.jit
def rope_apply_k_kernel(
    k_pe_ptr,                    # [num_tokens, rot_dim]
    cos_sin_cache_ptr,           # [max_position, rot_dim]
    positions_ptr,               # [num_tokens]
    slot_mapping_ptr,            # [num_tokens] - needed to skip padding tokens
    num_tokens,
    rot_dim,
    k_pe_stride_token,
    k_pe_stride_dim,
    cos_sin_stride_pos,
    cos_sin_stride_dim,
    IS_NEOX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply RoPE to K tensor (in-place, single head)"""
    token_idx = tl.program_id(0)
    
    # Load position for this token
    pos = tl.load(positions_ptr + token_idx)
    
    embed_dim = rot_dim // 2
    cos_sin_base = cos_sin_cache_ptr + pos * cos_sin_stride_pos
    k_pe_token_ptr = k_pe_ptr + token_idx * k_pe_stride_token
    
    for pair_start in range(0, embed_dim, BLOCK_SIZE):
        pair_offs = pair_start + tl.arange(0, BLOCK_SIZE)
        mask = pair_offs < embed_dim
        
        # Load cos and sin
        cos = tl.load(cos_sin_base + pair_offs * cos_sin_stride_dim, mask=mask, other=0.0)
        sin = tl.load(cos_sin_base + (pair_offs + embed_dim) * cos_sin_stride_dim, mask=mask, other=0.0)
        
        if IS_NEOX:
            pair_idx_x = pair_offs
            pair_idx_y = embed_dim + pair_offs
        else:
            pair_idx_x = pair_offs * 2
            pair_idx_y = pair_offs * 2 + 1
        
        # Load x and y
        x_src = tl.load(k_pe_token_ptr + pair_idx_x * k_pe_stride_dim, mask=mask, other=0.0)
        y_src = tl.load(k_pe_token_ptr + pair_idx_y * k_pe_stride_dim, mask=mask, other=0.0)
        
        # Apply rotation
        x_dst = x_src * cos - y_src * sin
        y_dst = y_src * cos + x_src * sin
        
        # Store back
        tl.store(k_pe_token_ptr + pair_idx_x * k_pe_stride_dim, x_dst, mask=mask)
        tl.store(k_pe_token_ptr + pair_idx_y * k_pe_stride_dim, y_dst, mask=mask)


@triton.jit
def kv_cache_write_kernel(
    k_pe_ptr,                    # [num_tokens, rot_dim] - already RoPE'd
    kv_c_ptr,                    # [num_tokens, kv_lora_rank]
    kv_cache_ptr,                # [num_blocks, block_size, kv_lora_rank + rot_dim]
    slot_mapping_ptr,            # [num_tokens]
    num_tokens,
    rot_dim,
    kv_lora_rank,
    k_pe_stride_token,
    k_pe_stride_dim,
    kv_c_stride_token,
    kv_c_stride_dim,
    block_stride,
    entry_stride,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Write K_PE and KV_C to KV cache"""
    token_idx = tl.program_id(0)
    
    # Load slot mapping
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    block_idx = slot_idx // block_size
    entry_idx = slot_idx % block_size
    
    kv_cache_entry_ptr = kv_cache_ptr + block_idx * block_stride + entry_idx * entry_stride
    
    # Write KV_C (kv_lora_rank elements at the beginning)
    kv_c_token_ptr = kv_c_ptr + token_idx * kv_c_stride_token
    for i in range(0, kv_lora_rank, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < kv_lora_rank
        vals = tl.load(kv_c_token_ptr + offs * kv_c_stride_dim, mask=mask, other=0.0)
        tl.store(kv_cache_entry_ptr + offs, vals, mask=mask)
    
    # Write K_PE (rot_dim elements after kv_lora_rank)
    k_pe_token_ptr = k_pe_ptr + token_idx * k_pe_stride_token
    kv_cache_k_ptr = kv_cache_entry_ptr + kv_lora_rank
    for i in range(0, rot_dim, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < rot_dim
        vals = tl.load(k_pe_token_ptr + offs * k_pe_stride_dim, mask=mask, other=0.0)
        tl.store(kv_cache_k_ptr + offs, vals, mask=mask)


# =============================================================================
# Fused Triton Kernel (Manual Fusion for comparison)
# =============================================================================

@triton.jit
def rope_and_cache_fused_kernel(
    positions_ptr,               # [num_tokens]
    q_pe_ptr,                    # [num_tokens, num_q_heads, rot_dim]
    k_pe_ptr,                    # [num_tokens, rot_dim]
    kv_c_ptr,                    # [num_tokens, kv_lora_rank]
    cos_sin_cache_ptr,           # [max_position, rot_dim]
    kv_cache_ptr,                # [num_blocks, block_size, kv_lora_rank + rot_dim]
    slot_mapping_ptr,            # [num_tokens]
    num_tokens,
    num_q_heads,
    rot_dim,
    kv_lora_rank,
    q_pe_stride_token,
    q_pe_stride_head,
    q_pe_stride_dim,
    k_pe_stride_token,
    k_pe_stride_dim,
    kv_c_stride_token,
    kv_c_stride_dim,
    cos_sin_stride_pos,
    cos_sin_stride_dim,
    block_stride,
    entry_stride,
    block_size,
    IS_NEOX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RoPE + KV Cache Write kernel
    
    Grid: (num_tokens, num_q_heads) - each program handles one token's one head for Q RoPE
    Only head_idx == 0 handles K RoPE and cache write
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Load position for this token
    pos = tl.load(positions_ptr + token_idx)
    embed_dim = rot_dim // 2
    cos_sin_base = cos_sin_cache_ptr + pos * cos_sin_stride_pos
    
    # =========================================================================
    # Part 1: RoPE on Q (each program handles one head)
    # =========================================================================
    q_pe_head_ptr = q_pe_ptr + token_idx * q_pe_stride_token + head_idx * q_pe_stride_head
    
    for pair_start in range(0, embed_dim, BLOCK_SIZE):
        pair_offs = pair_start + tl.arange(0, BLOCK_SIZE)
        mask = pair_offs < embed_dim
        
        cos = tl.load(cos_sin_base + pair_offs * cos_sin_stride_dim, mask=mask, other=0.0)
        sin = tl.load(cos_sin_base + (pair_offs + embed_dim) * cos_sin_stride_dim, mask=mask, other=0.0)
        
        if IS_NEOX:
            pair_idx_x = pair_offs
            pair_idx_y = embed_dim + pair_offs
        else:
            pair_idx_x = pair_offs * 2
            pair_idx_y = pair_offs * 2 + 1
        
        x_src = tl.load(q_pe_head_ptr + pair_idx_x * q_pe_stride_dim, mask=mask, other=0.0)
        y_src = tl.load(q_pe_head_ptr + pair_idx_y * q_pe_stride_dim, mask=mask, other=0.0)
        
        x_dst = x_src * cos - y_src * sin
        y_dst = y_src * cos + x_src * sin
        
        tl.store(q_pe_head_ptr + pair_idx_x * q_pe_stride_dim, x_dst, mask=mask)
        tl.store(q_pe_head_ptr + pair_idx_y * q_pe_stride_dim, y_dst, mask=mask)
    
    # =========================================================================
    # Only head_idx == 0 handles K RoPE and cache write
    # =========================================================================
    if head_idx != 0:
        return
    
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    # =========================================================================
    # Part 2: RoPE on K (single head)
    # =========================================================================
    k_pe_token_ptr = k_pe_ptr + token_idx * k_pe_stride_token
    
    for pair_start in range(0, embed_dim, BLOCK_SIZE):
        pair_offs = pair_start + tl.arange(0, BLOCK_SIZE)
        mask = pair_offs < embed_dim
        
        cos = tl.load(cos_sin_base + pair_offs * cos_sin_stride_dim, mask=mask, other=0.0)
        sin = tl.load(cos_sin_base + (pair_offs + embed_dim) * cos_sin_stride_dim, mask=mask, other=0.0)
        
        if IS_NEOX:
            pair_idx_x = pair_offs
            pair_idx_y = embed_dim + pair_offs
        else:
            pair_idx_x = pair_offs * 2
            pair_idx_y = pair_offs * 2 + 1
        
        x_src = tl.load(k_pe_token_ptr + pair_idx_x * k_pe_stride_dim, mask=mask, other=0.0)
        y_src = tl.load(k_pe_token_ptr + pair_idx_y * k_pe_stride_dim, mask=mask, other=0.0)
        
        x_dst = x_src * cos - y_src * sin
        y_dst = y_src * cos + x_src * sin
        
        tl.store(k_pe_token_ptr + pair_idx_x * k_pe_stride_dim, x_dst, mask=mask)
        tl.store(k_pe_token_ptr + pair_idx_y * k_pe_stride_dim, y_dst, mask=mask)
    
    # =========================================================================
    # Part 3: KV Cache Write
    # =========================================================================
    block_idx = slot_idx // block_size
    entry_idx = slot_idx % block_size
    kv_cache_entry_ptr = kv_cache_ptr + block_idx * block_stride + entry_idx * entry_stride
    
    # Write KV_C
    kv_c_token_ptr = kv_c_ptr + token_idx * kv_c_stride_token
    for i in range(0, kv_lora_rank, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < kv_lora_rank
        vals = tl.load(kv_c_token_ptr + offs * kv_c_stride_dim, mask=mask, other=0.0)
        tl.store(kv_cache_entry_ptr + offs, vals, mask=mask)
    
    # Write K_PE (after applying RoPE - reload from k_pe)
    kv_cache_k_ptr = kv_cache_entry_ptr + kv_lora_rank
    for i in range(0, rot_dim, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < rot_dim
        vals = tl.load(k_pe_token_ptr + offs * k_pe_stride_dim, mask=mask, other=0.0)
        tl.store(kv_cache_k_ptr + offs, vals, mask=mask)


# =============================================================================
# Optimized Fused Kernel (Memory Traffic Reduction)
# =============================================================================

@triton.jit
def rope_and_cache_fused_kernel_optimized(
    positions_ptr,               # [num_tokens]
    q_pe_ptr,                    # [num_tokens, num_q_heads, rot_dim]
    k_pe_ptr,                    # [num_tokens, rot_dim]
    kv_c_ptr,                    # [num_tokens, kv_lora_rank]
    cos_sin_cache_ptr,           # [max_position, rot_dim]
    kv_cache_ptr,                # [num_blocks, block_size, kv_lora_rank + rot_dim]
    slot_mapping_ptr,            # [num_tokens]
    num_tokens,
    num_q_heads,
    rot_dim,
    kv_lora_rank,
    q_pe_stride_token,
    q_pe_stride_head,
    q_pe_stride_dim,
    k_pe_stride_token,
    k_pe_stride_dim,
    kv_c_stride_token,
    kv_c_stride_dim,
    cos_sin_stride_pos,
    cos_sin_stride_dim,
    block_stride,
    entry_stride,
    block_size,
    IS_NEOX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,      # For Q/K RoPE (should be >= embed_dim, e.g., 64)
    KVC_BLOCK_SIZE: tl.constexpr,  # For KV_C copy (should be larger, e.g., 256 or 512)
):
    """Optimized Fused RoPE + KV Cache Write kernel
    
    Optimizations:
    1. Vectorized Q-RoPE without for loop (when BLOCK_SIZE >= embed_dim)
    2. K-RoPE results written directly to both k_pe and kv_cache (no re-read)
    3. Larger block size for KV_C copy
    
    Grid: (num_tokens, num_q_heads)
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # =========================================================================
    # Precompute constants and load cos/sin once (reused for Q and K)
    # =========================================================================
    pos = tl.load(positions_ptr + token_idx)
    embed_dim = rot_dim // 2
    cos_sin_base = cos_sin_cache_ptr + pos * cos_sin_stride_pos
    
    # Vectorized load of cos/sin (assuming BLOCK_SIZE >= embed_dim)
    offs_pair = tl.arange(0, BLOCK_SIZE)
    mask_pair = offs_pair < embed_dim
    
    cos = tl.load(cos_sin_base + offs_pair * cos_sin_stride_dim, mask=mask_pair, other=0.0)
    sin = tl.load(cos_sin_base + (offs_pair + embed_dim) * cos_sin_stride_dim, mask=mask_pair, other=0.0)
    
    # Precompute indices based on RoPE style
    if IS_NEOX:
        idx_x = offs_pair
        idx_y = offs_pair + embed_dim
    else:
        idx_x = offs_pair * 2
        idx_y = offs_pair * 2 + 1
    
    # =========================================================================
    # Part 1: Q-RoPE (Vectorized, no for loop)
    # =========================================================================
    q_pe_head_ptr = q_pe_ptr + token_idx * q_pe_stride_token + head_idx * q_pe_stride_head
    
    x_q = tl.load(q_pe_head_ptr + idx_x * q_pe_stride_dim, mask=mask_pair, other=0.0)
    y_q = tl.load(q_pe_head_ptr + idx_y * q_pe_stride_dim, mask=mask_pair, other=0.0)
    
    # Apply rotation and store
    q_res_x = x_q * cos - y_q * sin
    q_res_y = y_q * cos + x_q * sin
    
    tl.store(q_pe_head_ptr + idx_x * q_pe_stride_dim, q_res_x, mask=mask_pair)
    tl.store(q_pe_head_ptr + idx_y * q_pe_stride_dim, q_res_y, mask=mask_pair)
    
    # =========================================================================
    # Only head_idx == 0 handles K-RoPE and cache write
    # =========================================================================
    if head_idx != 0:
        return
    
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    # =========================================================================
    # Part 2 & 3: K-RoPE + Direct Cache Write (Merged - no re-read)
    # =========================================================================
    k_pe_token_ptr = k_pe_ptr + token_idx * k_pe_stride_token
    
    # Load K values
    x_k = tl.load(k_pe_token_ptr + idx_x * k_pe_stride_dim, mask=mask_pair, other=0.0)
    y_k = tl.load(k_pe_token_ptr + idx_y * k_pe_stride_dim, mask=mask_pair, other=0.0)
    
    # Apply rotation (results stay in registers)
    k_res_x = x_k * cos - y_k * sin
    k_res_y = y_k * cos + x_k * sin
    
    # Write back to k_pe (in-place update)
    tl.store(k_pe_token_ptr + idx_x * k_pe_stride_dim, k_res_x, mask=mask_pair)
    tl.store(k_pe_token_ptr + idx_y * k_pe_stride_dim, k_res_y, mask=mask_pair)
    
    # Compute cache address
    b_idx = slot_idx // block_size
    e_idx = slot_idx % block_size
    kv_cache_entry_ptr = kv_cache_ptr + b_idx * block_stride + e_idx * entry_stride
    
    # Write K_PE directly to cache (from registers, no re-read!)
    kv_cache_k_ptr = kv_cache_entry_ptr + kv_lora_rank
    tl.store(kv_cache_k_ptr + idx_x, k_res_x, mask=mask_pair)
    tl.store(kv_cache_k_ptr + idx_y, k_res_y, mask=mask_pair)
    
    # =========================================================================
    # Part 4: Copy KV_C to cache (larger block size for bandwidth)
    # =========================================================================
    kv_c_token_ptr = kv_c_ptr + token_idx * kv_c_stride_token
    
    for c_start in range(0, kv_lora_rank, KVC_BLOCK_SIZE):
        c_offs = c_start + tl.arange(0, KVC_BLOCK_SIZE)
        c_mask = c_offs < kv_lora_rank
        c_vals = tl.load(kv_c_token_ptr + c_offs * kv_c_stride_dim, mask=c_mask, other=0.0)
        tl.store(kv_cache_entry_ptr + c_offs, c_vals, mask=c_mask)


# =============================================================================
# PyTorch Reference Implementation
# =============================================================================

def rope_reference(x, cos, sin, is_neox=True):
    """
    Reference RoPE implementation in PyTorch
    x: [..., rot_dim]
    cos, sin: [rot_dim // 2] or broadcast compatible
    """
    rot_dim = x.shape[-1]
    embed_dim = rot_dim // 2
    
    if is_neox:
        # NeoX style: first half and second half
        x1, x2 = x[..., :embed_dim], x[..., embed_dim:]
    else:
        # GPT-J style: interleaved
        x1, x2 = x[..., ::2], x[..., 1::2]
    
    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    
    if is_neox:
        return torch.cat([out1, out2], dim=-1)
    else:
        out = torch.stack([out1, out2], dim=-1)
        return out.flatten(-2)


def concat_and_cache_mla_rope_reference(
    positions,           # [num_tokens]
    q_pe,                # [num_tokens, num_q_heads, rot_dim]
    k_pe,                # [num_tokens, rot_dim]
    kv_c,                # [num_tokens, kv_lora_rank]
    rope_cos_sin_cache,  # [max_position, rot_dim]
    rope_is_neox,
    kv_cache_slot_mapping,  # [num_tokens]
    kv_cache,            # [num_blocks, block_size, kv_lora_rank + rot_dim]
):
    """
    PyTorch reference implementation of the fused MLA RoPE + Cache kernel
    Modifies q_pe, k_pe, and kv_cache in-place
    """
    num_tokens = q_pe.shape[0]
    rot_dim = q_pe.shape[2]
    embed_dim = rot_dim // 2
    kv_lora_rank = kv_c.shape[1]
    block_size = kv_cache.shape[1]
    
    for token_idx in range(num_tokens):
        pos = positions[token_idx].item()
        
        # Get cos/sin for this position
        cos = rope_cos_sin_cache[pos, :embed_dim]
        sin = rope_cos_sin_cache[pos, embed_dim:]
        
        # Apply RoPE to Q (all heads)
        for head_idx in range(q_pe.shape[1]):
            q_pe[token_idx, head_idx] = rope_reference(
                q_pe[token_idx, head_idx], cos, sin, rope_is_neox
            )
        
        # Apply RoPE to K (single head)
        k_pe[token_idx] = rope_reference(k_pe[token_idx], cos, sin, rope_is_neox)
        
        # Write to KV cache
        slot_idx = kv_cache_slot_mapping[token_idx].item()
        block_idx = slot_idx // block_size
        entry_idx = slot_idx % block_size
        
        # Write kv_c at the beginning
        kv_cache[block_idx, entry_idx, :kv_lora_rank] = kv_c[token_idx]
        # Write k_pe after kv_lora_rank
        kv_cache[block_idx, entry_idx, kv_lora_rank:] = k_pe[token_idx]
    
    return q_pe, k_pe, kv_cache


# =============================================================================
# Wrapper Functions
# =============================================================================

def run_separate_kernels(
    positions, q_pe, k_pe, kv_c, rope_cos_sin_cache, 
    rope_is_neox, kv_cache_slot_mapping, kv_cache
):
    """Run the three separate Triton kernels"""
    num_tokens = q_pe.shape[0]
    num_q_heads = q_pe.shape[1]
    rot_dim = q_pe.shape[2]
    kv_lora_rank = kv_c.shape[1]
    block_size = kv_cache.shape[1]
    
    BLOCK_SIZE = 64
    
    # Kernel 1: RoPE on Q - 2D grid (num_tokens, num_q_heads) for better parallelism
    grid_q = (num_tokens, num_q_heads)
    rope_apply_q_kernel[grid_q](
        q_pe, rope_cos_sin_cache, positions,
        num_tokens, num_q_heads, rot_dim,
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        rope_cos_sin_cache.stride(0), rope_cos_sin_cache.stride(1),
        IS_NEOX=rope_is_neox,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Kernel 2: RoPE on K (needs slot_mapping to skip padding tokens)
    grid_k = (num_tokens,)
    rope_apply_k_kernel[grid_k](
        k_pe, rope_cos_sin_cache, positions, kv_cache_slot_mapping,
        num_tokens, rot_dim,
        k_pe.stride(0), k_pe.stride(1),
        rope_cos_sin_cache.stride(0), rope_cos_sin_cache.stride(1),
        IS_NEOX=rope_is_neox,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Kernel 3: KV Cache Write
    grid_cache = (num_tokens,)
    kv_cache_write_kernel[grid_cache](
        k_pe, kv_c, kv_cache, kv_cache_slot_mapping,
        num_tokens, rot_dim, kv_lora_rank,
        k_pe.stride(0), k_pe.stride(1),
        kv_c.stride(0), kv_c.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), block_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return q_pe, k_pe, kv_cache


def run_fused_kernel(
    positions, q_pe, k_pe, kv_c, rope_cos_sin_cache,
    rope_is_neox, kv_cache_slot_mapping, kv_cache
):
    """Run the manually fused Triton kernel"""
    num_tokens = q_pe.shape[0]
    num_q_heads = q_pe.shape[1]
    rot_dim = q_pe.shape[2]
    kv_lora_rank = kv_c.shape[1]
    block_size = kv_cache.shape[1]
    
    BLOCK_SIZE = 64
    # 2D grid: (num_tokens, num_q_heads) for better parallelism
    grid = (num_tokens, num_q_heads)
    
    rope_and_cache_fused_kernel[grid](
        positions, q_pe, k_pe, kv_c, rope_cos_sin_cache,
        kv_cache, kv_cache_slot_mapping,
        num_tokens, num_q_heads, rot_dim, kv_lora_rank,
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        k_pe.stride(0), k_pe.stride(1),
        kv_c.stride(0), kv_c.stride(1),
        rope_cos_sin_cache.stride(0), rope_cos_sin_cache.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), block_size,
        IS_NEOX=rope_is_neox,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return q_pe, k_pe, kv_cache


def run_fused_kernel_optimized(
    positions, q_pe, k_pe, kv_c, rope_cos_sin_cache,
    rope_is_neox, kv_cache_slot_mapping, kv_cache
):
    """Run the optimized fused Triton kernel
    
    Optimizations:
    1. Vectorized Q-RoPE (no loop when BLOCK_SIZE >= embed_dim)
    2. K-RoPE writes directly to k_pe AND cache (no re-read)
    3. Larger block size for KV_C copy
    """
    num_tokens = q_pe.shape[0]
    num_q_heads = q_pe.shape[1]
    rot_dim = q_pe.shape[2]
    kv_lora_rank = kv_c.shape[1]
    block_size = kv_cache.shape[1]
    
    # BLOCK_SIZE should be >= embed_dim (rot_dim // 2) for vectorized Q/K RoPE
    embed_dim = rot_dim // 2
    BLOCK_SIZE = max(64, embed_dim)  # Ensure we can vectorize
    KVC_BLOCK_SIZE = 256  # Larger block for KV_C copy (memory bandwidth)
    
    # 2D grid: (num_tokens, num_q_heads)
    grid = (num_tokens, num_q_heads)
    
    rope_and_cache_fused_kernel_optimized[grid](
        positions, q_pe, k_pe, kv_c, rope_cos_sin_cache,
        kv_cache, kv_cache_slot_mapping,
        num_tokens, num_q_heads, rot_dim, kv_lora_rank,
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        k_pe.stride(0), k_pe.stride(1),
        kv_c.stride(0), kv_c.stride(1),
        rope_cos_sin_cache.stride(0), rope_cos_sin_cache.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), block_size,
        IS_NEOX=rope_is_neox,
        BLOCK_SIZE=BLOCK_SIZE,
        KVC_BLOCK_SIZE=KVC_BLOCK_SIZE,
    )
    
    return q_pe, k_pe, kv_cache


# =============================================================================
# Test and Validation
# =============================================================================

def create_test_inputs(
    num_tokens=128,
    num_q_heads=32,
    rot_dim=64,
    kv_lora_rank=512,
    max_position=4096,
    num_blocks=256,
    block_size=16,
    device=DEVICE,
    dtype=torch.float16,
):
    """Create test inputs for validation"""
    # Positions (random valid positions)
    positions = torch.randint(0, max_position, (num_tokens,), device=device, dtype=torch.int64)
    
    # Q, K with positional encoding part
    q_pe = torch.randn(num_tokens, num_q_heads, rot_dim, device=device, dtype=dtype)
    k_pe = torch.randn(num_tokens, rot_dim, device=device, dtype=dtype)
    
    # KV content (LoRA rank)
    kv_c = torch.randn(num_tokens, kv_lora_rank, device=device, dtype=dtype)
    
    # RoPE cos/sin cache
    rope_cos_sin_cache = torch.randn(max_position, rot_dim, device=device, dtype=dtype)
    
    # Slot mapping - use unique slots to avoid conflicts (like real usage)
    # Each token gets a unique slot
    kv_cache_slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)
    
    # KV cache (ensure enough blocks for unique slots)
    kv_cache = torch.zeros(num_blocks, block_size, kv_lora_rank + rot_dim, device=device, dtype=dtype)
    
    return {
        'positions': positions,
        'q_pe': q_pe,
        'k_pe': k_pe,
        'kv_c': kv_c,
        'rope_cos_sin_cache': rope_cos_sin_cache,
        'kv_cache_slot_mapping': kv_cache_slot_mapping,
        'kv_cache': kv_cache,
    }


def validate_correctness(rtol=1e-2, atol=1e-2):
    """Validate Triton implementations against PyTorch reference"""
    print("=" * 70)
    print("MLA RoPE Fused Kernel - Correctness Validation")
    print("=" * 70)
    
    # Test parameters
    num_tokens = 128
    num_q_heads = 8
    rot_dim = 64
    kv_lora_rank = 512
    rope_is_neox = True
    
    print(f"\nTest Config:")
    print(f"  num_tokens={num_tokens}, num_q_heads={num_q_heads}")
    print(f"  rot_dim={rot_dim}, kv_lora_rank={kv_lora_rank}")
    print(f"  rope_is_neox={rope_is_neox}")
    
    # Create inputs
    inputs = create_test_inputs(
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        rot_dim=rot_dim,
        kv_lora_rank=kv_lora_rank,
    )
    
    # =========================================================================
    # Test 1: PyTorch Reference
    # =========================================================================
    print("\n" + "-" * 50)
    print("Running PyTorch Reference...")
    
    ref_q_pe = inputs['q_pe'].clone()
    ref_k_pe = inputs['k_pe'].clone()
    ref_kv_cache = inputs['kv_cache'].clone()
    
    concat_and_cache_mla_rope_reference(
        inputs['positions'],
        ref_q_pe, ref_k_pe,
        inputs['kv_c'],
        inputs['rope_cos_sin_cache'],
        rope_is_neox,
        inputs['kv_cache_slot_mapping'],
        ref_kv_cache,
    )
    
    # =========================================================================
    # Test 2: Separate Triton Kernels
    # =========================================================================
    print("Running Separate Triton Kernels...")
    
    sep_q_pe = inputs['q_pe'].clone()
    sep_k_pe = inputs['k_pe'].clone()
    sep_kv_cache = inputs['kv_cache'].clone()
    
    run_separate_kernels(
        inputs['positions'],
        sep_q_pe, sep_k_pe,
        inputs['kv_c'],
        inputs['rope_cos_sin_cache'],
        rope_is_neox,
        inputs['kv_cache_slot_mapping'],
        sep_kv_cache,
    )
    
    # Validate
    q_pe_match = torch.allclose(ref_q_pe, sep_q_pe, rtol=rtol, atol=atol)
    k_pe_match = torch.allclose(ref_k_pe, sep_k_pe, rtol=rtol, atol=atol)
    kv_cache_match = torch.allclose(ref_kv_cache, sep_kv_cache, rtol=rtol, atol=atol)
    
    print(f"  Q_PE match: {'✓' if q_pe_match else '✗'}")
    print(f"  K_PE match: {'✓' if k_pe_match else '✗'}")
    print(f"  KV_Cache match: {'✓' if kv_cache_match else '✗'}")
    
    if not q_pe_match:
        diff = (ref_q_pe - sep_q_pe).abs()
        print(f"    Q_PE max diff: {diff.max().item():.6f}, mean diff: {diff.mean().item():.6f}")
    if not k_pe_match:
        diff = (ref_k_pe - sep_k_pe).abs()
        print(f"    K_PE max diff: {diff.max().item():.6f}, mean diff: {diff.mean().item():.6f}")
    if not kv_cache_match:
        diff = (ref_kv_cache - sep_kv_cache).abs()
        print(f"    KV_Cache max diff: {diff.max().item():.6f}, mean diff: {diff.mean().item():.6f}")
    
    # =========================================================================
    # Test 3: Fused Triton Kernel
    # =========================================================================
    print("\nRunning Fused Triton Kernel...")
    
    fused_q_pe = inputs['q_pe'].clone()
    fused_k_pe = inputs['k_pe'].clone()
    fused_kv_cache = inputs['kv_cache'].clone()
    
    run_fused_kernel(
        inputs['positions'],
        fused_q_pe, fused_k_pe,
        inputs['kv_c'],
        inputs['rope_cos_sin_cache'],
        rope_is_neox,
        inputs['kv_cache_slot_mapping'],
        fused_kv_cache,
    )
    
    # Validate
    q_pe_match = torch.allclose(ref_q_pe, fused_q_pe, rtol=rtol, atol=atol)
    k_pe_match = torch.allclose(ref_k_pe, fused_k_pe, rtol=rtol, atol=atol)
    kv_cache_match = torch.allclose(ref_kv_cache, fused_kv_cache, rtol=rtol, atol=atol)
    
    print(f"  Q_PE match: {'✓' if q_pe_match else '✗'}")
    print(f"  K_PE match: {'✓' if k_pe_match else '✗'}")
    print(f"  KV_Cache match: {'✓' if kv_cache_match else '✗'}")
    
    if not q_pe_match:
        diff = (ref_q_pe - fused_q_pe).abs()
        print(f"    Q_PE max diff: {diff.max().item():.6f}, mean diff: {diff.mean().item():.6f}")
    if not k_pe_match:
        diff = (ref_k_pe - fused_k_pe).abs()
        print(f"    K_PE max diff: {diff.max().item():.6f}, mean diff: {diff.mean().item():.6f}")
    if not kv_cache_match:
        diff = (ref_kv_cache - fused_kv_cache).abs()
        print(f"    KV_Cache max diff: {diff.max().item():.6f}, mean diff: {diff.mean().item():.6f}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    all_passed = q_pe_match and k_pe_match and kv_cache_match
    print("\n" + "=" * 70)
    print(f"Validation Result: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
    print("=" * 70)
    
    return all_passed


def benchmark_performance(num_warmup=10, num_repeat=100):
    """Benchmark performance of different implementations"""
    print("\n" + "=" * 70)
    print("MLA RoPE Fused Kernel - Performance Benchmark")
    print("=" * 70)
    
    # Test parameters - realistic MLA sizes
    configs = [
        {'num_tokens': 128, 'num_q_heads': 8, 'rot_dim': 64, 'kv_lora_rank': 512},
        {'num_tokens': 256, 'num_q_heads': 8, 'rot_dim': 64, 'kv_lora_rank': 512},
        {'num_tokens': 512, 'num_q_heads': 16, 'rot_dim': 64, 'kv_lora_rank': 512},
        {'num_tokens': 1024, 'num_q_heads': 32, 'rot_dim': 64, 'kv_lora_rank': 512},
    ]
    
    rope_is_neox = True
    
    for config in configs:
        print(f"\nConfig: {config}")
        
        inputs = create_test_inputs(**config)
        
        # Warmup and benchmark separate kernels
        for _ in range(num_warmup):
            sep_q_pe = inputs['q_pe'].clone()
            sep_k_pe = inputs['k_pe'].clone()
            sep_kv_cache = inputs['kv_cache'].clone()
            run_separate_kernels(
                inputs['positions'], sep_q_pe, sep_k_pe,
                inputs['kv_c'], inputs['rope_cos_sin_cache'],
                rope_is_neox, inputs['kv_cache_slot_mapping'], sep_kv_cache,
            )
        
        torch.cuda.synchronize()
        import time
        start = time.perf_counter()
        for _ in range(num_repeat):
            sep_q_pe = inputs['q_pe'].clone()
            sep_k_pe = inputs['k_pe'].clone()
            sep_kv_cache = inputs['kv_cache'].clone()
            run_separate_kernels(
                inputs['positions'], sep_q_pe, sep_k_pe,
                inputs['kv_c'], inputs['rope_cos_sin_cache'],
                rope_is_neox, inputs['kv_cache_slot_mapping'], sep_kv_cache,
            )
        torch.cuda.synchronize()
        sep_time = (time.perf_counter() - start) / num_repeat * 1000
        
        # Warmup and benchmark fused kernel
        for _ in range(num_warmup):
            fused_q_pe = inputs['q_pe'].clone()
            fused_k_pe = inputs['k_pe'].clone()
            fused_kv_cache = inputs['kv_cache'].clone()
            run_fused_kernel(
                inputs['positions'], fused_q_pe, fused_k_pe,
                inputs['kv_c'], inputs['rope_cos_sin_cache'],
                rope_is_neox, inputs['kv_cache_slot_mapping'], fused_kv_cache,
            )
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_repeat):
            fused_q_pe = inputs['q_pe'].clone()
            fused_k_pe = inputs['k_pe'].clone()
            fused_kv_cache = inputs['kv_cache'].clone()
            run_fused_kernel(
                inputs['positions'], fused_q_pe, fused_k_pe,
                inputs['kv_c'], inputs['rope_cos_sin_cache'],
                rope_is_neox, inputs['kv_cache_slot_mapping'], fused_kv_cache,
            )
        torch.cuda.synchronize()
        fused_time = (time.perf_counter() - start) / num_repeat * 1000
        
        speedup = sep_time / fused_time
        print(f"  Separate kernels: {sep_time:.4f} ms")
        print(f"  Fused kernel:     {fused_time:.4f} ms")
        print(f"  Speedup:          {speedup:.2f}x")


def main():
    print("=" * 70)
    print("MLA RoPE Fused Kernel - Triton Implementation")
    print("=" * 70)
    print("\nThis implements the concat_and_cache_mla_rope_fused CUDA kernel in Triton")
    print("Decomposed into 3 independent kernels that can be auto-fused:")
    print("  1. rope_apply_q_kernel - RoPE on Q (multi-head)")
    print("  2. rope_apply_k_kernel - RoPE on K (single head)")
    print("  3. kv_cache_write_kernel - Write K+KV_C to cache")
    print("")
    
    # Run validation
    passed = validate_correctness()
    
    if passed:
        # Run benchmark
        benchmark_performance()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
