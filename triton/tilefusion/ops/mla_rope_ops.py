"""
MLA RoPE Kernel Ops for tilefusion

This module provides the TTIR generators and kernel registration for the 
MLA RoPE + Cache Write kernels, enabling automatic fusion via tilefusion.

The three kernels are:
1. rope_q: Apply RoPE to Q tensor (multi-head)
2. rope_k: Apply RoPE to K tensor (single head)
3. cache_write: Write K_PE and KV_C to KV cache
"""

import torch
import triton
import triton.language as tl
from tilefusion.utils.utils import DEVICE


# =============================================================================
# Triton Kernels (used to generate TTIR)
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
    
    Grid: (num_tokens,) - One program per token, handles all heads linearized.
    """
    token_idx = tl.program_id(0)
    pos = tl.load(positions_ptr + token_idx)
    
    embed_dim = rot_dim // 2
    cos_sin_base = cos_sin_cache_ptr + pos * cos_sin_stride_pos
    
    # Linearized loop over all (head * pair) elements
    num_pairs_total = num_q_heads * embed_dim
    for i_start in range(0, num_pairs_total, BLOCK_SIZE):
        i_offs = i_start + tl.arange(0, BLOCK_SIZE)
        mask = i_offs < num_pairs_total
        
        # Unpack head and pair indices
        head_idx = i_offs // embed_dim
        pair_idx = i_offs % embed_dim
        
        # Load cos/sin
        cos = tl.load(cos_sin_base + pair_idx * cos_sin_stride_dim, mask=mask, other=0.0)
        sin = tl.load(cos_sin_base + (pair_idx + embed_dim) * cos_sin_stride_dim, mask=mask, other=0.0)
        
        # Compute memory indices
        if IS_NEOX:
            idx_x = pair_idx
            idx_y = embed_dim + pair_idx
        else:
            idx_x = pair_idx * 2
            idx_y = pair_idx * 2 + 1
            
        # Pointers to the specific head
        q_ptr_base = q_pe_ptr + token_idx * q_pe_stride_token + head_idx * q_pe_stride_head
        
        x_src = tl.load(q_ptr_base + idx_x * q_pe_stride_dim, mask=mask, other=0.0)
        y_src = tl.load(q_ptr_base + idx_y * q_pe_stride_dim, mask=mask, other=0.0)
        
        x_dst = x_src * cos - y_src * sin
        y_dst = y_src * cos + x_src * sin
        
        tl.store(q_ptr_base + idx_x * q_pe_stride_dim, x_dst, mask=mask)
        tl.store(q_ptr_base + idx_y * q_pe_stride_dim, y_dst, mask=mask)

@triton.jit
def rope_apply_k_kernel(
    k_pe_ptr,                    # [num_tokens, rot_dim]
    cos_sin_cache_ptr,           # [max_position, rot_dim]
    positions_ptr,               # [num_tokens]
    slot_mapping_ptr,            # [num_tokens]
    num_tokens,
    rot_dim,
    k_pe_stride_token,
    k_pe_stride_dim,
    cos_sin_stride_pos,
    cos_sin_stride_dim,
    IS_NEOX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply RoPE to K tensor (single head)"""
    token_idx = tl.program_id(0)
    pos = tl.load(positions_ptr + token_idx)
    
    embed_dim = rot_dim // 2
    cos_sin_base = cos_sin_cache_ptr + pos * cos_sin_stride_pos
    k_pe_token_ptr = k_pe_ptr + token_idx * k_pe_stride_token
    
    for pair_start in range(0, embed_dim, BLOCK_SIZE):
        pair_offs = pair_start + tl.arange(0, BLOCK_SIZE)
        mask = pair_offs < embed_dim
        
        cos = tl.load(cos_sin_base + pair_offs * cos_sin_stride_dim, mask=mask, other=0.0)
        sin = tl.load(cos_sin_base + (pair_offs + embed_dim) * cos_sin_stride_dim, mask=mask, other=0.0)
        
        if IS_NEOX:
            idx_x = pair_offs
            idx_y = embed_dim + pair_offs
        else:
            idx_x = pair_offs * 2
            idx_y = pair_offs * 2 + 1
        
        x_src = tl.load(k_pe_token_ptr + idx_x * k_pe_stride_dim, mask=mask, other=0.0)
        y_src = tl.load(k_pe_token_ptr + idx_y * k_pe_stride_dim, mask=mask, other=0.0)
        
        x_dst = x_src * cos - y_src * sin
        y_dst = y_src * cos + x_src * sin
        
        tl.store(k_pe_token_ptr + idx_x * k_pe_stride_dim, x_dst, mask=mask)
        tl.store(k_pe_token_ptr + idx_y * k_pe_stride_dim, y_dst, mask=mask)


@triton.jit
def kv_cache_write_kernel(
    k_pe_ptr,                    # [num_tokens, rot_dim]
    kv_c_ptr,                    # [num_tokens, kv_lora_rank]
    kv_cache_ptr,                # [num_blocks, block_size, total_dim]
    slot_mapping_ptr,            # [num_tokens]
    num_tokens,
    rot_dim,
    kv_lora_rank,
    block_size,
    k_pe_stride_token,
    k_pe_stride_dim,
    kv_c_stride_token,
    kv_c_stride_dim,
    block_stride,
    entry_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Write to KV cache"""
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
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
    
    # Write K_PE
    k_pe_token_ptr = k_pe_ptr + token_idx * k_pe_stride_token
    kv_cache_k_ptr = kv_cache_entry_ptr + kv_lora_rank
    for i in range(0, rot_dim, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < rot_dim
        vals = tl.load(k_pe_token_ptr + offs * k_pe_stride_dim, mask=mask, other=0.0)
        tl.store(kv_cache_k_ptr + offs, vals, mask=mask)



# =============================================================================
# TTIR Generators
# =============================================================================

def _ttir_of_rope_q(num_tokens: int, num_q_heads: int, rot_dim: int, BM: int = 64, is_neox: bool = True) -> str:
    """Generate TTIR for Q RoPE kernel"""
    max_position = 4096
    
    q_pe = torch.randn(num_tokens, num_q_heads, rot_dim, device=DEVICE, dtype=torch.float16)
    cos_sin_cache = torch.randn(max_position, rot_dim, device=DEVICE, dtype=torch.float16)
    positions = torch.randint(0, max_position, (num_tokens,), device=DEVICE, dtype=torch.int64)
    
    grid = (num_tokens, 1, 1)
    
    compiled = rope_apply_q_kernel[grid](
        q_pe, cos_sin_cache, positions,
        num_tokens, num_q_heads, rot_dim,
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        cos_sin_cache.stride(0), cos_sin_cache.stride(1),
        IS_NEOX=is_neox,
        BLOCK_SIZE=BM,
    )
    return compiled.asm['ttir']


def _ttir_of_rope_k(num_tokens: int, rot_dim: int, BM: int = 64, is_neox: bool = True) -> str:
    """Generate TTIR for K RoPE kernel"""
    max_position = 4096
    
    k_pe = torch.randn(num_tokens, rot_dim, device=DEVICE, dtype=torch.float16)
    cos_sin_cache = torch.randn(max_position, rot_dim, device=DEVICE, dtype=torch.float16)
    positions = torch.randint(0, max_position, (num_tokens,), device=DEVICE, dtype=torch.int64)
    slot_mapping = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    
    grid = (num_tokens,)
    
    compiled = rope_apply_k_kernel[grid](
        k_pe, cos_sin_cache, positions, slot_mapping,  # 9 / / 10
        num_tokens, rot_dim,                         
        k_pe.stride(0), k_pe.stride(1),               # 11 
        cos_sin_cache.stride(0), cos_sin_cache.stride(1), #
        IS_NEOX=is_neox,
        BLOCK_SIZE=BM,
    )
    return compiled.asm['ttir']


def _ttir_of_cache_write(num_tokens: int, rot_dim: int, kv_lora_rank: int, num_blocks: int = 256, block_size: int = 16,BM: int = 64) -> str:
    """Generate TTIR for cache write kernel"""
    k_pe = torch.randn(num_tokens, rot_dim, device=DEVICE, dtype=torch.float16)
    kv_c = torch.randn(num_tokens, kv_lora_rank, device=DEVICE, dtype=torch.float16)
    kv_cache = torch.zeros(num_blocks, block_size, kv_lora_rank + rot_dim, device=DEVICE, dtype=torch.float16)
    slot_mapping = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    
    grid = (num_tokens,)
    
    compiled = kv_cache_write_kernel[grid](
        k_pe, kv_c, kv_cache, slot_mapping,
        num_tokens, rot_dim, kv_lora_rank, block_size,
        k_pe.stride(0), k_pe.stride(1),
        kv_c.stride(0), kv_c.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), 
        BLOCK_SIZE=BM,
    )
    # print(compiled.asm['ttir'])
    return compiled.asm['ttir']


# =============================================================================
# Kernel Registration for tilefusion
# =============================================================================

def register_mla_rope_kernels():
    """Register MLA RoPE kernels to the global registry"""
    from tilefusion.core.kernel_registry import register_kernel, KernelMetadata, KernelType, TensorSpec
    
    # 1. Q RoPE Kernel
    register_kernel(KernelMetadata(
        name="rope_q",
        kernel_type=KernelType.ELEMENTWISE,
        family="rope",
        ttir_generator=lambda num_tokens, num_q_heads, rot_dim, BM=64, is_neox=True: 
            _ttir_of_rope_q(num_tokens, num_q_heads, rot_dim, BM, is_neox),
        ttir_symbol="rope_q_kernel_for_ttir",
        tensors=[
            TensorSpec("q_pe", "input_output", ["num_tokens", "num_q_heads", "rot_dim"], "f16",
                      arg_index=0, stride_indices=[6, 7, 8]),
            TensorSpec("cos_sin_cache", "input", ["max_position", "rot_dim"], "f16",
                      arg_index=1, stride_indices=[9, 10]),
            TensorSpec("positions", "input", ["num_tokens"], "i64",
                      arg_index=2, stride_indices=[]),
        ],
        grid_template=("num_tokens", "1", "1"),
        block_params=["BM"],
        required_params=["num_tokens", "num_q_heads", "rot_dim"],
        dim_arg_indices={"num_tokens": 3, "num_q_heads": 4, "rot_dim": 5},
        description="Apply RoPE to Q tensor (multi-head)"
    ))
    
    # 2. K RoPE Kernel
    register_kernel(KernelMetadata(
        name="rope_k",
        kernel_type=KernelType.ELEMENTWISE,
        family="rope",
        ttir_generator=lambda num_tokens, rot_dim, BM=64, is_neox=True:
            _ttir_of_rope_k(num_tokens, rot_dim, BM, is_neox),
        ttir_symbol="rope_k_kernel_for_ttir",
        tensors=[
            TensorSpec("k_pe", "input_output", ["num_tokens", "rot_dim"], "f16",
                      arg_index=0, stride_indices=[6, 7]),
            TensorSpec("cos_sin_cache", "input", ["max_position", "rot_dim"], "f16",
                      arg_index=1, stride_indices=[8, 9]),
            TensorSpec("positions", "input", ["num_tokens"], "i64",
                      arg_index=2, stride_indices=[]),
            TensorSpec("slot_mapping", "input", ["num_tokens"], "i64",
                      arg_index=3, stride_indices=[]),
        ],
        grid_template=("num_tokens", "1", "1"),
        block_params=["BM"],
        required_params=["num_tokens", "rot_dim"],
        dim_arg_indices={"num_tokens": 4, "rot_dim": 5},
        description="Apply RoPE to K tensor (single head)"
    ))
    
    # 3. Cache Write Kernel
    register_kernel(KernelMetadata(
        name="cache_write",
        kernel_type=KernelType.CUSTOM,
        family="cache",
        ttir_generator=lambda num_tokens, rot_dim, kv_lora_rank, num_blocks=256, block_size=16, BM=64:
            _ttir_of_cache_write(num_tokens, rot_dim, kv_lora_rank, num_blocks, block_size, BM),
        ttir_symbol="cache_write_kernel_for_ttir",
        tensors=[
            TensorSpec("k_pe", "input", ["num_tokens", "rot_dim"], "f16",
                      arg_index=0, stride_indices=[8, 9]),
            TensorSpec("kv_c", "input", ["num_tokens", "kv_lora_rank"], "f16",
                      arg_index=1, stride_indices=[10, 11]),
            TensorSpec("kv_cache", "output", ["num_blocks", "block_size", "total_dim"], "f16",
                      arg_index=2, stride_indices=[12, 13]),
            TensorSpec("slot_mapping", "input", ["num_tokens"], "i64",
                      arg_index=3, stride_indices=[]),
        ],
        grid_template=("num_tokens", "1", "1"),
        block_params=["BM"],
        required_params=["num_tokens", "rot_dim", "kv_lora_rank", "block_size"],
        dim_arg_indices={"num_tokens": 4, "rot_dim": 5, "kv_lora_rank": 6, "block_size": 7},
        description="Write K_PE and KV_C to KV cache"
    ))
    
    print("Registered MLA RoPE kernels: rope_q, rope_k, cache_write")


# Auto-register on import
# register_mla_rope_kernels()
