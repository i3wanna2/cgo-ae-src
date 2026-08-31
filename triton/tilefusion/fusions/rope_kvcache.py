import json
import math
import time
import tempfile
import sys
import os
import torch
import torch.nn.functional as F
import triton
from triton.compiler import compile as triton_compile

# Add CUDA extension path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ops", "csrc", "mla_rope"))
import mla_rope_cuda

# Use local module imports (same style as other demos)
from tilefusion.ops.gemm_mgrid_kernel import _ttir_of_gemm
from tilefusion.ops.gemm_mgrid_loopk_kernel import _ttir_of_gemm_loopk
from tilefusion.ops.softmax_blockM import _ttir_of_softmax
from tilefusion.utils.utils import DEVICE, build_combined_module, benchmark_performance, validate_correctness
from tilefusion.core.compiler import fuse_kernels_in_ttir,opt_kernels_in_ttir
from tilefusion.ops.elementwise import _ttir_of_addmask, _ttir_of_scale
from tilefusion.ops.gems_attn import scaled_dot_product_attention
from tilefusion.ops.mla_rope_ops import _ttir_of_rope_q, _ttir_of_rope_k, _ttir_of_cache_write, rope_apply_q_kernel, rope_apply_k_kernel, kv_cache_write_kernel


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
        # if slot_idx >= 0:
        block_idx = slot_idx // block_size
        entry_idx = slot_idx % block_size
        
        # Write kv_c at the beginning
        kv_cache[block_idx, entry_idx, :kv_lora_rank] = kv_c[token_idx]
        # Write k_pe after kv_lora_rank
        kv_cache[block_idx, entry_idx, kv_lora_rank:] = k_pe[token_idx]
    
    return q_pe, k_pe, kv_cache

def triton_baseline(
    q_pe,                # [num_tokens, num_q_heads, rot_dim]
    rope_cos_sin_cache,  # [max_position, rot_dim]
    positions,           # [num_tokens]
    num_tokens,
    num_q_heads,
    rot_dim,
    k_pe,                # [num_tokens, rot_dim]
    kv_cache_slot_mapping,  # [num_tokens]
    kv_c,                # [num_tokens, kv_lora_rank]
    kv_cache,            # [num_blocks, block_size, kv_lora_rank + rot_dim]
    kv_lora_rank,
    block_size,
):
    BM = 64
    IS_NEOX = True
    grid = (num_tokens, 1, 1)
    rope_apply_q_kernel[grid](q_pe, rope_cos_sin_cache, positions, num_tokens, num_q_heads, rot_dim,
                            q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
                            rope_cos_sin_cache.stride(0), rope_cos_sin_cache.stride(1),
                            IS_NEOX=IS_NEOX, BLOCK_SIZE=BM)
    rope_apply_k_kernel[grid](k_pe, rope_cos_sin_cache, positions, kv_cache_slot_mapping, 
                                num_tokens, rot_dim, k_pe.stride(0), k_pe.stride(1),
                                rope_cos_sin_cache.stride(0), rope_cos_sin_cache.stride(1),
                                IS_NEOX=IS_NEOX, BLOCK_SIZE=BM)
    kv_cache_write_kernel[grid](k_pe, kv_c, kv_cache, kv_cache_slot_mapping, 
                                num_tokens, rot_dim, kv_lora_rank, block_size,
                                k_pe.stride(0), k_pe.stride(1),
                                kv_c.stride(0), kv_c.stride(1),
                                kv_cache.stride(0), kv_cache.stride(1), BLOCK_SIZE=BM)
    
def build_and_compile_fused_kernel(num_tokens, num_q_heads, rot_dim, kv_lora_rank, num_blocks, block_size, BM, num_warps=4, num_stages=2):

    current_ttir = _ttir_of_rope_q(num_tokens, num_q_heads, rot_dim, BM)
    current_producer_name = "rope_apply_q_kernel"

    # Define fusion stages with 4D support
    # Verified parameter mappings:
    # For now, only fuse up to softmax (GEMM_loopk has dependency issues)
    stages = [
        # Stage 1: GEMM + Scale
        ("rope_apply_k_kernel",             lambda: _ttir_of_rope_k(num_tokens, rot_dim, BM),  
         [1,2,3,5,8], [1,2,4,5,7]),
        # Stage 2: (GEMM+Scale) + AddMask  
        ("kv_cache_write_kernel",          lambda: _ttir_of_cache_write(num_tokens, rot_dim, kv_lora_rank, num_blocks, block_size, BM),    
         [9,10,3,5,11], [0,3,4,5,8]),
    ]

    fused_path = None
    for consumer_kernel_name, ttir_fn, prod_out_idx, cons_in_idx in stages:
        consumer_ttir = ttir_fn()
        combined = build_combined_module(current_ttir, consumer_ttir, current_producer_name, consumer_kernel_name)
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
            f.write(combined)
            combined_path = f.name
        fused_path, fused_ttir_text = fuse_kernels_in_ttir(
            combined_path,
            producer_kernel_name=current_producer_name,
            consumer_kernel_name=consumer_kernel_name,
            producer_output_arg_idx=prod_out_idx,
            consumer_input_arg_idx=cons_in_idx,
        )
        current_ttir = fused_ttir_text
        current_producer_name = f"{current_producer_name}_{consumer_kernel_name}"
    fused_path, fused_ttir_text = opt_kernels_in_ttir(fused_path)
    # Compile kernel with optimization parameters
    options = {
        'num_warps': num_warps,
        'num_stages': num_stages,
    }
    compiled = triton_compile(fused_path, options=options)
    grid = (num_tokens, 1, 1)
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled.asm['ttgir'])
        combined_path = f.name
        print(f"Fused module written to: {combined_path}")
    return compiled, grid


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
    
    # Q, K with positional encoding part - reduce scale to prevent overflow
    q_pe = 0.5 * torch.randn(num_tokens, num_q_heads, rot_dim, device=device, dtype=dtype)
    k_pe = 0.5 * torch.randn(num_tokens, rot_dim, device=device, dtype=dtype)
    
    # KV content (LoRA rank)
    kv_c = 0.5 * torch.randn(num_tokens, kv_lora_rank, device=device, dtype=dtype)
    
    # RoPE cos/sin cache - use real trigonometric values in [-1, 1]
    # This is critical to avoid values exploding
    t = torch.arange(max_position, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000**(torch.arange(0, rot_dim, 2, device=device, dtype=torch.float32) / rot_dim))
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    # Concatenate according to NeoX convention: [max_pos, embed_dim] + [max_pos, embed_dim]
    rope_cos_sin_cache = torch.cat([cos, sin], dim=-1)
    
    # Slot mapping - use unique slots
    kv_cache_slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64)
    
    # KV cache
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

def get_hw_name():
    """Get GPU hardware name formatted for JSON output"""
    if not torch.cuda.is_available():
        return "CPU"
    name = torch.cuda.get_device_name(0)
    return name.replace(" ", "_")


def run_single_config(num_tokens, num_q_heads, rot_dim, kv_lora_rank, num_blocks, block_size, rope_is_neox, hw_name):
    """Run benchmark and validation for a single num_tokens configuration.
    Returns a list of result dicts."""
    print(f"\n{'=' * 70}")
    print(f"num_tokens={num_tokens}, num_q_heads={num_q_heads}, rot_dim={rot_dim}, kv_lora_rank={kv_lora_rank}")
    print(f"{'=' * 70}")

    inputs = create_test_inputs(num_tokens=num_tokens, num_q_heads=num_q_heads, rot_dim=rot_dim, kv_lora_rank=kv_lora_rank,
                                num_blocks=num_blocks, block_size=block_size)

    BM = 64
    num_warps = 4
    num_stages = 3


    compiled, grid = build_and_compile_fused_kernel(num_tokens, num_q_heads, rot_dim, 
                                                            kv_lora_rank, num_blocks,
                                                            block_size, BM=BM, num_warps=num_warps, num_stages=num_stages)
 
    # # Clones for baseline validation
    # q_pe_ref = inputs['q_pe'].clone()
    # k_pe_ref = inputs['k_pe'].clone()
    # kv_cache_ref = inputs['kv_cache'].clone()

    # def torch_baseline_launcher():
    #     # Correct argument order: (positions, q_pe, k_pe, kv_c, rope_cos_sin_cache, rope_is_neox, slot_mapping, kv_cache)
    #     concat_and_cache_mla_rope_reference(
    #         inputs['positions'], q_pe_ref, k_pe_ref, 
    #         inputs['kv_c'], inputs['rope_cos_sin_cache'], 
    #         rope_is_neox, inputs['kv_cache_slot_mapping'], kv_cache_ref
    #     )

    # CUDA launcher
    q_pe_cuda = inputs['q_pe'].clone()
    k_pe_cuda = inputs['k_pe'].clone()
    kv_cache_cuda = inputs['kv_cache'].clone()

    def cuda_launcher():
        mla_rope_cuda.concat_and_cache_mla_rope_fused(
            inputs['positions'],
            q_pe_cuda,
            k_pe_cuda,
            inputs['kv_c'],
            inputs['rope_cos_sin_cache'],
            rope_is_neox,
            inputs['kv_cache_slot_mapping'],
            kv_cache_cuda,
        )

    # # Triton Fused launcher
    # q_pe_triton = inputs['q_pe'].clone()
    # k_pe_triton = inputs['k_pe'].clone()
    # kv_cache_triton = inputs['kv_cache'].clone()
    
    # def triton_fused_launcher():
    #     args_triton = [
    #         q_pe_triton, inputs['rope_cos_sin_cache'], inputs['positions'],
    #         num_tokens, num_q_heads, rot_dim,
    #         q_pe_triton.stride(0), q_pe_triton.stride(1), 
    #         inputs['rope_cos_sin_cache'].stride(0),
    #         k_pe_triton, inputs['kv_cache_slot_mapping'], k_pe_triton.stride(0),
    #         inputs['kv_c'], kv_cache_triton,
    #         kv_lora_rank, block_size,
    #         inputs['kv_c'].stride(0),
    #         kv_cache_triton.stride(0), kv_cache_triton.stride(1),
    #     ]
    #     compiled[grid](*args_triton)

    # Benchmark and validate
    print("\nBenchmarking Torch vs CUDA vs Triton...")
    
    # def fresh_torch_launcher():
    #     q, k, c = inputs['q_pe'].clone(), inputs['k_pe'].clone(), inputs['kv_cache'].clone()
    #     concat_and_cache_mla_rope_reference(
    #         inputs['positions'], q, k, 
    #         inputs['kv_c'], inputs['rope_cos_sin_cache'], 
    #         rope_is_neox, inputs['kv_cache_slot_mapping'], c
    #     )

    def fresh_cuda_launcher():
        q, k, c = inputs['q_pe'].clone(), inputs['k_pe'].clone(), inputs['kv_cache'].clone()
        mla_rope_cuda.concat_and_cache_mla_rope_fused(
            inputs['positions'], q, k, inputs['kv_c'],
            inputs['rope_cos_sin_cache'], rope_is_neox,
            inputs['kv_cache_slot_mapping'], c,
        )

    def triton_baseline_launcher():
        q, k, c = inputs['q_pe'].clone(), inputs['k_pe'].clone(), inputs['kv_cache'].clone()
        args_triton = [
            q, inputs['rope_cos_sin_cache'], inputs['positions'],
            num_tokens, num_q_heads, rot_dim,
            k, inputs['kv_cache_slot_mapping'],
            inputs['kv_c'], c,
            kv_lora_rank, block_size,
        ]
        triton_baseline(*args_triton)

    def fresh_triton_launcher():
        q, k, c = inputs['q_pe'].clone(), inputs['k_pe'].clone(), inputs['kv_cache'].clone()
        args = [
            q, inputs['rope_cos_sin_cache'], inputs['positions'],
            num_tokens, num_q_heads, rot_dim,
            q.stride(0), q.stride(1), inputs['rope_cos_sin_cache'].stride(0),
            k, inputs['kv_cache_slot_mapping'], k.stride(0),
            inputs['kv_c'], c,
            kv_lora_rank, block_size,
            inputs['kv_c'].stride(0),
            c.stride(0), c.stride(1),
        ]
        compiled[grid](*args)

    print("\nBenchmarking CUDA vs Triton vs Triton_baseline...")
    bench_result = benchmark_performance(fresh_cuda_launcher, fresh_triton_launcher, triton_baseline_launcher,
                          labels=["CUDA", "Triton", "Triton_baseline"])

    # print("\nValidating Triton vs PyTorch...")
    # validate_correctness(
    #     torch_baseline_launcher, 
    #     triton_fused_launcher, 
    #     [q_pe_ref, k_pe_ref, kv_cache_ref], 
    #     [q_pe_triton, k_pe_triton, kv_cache_triton],
    #     rtol=1e-1, 
    #     atol=1e-1
    # )

    # # Reset/Clone for Triton vs CUDA validation

    # Build JSON records for this config
    shape_meaning = "num_q_heads,num_tokens,rot_dim,kv_lora_rank,block_size"
    shape_str = f"[{num_q_heads},{num_tokens},{rot_dim},{kv_lora_rank},{block_size}]"
    # label -> method name mapping
    method_map = {"CUDA": "CUDA", "Triton": "Triton_fused", "Triton_baseline": "Triton_baseline"}
    records = []
    for label, ms in bench_result["timings"]:
        record = {
            "hw": hw_name,
            "type": "op",
            "method": method_map.get(label, label),
            "op": "rope_kvcache",
            "seqlen": num_tokens,
            "shape_meaning": shape_meaning,
            "shape": shape_str,
            "time": "ms",
            "avg": round(ms, 3),
        }
        records.append(record)

    # Validation
    q_triton2 = inputs['q_pe'].clone()
    k_triton2 = inputs['k_pe'].clone()
    c_triton2 = inputs['kv_cache'].clone()

    def triton_baseline_launcher2():
        args_triton = [
            q_triton2, inputs['rope_cos_sin_cache'], inputs['positions'],
            num_tokens, num_q_heads, rot_dim,
            k_triton2, inputs['kv_cache_slot_mapping'],
            inputs['kv_c'], c_triton2,
            kv_lora_rank, block_size,
        ]
        triton_baseline(*args_triton)
        
    # def triton_launcher2():
    #     args = [
    #         q_triton2, inputs['rope_cos_sin_cache'], inputs['positions'],
    #         num_tokens, num_q_heads, rot_dim,
    #         q_triton2.stride(0), q_triton2.stride(1), 
    #         inputs['rope_cos_sin_cache'].stride(0),
    #         k_triton2, inputs['kv_cache_slot_mapping'], k_triton2.stride(0),
    #         inputs['kv_c'], c_triton2,
    #         kv_lora_rank, block_size,
    #         inputs['kv_c'].stride(0),
    #         c_triton2.stride(0), c_triton2.stride(1),
    #     ]
    #     compiled[grid](*args)

    print("\nValidating Triton vs CUDA...")
    validate_correctness(
        cuda_launcher,
        triton_baseline_launcher2,
        [q_pe_cuda, k_pe_cuda, kv_cache_cuda],
        [q_triton2, k_triton2, c_triton2],
        rtol=1e-1,
        atol=1e-1
    )

    return records


def main():
    """Validate Triton implementations against PyTorch reference"""
    print("=" * 70)
    print("MLA RoPE Fused Kernel - Correctness Validation")
    print("=" * 70)

    # Fixed parameters
    num_q_heads = 128
    rot_dim = 64
    kv_lora_rank = 512
    num_blocks = 512
    block_size = 16
    rope_is_neox = True

    # Sweep over num_tokens
    token_sizes = [128, 256, 512, 1024, 2048, 4096]

    hw_name = get_hw_name()
    print(f"\nDetected hardware: {hw_name}")

    all_records = []
    for num_tokens in token_sizes:
        records = run_single_config(
            num_tokens, num_q_heads, rot_dim, kv_lora_rank,
            num_blocks, block_size, rope_is_neox, hw_name,
        )
        all_records.extend(records)

    # Write results to JSON
    output_path = os.path.join(os.path.dirname(__file__), f"../result/rope_kvcache_results_{get_hw_name()}.json")
    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=4)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
