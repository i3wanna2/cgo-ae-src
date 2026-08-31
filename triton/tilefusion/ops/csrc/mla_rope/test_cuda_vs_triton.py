"""
Test CUDA kernel vs Triton implementation correctness
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import mla_rope_cuda

# Add tilefusion to path
sys.path.insert(0, '/home/meiziyuan/triton')
from tilefusion.fusions.mla_rope_fused import (
    run_separate_kernels, 
    run_fused_kernel,
    run_fused_kernel_optimized,
    create_test_inputs
)

DEVICE = "cuda"


def test_cuda_vs_triton():
    """Compare CUDA reference with Triton implementations"""
    print("=" * 70)
    print("CUDA Kernel vs Triton Implementation - Correctness Test")
    print("=" * 70)
    
    # Test configurations
    configs = [
        {'num_tokens': 32, 'num_q_heads': 8, 'rot_dim': 64, 'kv_lora_rank': 512},
        {'num_tokens': 128, 'num_q_heads': 8, 'rot_dim': 64, 'kv_lora_rank': 512},
        {'num_tokens': 256, 'num_q_heads': 16, 'rot_dim': 64, 'kv_lora_rank': 512},
    ]
    
    for rope_is_neox in [True, False]:
        print(f"\n{'='*50}")
        print(f"RoPE Style: {'NeoX' if rope_is_neox else 'GPT-J'}")
        print(f"{'='*50}")
        
        for config in configs:
            print(f"\nConfig: {config}")
            
            inputs = create_test_inputs(**config)
            
            # =========================================================
            # CUDA Reference
            # =========================================================
            cuda_q_pe = inputs['q_pe'].clone()
            cuda_k_pe = inputs['k_pe'].clone()
            cuda_kv_cache = inputs['kv_cache'].clone()
            
            mla_rope_cuda.concat_and_cache_mla_rope_fused(
                inputs['positions'],
                cuda_q_pe,
                cuda_k_pe,
                inputs['kv_c'],
                inputs['rope_cos_sin_cache'],
                rope_is_neox,
                inputs['kv_cache_slot_mapping'],
                cuda_kv_cache,
            )
            
            # =========================================================
            # Triton Separate Kernels
            # =========================================================
            triton_sep_q_pe = inputs['q_pe'].clone()
            triton_sep_k_pe = inputs['k_pe'].clone()
            triton_sep_kv_cache = inputs['kv_cache'].clone()
            
            run_separate_kernels(
                inputs['positions'],
                triton_sep_q_pe,
                triton_sep_k_pe,
                inputs['kv_c'],
                inputs['rope_cos_sin_cache'],
                rope_is_neox,
                inputs['kv_cache_slot_mapping'],
                triton_sep_kv_cache,
            )
            
            # =========================================================
            # Triton Fused Kernel
            # =========================================================
            triton_fused_q_pe = inputs['q_pe'].clone()
            triton_fused_k_pe = inputs['k_pe'].clone()
            triton_fused_kv_cache = inputs['kv_cache'].clone()
            
            run_fused_kernel(
                inputs['positions'],
                triton_fused_q_pe,
                triton_fused_k_pe,
                inputs['kv_c'],
                inputs['rope_cos_sin_cache'],
                rope_is_neox,
                inputs['kv_cache_slot_mapping'],
                triton_fused_kv_cache,
            )
            
            # =========================================================
            # Triton Fused Optimized Kernel
            # =========================================================
            triton_opt_q_pe = inputs['q_pe'].clone()
            triton_opt_k_pe = inputs['k_pe'].clone()
            triton_opt_kv_cache = inputs['kv_cache'].clone()
            
            run_fused_kernel_optimized(
                inputs['positions'],
                triton_opt_q_pe,
                triton_opt_k_pe,
                inputs['kv_c'],
                inputs['rope_cos_sin_cache'],
                rope_is_neox,
                inputs['kv_cache_slot_mapping'],
                triton_opt_kv_cache,
            )
            
            # =========================================================
            # Compare Results
            # =========================================================
            rtol, atol = 1e-2, 1e-2
            
            # Triton Separate vs CUDA
            sep_q_match = torch.allclose(cuda_q_pe, triton_sep_q_pe, rtol=rtol, atol=atol)
            sep_k_match = torch.allclose(cuda_k_pe, triton_sep_k_pe, rtol=rtol, atol=atol)
            sep_cache_match = torch.allclose(cuda_kv_cache, triton_sep_kv_cache, rtol=rtol, atol=atol)
            
            # Triton Fused vs CUDA
            fused_q_match = torch.allclose(cuda_q_pe, triton_fused_q_pe, rtol=rtol, atol=atol)
            fused_k_match = torch.allclose(cuda_k_pe, triton_fused_k_pe, rtol=rtol, atol=atol)
            fused_cache_match = torch.allclose(cuda_kv_cache, triton_fused_kv_cache, rtol=rtol, atol=atol)
            
            # Triton Optimized vs CUDA
            opt_q_match = torch.allclose(cuda_q_pe, triton_opt_q_pe, rtol=rtol, atol=atol)
            opt_k_match = torch.allclose(cuda_k_pe, triton_opt_k_pe, rtol=rtol, atol=atol)
            opt_cache_match = torch.allclose(cuda_kv_cache, triton_opt_kv_cache, rtol=rtol, atol=atol)
            
            print(f"  Triton Separate vs CUDA:")
            print(f"    Q_PE: {'✓' if sep_q_match else '✗'}, K_PE: {'✓' if sep_k_match else '✗'}, KV_Cache: {'✓' if sep_cache_match else '✗'}")
            
            if not sep_q_match:
                diff = (cuda_q_pe - triton_sep_q_pe).abs()
                print(f"      Q_PE max diff: {diff.max().item():.6f}")
            if not sep_k_match:
                diff = (cuda_k_pe - triton_sep_k_pe).abs()
                print(f"      K_PE max diff: {diff.max().item():.6f}")
            if not sep_cache_match:
                diff = (cuda_kv_cache - triton_sep_kv_cache).abs()
                print(f"      KV_Cache max diff: {diff.max().item():.6f}")
            
            print(f"  Triton Fused vs CUDA:")
            print(f"    Q_PE: {'✓' if fused_q_match else '✗'}, K_PE: {'✓' if fused_k_match else '✗'}, KV_Cache: {'✓' if fused_cache_match else '✗'}")
            
            if not fused_q_match:
                diff = (cuda_q_pe - triton_fused_q_pe).abs()
                print(f"      Q_PE max diff: {diff.max().item():.6f}")
            if not fused_k_match:
                diff = (cuda_k_pe - triton_fused_k_pe).abs()
                print(f"      K_PE max diff: {diff.max().item():.6f}")
            if not fused_cache_match:
                diff = (cuda_kv_cache - triton_fused_kv_cache).abs()
                print(f"      KV_Cache max diff: {diff.max().item():.6f}")
            
            print(f"  Triton Optimized vs CUDA:")
            print(f"    Q_PE: {'✓' if opt_q_match else '✗'}, K_PE: {'✓' if opt_k_match else '✗'}, KV_Cache: {'✓' if opt_cache_match else '✗'}")
            
            if not opt_q_match:
                diff = (cuda_q_pe - triton_opt_q_pe).abs()
                print(f"      Q_PE max diff: {diff.max().item():.6f}")
            if not opt_k_match:
                diff = (cuda_k_pe - triton_opt_k_pe).abs()
                print(f"      K_PE max diff: {diff.max().item():.6f}")
            if not opt_cache_match:
                diff = (cuda_kv_cache - triton_opt_kv_cache).abs()
                print(f"      KV_Cache max diff: {diff.max().item():.6f}")


def benchmark_cuda_vs_triton():
    """Benchmark CUDA vs Triton performance"""
    import time
    
    print("\n" + "=" * 70)
    print("CUDA Kernel vs Triton Implementation - Performance Benchmark")
    print("=" * 70)
    
    configs = [
        {'num_tokens': 128, 'num_q_heads': 8, 'rot_dim': 64, 'kv_lora_rank': 512},
        {'num_tokens': 256, 'num_q_heads': 16, 'rot_dim': 64, 'kv_lora_rank': 512},
        {'num_tokens': 512, 'num_q_heads': 32, 'rot_dim': 64, 'kv_lora_rank': 512},
        {'num_tokens': 1024, 'num_q_heads': 32, 'rot_dim': 64, 'kv_lora_rank': 512},
    ]
    
    rope_is_neox = True
    num_warmup = 10
    num_repeat = 100
    
    for config in configs:
        print(f"\nConfig: {config}")
        inputs = create_test_inputs(**config)
        
        # Warmup CUDA
        for _ in range(num_warmup):
            q = inputs['q_pe'].clone()
            k = inputs['k_pe'].clone()
            kv = inputs['kv_cache'].clone()
            mla_rope_cuda.concat_and_cache_mla_rope_fused(
                inputs['positions'], q, k, inputs['kv_c'],
                inputs['rope_cos_sin_cache'], rope_is_neox,
                inputs['kv_cache_slot_mapping'], kv,
            )
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_repeat):
            q = inputs['q_pe'].clone()
            k = inputs['k_pe'].clone()
            kv = inputs['kv_cache'].clone()
            mla_rope_cuda.concat_and_cache_mla_rope_fused(
                inputs['positions'], q, k, inputs['kv_c'],
                inputs['rope_cos_sin_cache'], rope_is_neox,
                inputs['kv_cache_slot_mapping'], kv,
            )
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / num_repeat * 1000
        
        # Warmup Triton Separate
        for _ in range(num_warmup):
            q = inputs['q_pe'].clone()
            k = inputs['k_pe'].clone()
            kv = inputs['kv_cache'].clone()
            run_separate_kernels(
                inputs['positions'], q, k, inputs['kv_c'],
                inputs['rope_cos_sin_cache'], rope_is_neox,
                inputs['kv_cache_slot_mapping'], kv,
            )
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_repeat):
            q = inputs['q_pe'].clone()
            k = inputs['k_pe'].clone()
            kv = inputs['kv_cache'].clone()
            run_separate_kernels(
                inputs['positions'], q, k, inputs['kv_c'],
                inputs['rope_cos_sin_cache'], rope_is_neox,
                inputs['kv_cache_slot_mapping'], kv,
            )
        torch.cuda.synchronize()
        triton_sep_time = (time.perf_counter() - start) / num_repeat * 1000
        
        # Warmup Triton Fused
        for _ in range(num_warmup):
            q = inputs['q_pe'].clone()
            k = inputs['k_pe'].clone()
            kv = inputs['kv_cache'].clone()
            run_fused_kernel(
                inputs['positions'], q, k, inputs['kv_c'],
                inputs['rope_cos_sin_cache'], rope_is_neox,
                inputs['kv_cache_slot_mapping'], kv,
            )
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_repeat):
            q = inputs['q_pe'].clone()
            k = inputs['k_pe'].clone()
            kv = inputs['kv_cache'].clone()
            run_fused_kernel(
                inputs['positions'], q, k, inputs['kv_c'],
                inputs['rope_cos_sin_cache'], rope_is_neox,
                inputs['kv_cache_slot_mapping'], kv,
            )
        torch.cuda.synchronize()
        triton_fused_time = (time.perf_counter() - start) / num_repeat * 1000
        
        # Warmup Triton Optimized
        for _ in range(num_warmup):
            q = inputs['q_pe'].clone()
            k = inputs['k_pe'].clone()
            kv = inputs['kv_cache'].clone()
            run_fused_kernel_optimized(
                inputs['positions'], q, k, inputs['kv_c'],
                inputs['rope_cos_sin_cache'], rope_is_neox,
                inputs['kv_cache_slot_mapping'], kv,
            )
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_repeat):
            q = inputs['q_pe'].clone()
            k = inputs['k_pe'].clone()
            kv = inputs['kv_cache'].clone()
            run_fused_kernel_optimized(
                inputs['positions'], q, k, inputs['kv_c'],
                inputs['rope_cos_sin_cache'], rope_is_neox,
                inputs['kv_cache_slot_mapping'], kv,
            )
        torch.cuda.synchronize()
        triton_opt_time = (time.perf_counter() - start) / num_repeat * 1000
        
        print(f"  CUDA:              {cuda_time:.4f} ms")
        print(f"  Triton Separate:   {triton_sep_time:.4f} ms ({cuda_time/triton_sep_time:.2f}x vs CUDA)")
        print(f"  Triton Fused:      {triton_fused_time:.4f} ms ({cuda_time/triton_fused_time:.2f}x vs CUDA)")
        print(f"  Triton Optimized:  {triton_opt_time:.4f} ms ({cuda_time/triton_opt_time:.2f}x vs CUDA)")


if __name__ == "__main__":
    test_cuda_vs_triton()
    benchmark_cuda_vs_triton()
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
