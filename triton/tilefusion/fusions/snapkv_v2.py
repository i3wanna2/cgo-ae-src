import math
import time
import tempfile
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.compiler import compile as triton_compile

# Use local module imports
from tilefusion.ops.gemm_mgrid_kernel import _ttir_of_gemm
from tilefusion.ops.gemm_mgrid_loopk_kernel import _ttir_of_gemm_loopk
from tilefusion.ops.softmax_blockM import _ttir_of_softmax
from tilefusion.utils.utils import DEVICE, build_combined_module, benchmark_performance, validate_correctness
from tilefusion.core.compiler import fuse_kernels_in_ttir, opt_kernels_in_ttir
from tilefusion.ops.elementwise import _ttir_of_addmask, _ttir_of_scale
from tilefusion.ops.sum import _ttir_of_sum
from tilefusion.ops.softmax_store_sum import _ttir_of_softmax_store, _ttir_of_softmax_recompute_ngrid
from tilefusion.ops.gemm_ngrid_loopm_kernel import _ttir_of_gemm_ngrid
from tilefusion.ops.elementwize_ngrid_loopm import _ttir_of_addmask_ngrid, _ttir_of_scale_ngrid
from tilefusion.ops.avg_pool1d import triton_avg_pool1d

def torch_snapkv_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor, scale: float, kernel_size: int = 5) -> tuple:
    """Pure-Torch reference for attention with SnapKV scoring:
    - Standard attention: O = softmax((Q @ K^T) * scale + Mask) @ V
    - SnapKV score: avg_pool1d(sum(attention_probs, dim=M), kernel_size)
    
    Shapes:
        Q: [batch, heads, M, K], Kmat: [batch, heads, N, K], V: [batch, heads, N, K], Mask: [batch, heads, M, N]
    Returns: 
        O: [batch, heads, M, K] (output attention)
        snapkv_score: [batch, heads, N] (pooled attention weights over sequence)
    """
    assert Q.is_cuda and Kmat.is_cuda and V.is_cuda and Mask.is_cuda, "torch ref expects CUDA tensors"
    
    # Compute attention scores
    scores = Q @ Kmat.transpose(-2, -1)  # [batch, heads, M, N]
    scores = scores * scale
    scores = scores + Mask
    
    # Compute attention probabilities
    probs = torch.softmax(scores.float(), dim=-1).to(Q.dtype)  # [batch, heads, M, N]
    
    # Compute output
    O = probs @ V  # [batch, heads, M, K]
    
    # SnapKV score: sum over query dimension, then avg_pool1d
    snapkv_score_sum = probs.sum(dim=2)  # [batch, heads, N]
    snapkv_score = F.avg_pool1d(
        snapkv_score_sum, 
        kernel_size=kernel_size, 
        padding=kernel_size // 2, 
        stride=1
    )  # [batch, heads, N]
    
    return O, snapkv_score, snapkv_score_sum

def build_and_compile_fused_kernel(args, args2, M, N, K, BM, BN, scale, batch, heads, num_warps=4, num_stages=2):
    """
    Build and compile fused attention kernel + SnapKV scoring pipeline.
    
    Pipeline 1 (M-grid): GEMM -> Scale -> AddMask -> Softmax -> GEMM (standard attention)
    Pipeline 2 (N-grid): GEMM -> Scale -> AddMask -> Softmax -> Sum (H2O-style score, before pooling)
    """
    # ==================== Pipeline 1: M-grid Attention ====================
    current_ttir = _ttir_of_gemm(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_mgrid_kernel"

    stages = [
        ("scale_kernel", lambda: _ttir_of_scale(M, N, BM, BN, scale, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        ("add_mask_kernel", lambda: _ttir_of_addmask(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        ("softmax_fwd_kernel", lambda: _ttir_of_softmax_store(M, N, BM, BN, batch, heads),       
         [18,3,4,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10]),
        ("gemm_mgrid_loopk_kernel", lambda: _ttir_of_gemm_loopk(M, K, N, BM, BN, batch, heads),  
         [19,3,5,4,12,13,14], [0,3,4,5,6,7,8]),
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
    
    options = {'num_warps': num_warps, 'num_stages': num_stages}
    compiled = triton_compile(fused_path, options=options)
    grid = (triton.cdiv(M, BM), batch * heads, 1)
    
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled.asm['ttgir'])
        print(f"Fused M-grid module written to: {f.name}")

    # ==================== Pipeline 2: N-grid SnapKV Score (before pooling) ====================
    current_ttir = _ttir_of_gemm_ngrid(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_ngrid_kernel"
    
    stages2 = [
        ("scale_ngrid_kernel", lambda: _ttir_of_scale_ngrid(M, N, BM, BN, scale, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        ("add_mask_ngrid_kernel", lambda: _ttir_of_addmask_ngrid(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        ("softmax_recompute_ngrid_kernel", lambda: _ttir_of_softmax_recompute_ngrid(M, N, BM, BN, batch, heads),       
         [18,3,4,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10]),
        ("sum_h2o_kernel", lambda: _ttir_of_sum(M, N, BM, BN, batch, heads),  
         [19,3,4,12,13,14], [0,2,3,4,5,6]),
    ]

    fused_path2 = None
    for consumer_kernel_name, ttir_fn, prod_out_idx, cons_in_idx in stages2:
        consumer_ttir = ttir_fn()
        combined = build_combined_module(current_ttir, consumer_ttir, current_producer_name, consumer_kernel_name)
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
            f.write(combined)
            combined_path = f.name
        fused_path2, fused_ttir_text = fuse_kernels_in_ttir(
            combined_path,
            producer_kernel_name=current_producer_name,
            consumer_kernel_name=consumer_kernel_name,
            producer_output_arg_idx=prod_out_idx,
            consumer_input_arg_idx=cons_in_idx,
        )
        current_ttir = fused_ttir_text
        current_producer_name = f"{current_producer_name}_{consumer_kernel_name}"
    
    fused_path2, fused_ttir_text = opt_kernels_in_ttir(fused_path2, direction="1")
    
    compiled2 = triton_compile(fused_path2, options=options)
    grid2 = (triton.cdiv(N, BN), batch * heads, 1)
    
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled2.asm['ttgir'])
        print(f"Fused N-grid (SnapKV score) module written to: {f.name}")

    def fused_launcher():
        compiled[grid](*args)
        compiled2[grid2](*args2)
        
    return fused_launcher

def main():
    # Problem sizes
    batch = 1
    heads = 32
    M = 4096
    N = 4096
    K = 128
    scale = 1.0 / math.sqrt(K)
    kernel_size = 5  # SnapKV pooling kernel size
    
    print(f"Problem size: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")
    print(f"SnapKV kernel_size: {kernel_size}")

    # Prepare input data
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    
    # Causal mask
    causal_mask_bool = torch.triu(torch.ones(M, N, dtype=torch.bool, device=DEVICE), diagonal=1)
    causal_mask_float = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
    
    # Output buffers
    scores_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale_out_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    mask_out_fused = torch.full((batch, heads, M, N), -torch.inf, device=DEVICE, dtype=torch.float16)
    softmax_sum_out = torch.zeros(batch, heads, M, 1, device=DEVICE, dtype=torch.float32)
    softmax_out_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Out_fused = torch.zeros(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Out_h2o_fused = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.float16)  # Before pooling
    Out_pool_fused = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.float16)  
    
    # Args for Pipeline 1 (M-grid attention)
    args = [
        Q, Kmat, scores_fused,
        M, N, K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale_out_fused,
        scale,
        Mask, mask_out_fused,
        softmax_out_fused,
        softmax_sum_out,
        softmax_sum_out.stride(0), softmax_sum_out.stride(1),
        V, Out_fused,
        V.stride(0), V.stride(1), V.stride(2),
        Out_fused.stride(0), Out_fused.stride(1), Out_fused.stride(2),
    ]
    
    # Args for Pipeline 2 (N-grid SnapKV score before pooling)
    args2 = [
        Q, Kmat, scores_fused,
        M, N, K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale_out_fused,
        scale,
        Mask, mask_out_fused,
        softmax_out_fused,
        softmax_sum_out,
        softmax_sum_out.stride(0), softmax_sum_out.stride(1),
        Out_h2o_fused, 
        Out_h2o_fused.stride(0), Out_h2o_fused.stride(1),
    ]
    
    BM = 64
    BN = 64
    num_warps = 4
    num_stages = 3
    
    fused_launcher = build_and_compile_fused_kernel(args, args2, M, N, K, BM, BN, scale, batch, heads, num_warps, num_stages)
    
    # SnapKV-specific: Apply avg_pool1d after fused kernel
    def snapkv_fused_launcher():
        fused_launcher()
        triton_avg_pool1d(Out_h2o_fused, Out_pool_fused)  
        # Out_pool_fused.copy_(F.avg_pool1d(
        #     Out_h2o_fused, 
        #     kernel_size=kernel_size, 
        #     padding=kernel_size // 2, 
        #     stride=1
        # )) 
    
    def torch_baseline_launcher():
        return torch_snapkv_ref(Q, Kmat, V, Mask, scale, kernel_size)
    
    # Run and get outputs
    snapkv_fused_launcher()
    Out_ref, snapkv_score_ref, snapkv_score_sum = torch_baseline_launcher()
    
    # Validate
    print("\n" + "="*60)
    print("Validating Attention Output")
    print("="*60)
    validate_correctness(
        lambda: None, 
        lambda: None, 
        [Out_ref, snapkv_score_sum, snapkv_score_ref], 
        [Out_fused, Out_h2o_fused, Out_pool_fused],
        rtol=1e-2, 
        atol=1e-2
    )
    
    # Benchmark
    print("\n" + "="*60)
    print("Benchmarking Performance")
    print("="*60)
    benchmark_performance(torch_baseline_launcher, snapkv_fused_launcher)

if __name__ == "__main__":
    main()
