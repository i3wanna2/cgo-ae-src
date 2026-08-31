import math
import time
import tempfile
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.compiler import compile as triton_compile
from triton.runtime.driver import driver

# Use local module imports (same style as other demos)
from tilefusion.ops.gemm_mgrid_kernel import _ttir_of_gemm
from tilefusion.ops.gemm_mgrid_loopk_kernel import _ttir_of_gemm_loopk
from tilefusion.ops.softmax_blockM import _ttir_of_softmax, softmax_triton
from tilefusion.utils.utils import DEVICE, build_combined_module, benchmark_performance, validate_correctness
from tilefusion.core.compiler import fuse_kernels_in_ttir, opt_kernels_in_ttir
from tilefusion.ops.elementwise import _ttir_of_addmask, _ttir_of_scale
from tilefusion.core.autotuner import autotune_kernel, get_autotune_configs
from tilefusion.ops.sum import _ttir_of_sum, _ttir_of_sum_and_square
from tilefusion.ops.softmax_store_sum import _ttir_of_softmax_store, _ttir_of_softmax_recompute_ngrid
from tilefusion.ops.gemm_ngrid_loopm_kernel import _ttir_of_gemm_ngrid
from tilefusion.ops.elementwize_ngrid_loopm import _ttir_of_addmask_ngrid, _ttir_of_scale_ngrid

def torch_attention_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor, scale: float) -> tuple:
    """Pure-Torch reference for attention with ROCO (Robust Contextual Compression) scoring:
    scores = softmax((Q @ K^T) * scale + Mask, dim=-1) @ V
    roco_score = sum(attention_probs, dim=M) for token importance
    roco_sq_score = sum(attention_probs^2, dim=M) for variance-aware importance
    
    - Computes in fp32 for stability and casts back to Q.dtype (fp16) at the end.
    - All tensors are expected to be on the same CUDA device.
    
    Shapes:
        Q: [batch, heads, M, K], Kmat: [batch, heads, K, N], V: [batch, heads, N, K], Mask: [batch, heads, M, N]
    Returns: 
        O: [batch, heads, M, K] with dtype == Q.dtype (output attention)
        roco_score: [batch, heads, N] (sum of attention weights over sequence dimension)
        roco_sq_score: [batch, heads, N] (sum of squared attention weights)
    """
    assert Q.is_cuda and Kmat.is_cuda and V.is_cuda and Mask.is_cuda, "torch ref expects CUDA tensors"
    q = Q
    v = V
    mask = Mask
    
    # Compute attention scores
    scores = q @ Kmat.transpose(-2, -1)  # [batch, heads, M, K] @ [batch, heads, K, N] -> [batch, heads, M, N]
    scores = scores * scale
    scores = scores + mask
    
    # Compute attention probabilities
    numerators = torch.exp(scores)
    denominators = torch.sum(numerators, dim=-1, keepdim=True)
    probs = numerators / denominators  # [batch, heads, M, N]
    
    # Compute output
    O = probs @ v  # [batch, heads, M, N] @ [batch, heads, N, K] -> [batch, heads, M, K]
    
    # Compute ROCO scores: sum over query sequence dimension (dim=2)
    roco_score = probs.sum(dim=2)  # [batch, heads, M, N] -> [batch, heads, N]
    roco_sq_score = (probs ** 2).sum(dim=2)  # [batch, heads, M, N] -> [batch, heads, N]
    
    return O, roco_score, roco_sq_score, denominators

def build_and_compile_fused_kernel(args, args2, M, N, K, BM, BN, scale, batch, heads, num_warps=4, num_stages=2):
    """
    Build and compile a fused ROCO attention kernel with given block sizes.
    
    Args:
        M, N, K: Problem dimensions (per head)
        BM, BN: Block sizes
        scale: Scale factor for attention
        batch: Batch size
        heads: Number of attention heads
        num_warps: Number of warps per CTA (default: 4)
        num_stages: Number of pipeline stages (default: 2)
    
    Returns:
        Tuple of (compiled_kernel, grid)
    """
    # ==================== Pipeline 1: M-grid Attention ====================
    # Build TTIR for Q @ K^T with 4D support
    current_ttir = _ttir_of_gemm(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_mgrid_kernel"

    # Define fusion stages with 4D support
    stages = [
        # Stage 1: GEMM + Scale
        ("scale_kernel", lambda: _ttir_of_scale(M, N, BM, BN, scale, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 2: (GEMM+Scale) + AddMask  
        ("add_mask_kernel", lambda: _ttir_of_addmask(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        # Stage 3: (GEMM+Scale+AddMask) + Softmax (with sum output)
        ("softmax_fwd_kernel", lambda: _ttir_of_softmax_store(M, N, BM, BN, batch, heads),       
         [18,3,4,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10]),
        # Stage 4: (GEMM+Scale+AddMask+Softmax) + GEMM_loopk (P @ V)
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
    
    # Compile kernel with optimization parameters
    options = {
        'num_warps': num_warps,
        'num_stages': num_stages,
    }
    compiled = triton_compile(fused_path, options=options)
    grid = (triton.cdiv(M, BM), batch * heads, 1)
    
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled.asm['ttgir'])
        combined_path = f.name
        print(f"Fused M-grid module written to: {combined_path}")
        
    # ==================== Pipeline 2: N-grid ROCO Score (Sum & Square) ====================
    current_ttir = _ttir_of_gemm_ngrid(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_ngrid_kernel"
    
    stages2 = [
        # Stage 1: GEMM + Scale
        ("scale_ngrid_kernel", lambda: _ttir_of_scale_ngrid(M, N, BM, BN, scale, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 2: (GEMM+Scale) + AddMask  
        ("add_mask_ngrid_kernel", lambda: _ttir_of_addmask_ngrid(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        # Stage 3: (GEMM+Scale+AddMask) + Softmax (recompute normalization)
        ("softmax_recompute_ngrid_kernel", lambda: _ttir_of_softmax_recompute_ngrid(M, N, BM, BN, batch, heads),       
         [18,3,4,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10]),
        # Stage 4: Softmax + Sum & Square (roco_score = sum(probs), roco_sq_score = sum(probs^2))
        ("sum_and_square_kernel", lambda: _ttir_of_sum_and_square(M, N, BM, BN, batch, heads),  
         [19,3,4,12,13,14], [0,3,4,5,6,7]),
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
        combined_path = f.name
        print(f"Fused N-grid (roco_score & sq_score) module written to: {combined_path}")

    # Combined launcher
    def fused_launcher():
        compiled[grid](*args)     # M-grid: Attention output
        compiled2[grid2](*args2)  # N-grid: ROCO score & squared score

    return fused_launcher


def run_attention_with_config(config, args, args2, Q, Kmat, V, Mask, scale, M, N, K, batch, heads, warmup_iters=10, bench_iters=100):
    """
    Run the fused attention kernel with a specific configuration and benchmark it.
    
    Args:
        config: Configuration dict with 'BM', 'BN', 'num_stages', 'num_warps'
        Q, Kmat, V, Mask: Input tensors (4D: [batch, heads, M, K/N])
        scale: Scale factor
        M, N, K: Problem dimensions (per head)
        batch: Batch size
        heads: Number of attention heads
        warmup_iters: Number of warmup iterations
        bench_iters: Number of benchmark iterations
    
    Returns:
        Tuple of (output_tensor, elapsed_time_ms)
    """
    BM = config['BM']
    BN = config['BN']
    num_warps = config.get('num_warps', 4)
    num_stages = config.get('num_stages', 2)
    
    # Compile kernel with given block sizes
    launcher = build_and_compile_fused_kernel(args, args2, M, N, K, BM, BN, scale, batch, heads, num_warps, num_stages)
    Out_fused = torch.zeros(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    
    # Warmup
    for _ in range(warmup_iters):
        launcher()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(bench_iters):
        launcher()
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed_ms = (end - start) * 1000 / bench_iters
    return Out_fused, elapsed_ms


def main():
    # Problem sizes with 4D support (batch, heads, M, N/K)
    batch = 1
    heads = 32
    M = 4096    # M dimension (per head)
    N = 4096    # N dimension (per head)
    K = 128     # K dimension (head_dim)
    scale = 1.0 / math.sqrt(K)

    print(f"Problem size: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")

    # Prepare input data (4D tensors)
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    
    # Causal mask
    causal_mask_bool = torch.triu(
        torch.ones(M, N, dtype=torch.bool, device=DEVICE),
        diagonal=1
    )
    causal_mask_float = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
    
    # Allocate output buffers (4D)
    scores_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale_out_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    mask_out_fused = torch.full((batch, heads, M, N), 
                            fill_value=-torch.inf, 
                            device=DEVICE, 
                            dtype=torch.float16)
    softmax_sum_out = torch.zeros(batch, heads, M, 1, device=DEVICE, dtype=torch.float32)
    softmax_out_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    softmax_out_test = torch.zeros(batch, heads, M, 1, device=DEVICE, dtype=torch.float32)
    Out_fused = torch.zeros(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Out_baseline = torch.zeros(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    
    # ROCO score buffers
    Out_roco_fused = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.float16)
    Out_roco_baseline = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.float16)
    
    # ROCO squared score buffers
    softmax_sq_out = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Out_roco_sq_fused = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.float16)
    Out_roco_sq_baseline = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.float16)
    
    # Args for M-grid attention pipeline
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
    
    # Args for N-grid roco_score pipeline (reuses intermediate buffers)
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
        Out_roco_fused,
        Out_roco_sq_fused,
        Out_roco_fused.stride(0), Out_roco_fused.stride(1),
        Out_roco_sq_fused.stride(0), Out_roco_sq_fused.stride(1)
    ]
    
    DEBUG = True  

    
    if DEBUG:
        BM = 64
        BN = 64
        num_warps = 4
        num_stages = 3
    else:
        best_config, best_output, best_time, all_valid = autotune_kernel(
            get_autotune_configs(),
            run_attention_with_config,
            (args, args2, Q, Kmat, V, Mask, scale, M, N, K, batch, heads)
        )
        BM = best_config['BM']
        BN = best_config['BN']
        num_warps = best_config.get('num_warps', 4)
        num_stages = best_config.get('num_stages', 2)

    fused_launcher = build_and_compile_fused_kernel(args, args2, M, N, K, BM, BN, scale, batch, heads, num_warps, num_stages)
    
    def torch_baseline_launcher():
        # torch_attention_ref returns (O, roco_score, roco_sq_score, denominators)
        attn_out, roco_out, roco_sq_out, sm_out = torch_attention_ref(Q, Kmat, V, Mask, scale)
        Out_baseline.copy_(attn_out)
        Out_roco_baseline.copy_(roco_out)
        Out_roco_sq_baseline.copy_(roco_sq_out)
        softmax_out_test.copy_(sm_out)
        
    benchmark_performance(torch_baseline_launcher, fused_launcher)

if __name__ == "__main__":
    main()
