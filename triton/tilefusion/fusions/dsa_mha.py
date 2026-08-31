import torch
import torch.nn.functional as F

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
from tilefusion.core.compiler import fuse_kernels_in_ttir,opt_kernels_in_ttir, opt_kernels_in_ttir_test
from tilefusion.ops.elementwise import _ttir_of_addmask, _ttir_of_scale
from tilefusion.core.autotuner import autotune_kernel, get_autotune_configs
from flash_attn.flash_attn_interface import flash_attn_func
from tilefusion.ops.gems_attn import scaled_dot_product_attention
from tilefusion.ops.sum import _ttir_of_sum, sum_h2o_triton
from tilefusion.ops.softmax_store_sum import _ttir_of_softmax_store, _ttir_of_softmax_recompute, _ttir_of_softmax_recompute_ngrid
from tilefusion.ops.gemm_ngrid_loopm_kernel import _ttir_of_gemm_ngrid
from tilefusion.ops.elementwize_ngrid_loopm import _ttir_of_addmask_ngrid, _ttir_of_scale_ngrid
from tilefusion.ops.relu import _ttir_of_relu
from tilefusion.ops.mul import _ttir_of_mul
from tilefusion.ops.scatter import scatter_4d_triton, _ttir_of_scatter
from tilefusion.ops.topk import topk_bhmn, _ttir_of_topk_bhmn
from tilefusion.ops.sum import sum_dim1_triton, broadcast_bmn_to_bhmn

def torch_dsa_attention_ref(
    # --- MHA Inputs (for "Core Attention") ---
    Q_mha: torch.Tensor,
    K_mha: torch.Tensor,
    V_mha: torch.Tensor,
    # --- Indexer Inputs (for "Lightning Indexer") ---
    Q_idx: torch.Tensor,
    K_idx: torch.Tensor,
    W_idx: torch.Tensor,
    # --- Common Inputs ---
    Mask: torch.Tensor,  # Shape [B, M, N]
    index_mask: torch.Tensor,  # [B, M, N] <--- 新增参数 full_(-inf) 的掩码
    mha_scale: float,
    index_topk: int
) -> torch.Tensor:
    # --- 1. Indexer Path (to get topk_indices) ---
    # Implements: index_scores = Σ_j ( W_idx * ReLU(Q_idx @ K_idx.T) )
    B, H, M, _ = Q_idx.shape
    N = K_idx.shape[2]
    # dots: [B, H_idx, M, N]
    dots = Q_idx @ K_idx.transpose(-2, -1)
    
    # activated: [B, H_idx, M, N]
    activated = F.relu(dots)
    
    # W_idx: [B, H_idx, M, 1] (broadcasts over N)
    # weighted: [B, H_idx, M, N]
    weighted = activated * W_idx
    
    # index_scores: [B, M, N] (Sum over H_idx dim)
    index_scores = weighted.sum(dim=1)
    index_scores = index_scores.unsqueeze(1).expand(B, H, M, N).contiguous()
    index_scores_masked = index_scores + Mask

    k = min(index_topk, index_scores_masked.size(-1))
    topk_indices = index_scores_masked.topk(k, dim=-1)[1] # Shape: [B, M, k]

    # index_mask: [B, M, N]
    index_mask_scatter = torch.scatter(index_mask, -1, topk_indices, 0.0)
    
    # --- 2. MHA Path (using sparse mask from Indexer) ---
    
    # mha_scores: [B, H_mha, M, N]
    mha_scores = Q_mha @ K_mha.transpose(-2, -1)
    mha_scores = mha_scores * mha_scale
    
    total_scores = mha_scores + index_mask_scatter
    # Softmax
    probs = torch.softmax(total_scores, dim=-1) # Shape: [B, H_mha, M, N]

    # 输出
    # O: [B, H_mha, M, N] @ [B, H_mha, N, D_mha_v] -> [B, H_mha, M, D_mha_v]
    O = probs @ V_mha
    
    return O, topk_indices, index_mask_scatter, mha_scores


def build_and_compile_fused_kernel(args, args2, args3, M, N, K, BM, BN, scale, batch, heads, num_warps=4, num_stages=2):
    current_ttir = _ttir_of_gemm(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_mgrid_kernel"

    # Define fusion stages with 4D support
    # Verified parameter mappings:
    # For now, only fuse up to softmax (GEMM_loopk has dependency issues)
    stages = [
        # Stage 1: GEMM + Scale
        ("relu_kernel",             lambda: _ttir_of_relu(M, N, BM, BN, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 2: (GEMM+Scale) + AddMask  
        ("mul_kernel",          lambda: _ttir_of_mul(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
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
    fused_path, fused_ttir_text = opt_kernels_in_ttir(fused_path, "2")
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
        print(f"Fused module written to: {combined_path}")
    
    ###########################################################################
        # Build TTIR for Q @ K^T with 4D support
    current_ttir = _ttir_of_addmask(M, N, BM, BN, batch, heads)
    current_producer_name = "add_mask_kernel"

    # Define fusion stages with 4D support
    # Verified parameter mappings:
    # For now, only fuse up to softmax (GEMM_loopk has dependency issues)
    stages = [
        # Stage 1: GEMM + Scale
        ("topk_forward_bhmn",             lambda: _ttir_of_topk_bhmn(M, N, args[7], BM, BN, batch, heads),  
         [2,3,4,11,12,13,], [0,4,5,6,7,8]),
        # # Stage 2: (GEMM+Scale) + AddMask  
        # ("_scatter4d_kernel_scalar",          lambda: _ttir_of_scatter(M, N, args[7], BM, BN, batch, heads),       
        #  [15], [1]),
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
    fused_path, fused_ttir_text = opt_kernels_in_ttir(fused_path, direction="2")
    # Compile kernel with optimization parameters
    options = {
        'num_warps': num_warps,
        'num_stages': num_stages,
    }
    compiled2 = triton_compile(fused_path, options=options)
    grid = (triton.cdiv(M, BM), batch * heads, 1)
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled2.asm['ttgir'])
        combined_path = f.name
        print(f"Fused module written to: {combined_path}")

    ###########################################################################
    # Build TTIR for Q @ K^T with 4D support
    current_ttir = _ttir_of_gemm(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_mgrid_kernel"

    # Define fusion stages with 4D support
    # Verified parameter mappings:
    # For now, only fuse up to softmax (GEMM_loopk has dependency issues)
    stages = [
        # Stage 1: GEMM + Scale
        ("scale_kernel",             lambda: _ttir_of_scale(M, N, BM, BN, scale, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 2: (GEMM+Scale) + AddMask  
        ("add_mask_kernel",          lambda: _ttir_of_addmask(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        # Stage 3: (GEMM+Scale+AddMask) + Softmax
        ("softmax_fwd_kernel",           lambda: _ttir_of_softmax_store(M, N, BM, BN, batch, heads),       
         [18,3,4,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10]),
        # Note: gemm_loopk fusion disabled due to dependency issues - using torch.matmul instead
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
    fused_path, fused_ttir_text = opt_kernels_in_ttir(fused_path, direction="2")
    # Compile kernel with optimization parameters
    options = {
        'num_warps': num_warps,
        'num_stages': num_stages,
    }
    compiled3 = triton_compile(fused_path, options=options)
    grid = (triton.cdiv(M, BM), batch * heads, 1)
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled3.asm['ttgir'])
        combined_path = f.name
        print(f"Fused module written to: {combined_path}")  
        
    tmp = [args[0], args[1], args[2], 
           M, N, K, 
           args[0].stride(0), args[0].stride(1), args[0].stride(2),
           args[1].stride(0), args[1].stride(1), args[1].stride(2),
           args[2].stride(0), args[2].stride(1), args[2].stride(2),
           args[3], 
           args[4], args[5],
        ]

    def fused_launcher():
        compiled[grid](*tmp)
        # index_scores = args[5].sum(dim=1)
        index_scores = sum_dim1_triton(args[5]) 
        args2[0] = broadcast_bmn_to_bhmn(index_scores, heads)
        # args2[0] = index_scores.unsqueeze(1).expand(batch, heads, M, N).contiguous()
        compiled2[grid](*args2)
        scatter_4d_triton(args[8], args2[15], 0.0)
        args3[17].copy_(args[8])
        compiled3[grid](*args3)

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

    # Note: num_stages and num_warps are currently not directly used in compilation
    # They would need to be passed to the Triton compiler if supported
    
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
    K = 128      # K dimension (head_dim)
    scale = 1.0 / math.sqrt(128)

    print(f"Problem size: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")

    # Prepare input data (4D tensors)
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    
    Q_idx = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    K_idx = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    W_idx = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    # Create causal mask [B, M, N]
    causal_mask_bool = torch.triu(
        torch.ones(M, N, dtype=torch.bool, device=DEVICE),
        diagonal=1
    )  
    causal_mask_float = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
    mask_tmp_fused = torch.zeros(batch,heads,M,N, device=DEVICE, dtype=torch.float16)
    index_mask = torch.full((batch, heads, M, N), fill_value=-torch.inf, device=DEVICE, dtype=torch.float16)
    # top_k = torch.randint(1, N//2, (1,)).item()  # Random top-k for testing
    top_k = 2
    # Baseline middle outputs
    Out_baseline = torch.zeros(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    index_scores_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    index_scores_masked = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    y_vals = torch.zeros(batch, heads, M, top_k, device=DEVICE, dtype=torch.float16)
    y_indx = torch.zeros(batch, heads, M, top_k, device=DEVICE, dtype=torch.int32)
    # Allocate output buffers (4D)
    scores_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale_out_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    # mask_out_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    mask_out_fused = torch.full((batch, heads, M, N), 
                            fill_value=-torch.inf, 
                            device=DEVICE, 
                            dtype=torch.float16)
    softmax_sum_out = torch.zeros(batch, heads, M, 1, device=DEVICE, dtype=torch.float32)
    softmax_out_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    topk_indices_fused = torch.zeros(batch, heads, M, top_k, device=DEVICE, dtype=torch.int64)
    index_mask_bcasts_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.int64)
    Out_fused = torch.zeros(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    QK_idx = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    relu_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)  # Changed from M to N
    mul_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)

    args = [
        Q_idx, K_idx, QK_idx, relu_fused, W_idx, mul_fused, Mask, top_k, index_mask, topk_indices_fused,index_mask_bcasts_fused
    ]
    
    args2 = [index_scores_fused, Mask, index_scores_masked, 
        M, N,
        index_scores_fused.stride(0), index_scores_fused.stride(1), index_scores_fused.stride(2),
        Mask.stride(0), Mask.stride(1), Mask.stride(2),
        index_scores_masked.stride(0), index_scores_masked.stride(1), index_scores_masked.stride(2),
        y_vals,y_indx,
        heads,
        y_vals.stride(0), y_vals.stride(1), y_vals.stride(2), 
    ]
    
    args3 = [
        Q, Kmat, scores_fused,
        M, N, K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale_out_fused,
        scale,
        mask_tmp_fused, mask_out_fused,
        # Mask.stride(0), Mask.stride(1), Mask.stride(2),
        # mask_out_fused.stride(0), mask_out_fused.stride(1), mask_out_fused.stride(2),
        softmax_out_fused,
        softmax_sum_out,
        # softmax_out_fused.stride(0), softmax_out_fused.stride(1), softmax_out_fused.stride(2),
        softmax_sum_out.stride(0), softmax_sum_out.stride(1),
        V, Out_fused,
        V.stride(0), V.stride(1), V.stride(2),
        Out_fused.stride(0), Out_fused.stride(1), Out_fused.stride(2),
    ]
    
    DEBUG = True  
    USE_PATH = False
    USE_NCU = False
    
    if DEBUG:
        BM = 64
        BN = 64
        num_warps = 4
        num_stages = 3
    else :
        best_config, best_output, best_time, all_valid = autotune_kernel(
            get_autotune_configs(),
            run_attention_with_config,
            (args, args2, Q, Kmat, V, Mask, scale, M, N, K, batch, heads)
        )
        BM = best_config['BM']
        BN = best_config['BN']
        num_warps = best_config.get('num_warps', 4)
        num_stages = best_config.get('num_stages', 2)


    if USE_PATH:
        # /tmp/tmpuqzib7_f.ttir  tt.divis
        # /tmp/tmpfwhmmdij.ttir
        fused_path = "/tmp/tmpcen0ubfp.ttir"
        options = {
            'num_warps': num_warps,
            'num_stages': num_stages,
        }
        compiled = triton_compile(fused_path, options=options)
        grid = (triton.cdiv(M, BM), batch * heads, 1)
    else :
        fused_launcher = build_and_compile_fused_kernel(args, args2, args3, M, N, K, BM, BN, scale, batch, heads, num_warps, num_stages)
    def torch_baseline_launcher():
        # torch_attention_ref returns (O, h2o_score)
        attn_out, topk_indice, index_mask_bcast, mha_score = torch_dsa_attention_ref(Q, Kmat, V, Q_idx, K_idx, W_idx, Mask, index_mask, scale, top_k)
        Out_baseline.copy_(attn_out)

    if not USE_NCU:
        # Validate H2O scores
        print("\n" + "="*60)
        print("Validating H2O Scores")
        print("="*60)
        validate_correctness(
            torch_baseline_launcher, 
            fused_launcher, 
            [Out_baseline, ], 
            [Out_fused, ],
            rtol=1e-1, 
            atol=1e-1
        )
        benchmark_performance(torch_baseline_launcher, fused_launcher)
    else :
        fused_launcher()

if __name__ == "__main__":
    main()
