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
from tilefusion.ops.elementwise import _ttir_of_addmask, _ttir_of_scale_c, _ttir_of_log_neg
from tilefusion.ops.softmax_store_sum import _ttir_of_softmax_store, _ttir_of_softmax_recompute_ngrid, _ttir_of_softmax_reduce_kernel
from tilefusion.ops.gemm_ngrid_loopm_kernel import _ttir_of_gemm_ngrid
from tilefusion.ops.elementwize_ngrid_loopm import _ttir_of_addmask_ngrid, _ttir_of_scale_ngrid_c, _ttir_of_log_neg_ngrid, _ttir_of_add_ngrid
from tilefusion.ops.sum import _ttir_of_sum

def torch_keyformer_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor, exp_rand: torch.Tensor, scale: float, tau: float) -> tuple:
    """Pure-Torch reference for Keyformer attention."""
    assert Q.is_cuda and Kmat.is_cuda and V.is_cuda and Mask.is_cuda, "torch ref expects CUDA tensors"
    
    # Standard Attention
    scores = Q @ Kmat.transpose(-2, -1)
    scores = scores * scale
    scores = scores + Mask
    
    probs = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
    O = probs @ V
    
    # Keyformer Score
    gumbels = -torch.log(exp_rand)
    kf_score_scale = (scores.float() + gumbels) / tau
    kf_score_sm = torch.softmax(kf_score_scale, dim=-1)
    kf_score = kf_score_sm.sum(dim=2)
    
    return O, kf_score, kf_score_sm, kf_score_scale

def build_and_compile_fused_kernel(args, args2, args3, M, N, K, BM, BN, scale, tau, batch, heads, num_warps=4, num_stages=2):
    # ==================== Pipeline 1: M-grid Attention ====================
    current_ttir = _ttir_of_gemm(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_mgrid_kernel"

    stages = [
        # Stage 1: GEMM + Scale
        ("scale_kernel_c",             lambda: _ttir_of_scale_c(M, N, BM, BN, scale, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 2: (GEMM+Scale) + AddMask  
        ("add_mask_kernel",          lambda: _ttir_of_addmask(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        # Stage 3: (GEMM+Scale+AddMask) + Softmax
        ("softmax_kernel",           lambda: _ttir_of_softmax(M, N, BM, BN, batch, heads),       
         [17,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Note: gemm_loopk fusion disabled due to dependency issues - using torch.matmul instead
        ("gemm_mgrid_loopk_kernel", lambda: _ttir_of_gemm_loopk(M, K, N, BM, BN, batch, heads),  
         [18,3,4,12,13,14,9,10,11,6,7,8], [0,3,5,6,7,8,9,10,11,12,13,14]),
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
    
    fused_path, fused_ttir_text = opt_kernels_in_ttir(fused_path, mask_arg_index=16)
    
    options = {'num_warps': num_warps, 'num_stages': num_stages}
    compiled = triton_compile(fused_path, options=options)
    grid = (triton.cdiv(M, BM), batch * heads, 1)
    
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled.asm['ttgir'])
        print(f"Fused M-grid module written to: {f.name}")

    # ==================== Pipeline 2: N-grid Keyformer Score ====================
    BM = 64
    BN = 128
    current_ttir = _ttir_of_gemm(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_mgrid_kernel"

    stages2 = [
        # Stage 1: GEMM + Scale
        ("scale_kernel_c",             lambda: _ttir_of_scale_c(M, N, BM, BN, scale, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 2: (GEMM+Scale) + AddMask  
        ("add_mask_kernel",          lambda: _ttir_of_addmask(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        ("log_neg_kernel",           lambda: _ttir_of_log_neg(M, N, BM, BN, batch, heads),       
         [3,4,12,13,14,12,13,14], [2,3,4,5,6,7,8,9]),
        ("add_mask_kernel",          lambda: _ttir_of_addmask(M, N, BM, BN, batch, heads),       
         [17,19,3,4,12,13,14,12,13,14,12,13,14], [0,1,3,4,5,6,7,8,9,10,11,12,13]),
        ("scale_kernel_c",             lambda: _ttir_of_scale_c(M, N, BM, BN, 1.0 / tau, batch, heads),  
         [20,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        ("softmax_reduce_kernel",           lambda: _ttir_of_softmax_reduce_kernel(M, N, BM, BN, batch, heads),       
         [21,3,4,12,13,14,], [0,2,3,4,5,6,]),
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
    
    fused_path2, fused_ttir_text = opt_kernels_in_ttir(fused_path2,mask_arg_index=16)
    
    compiled2 = triton_compile(fused_path2, options=options)
    grid1 = (triton.cdiv(M, BM), batch * heads, 1)
    
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled2.asm['ttgir'])
        print(f"Fused N-grid (KF score) module written to: {f.name}")

    # # ==================== Pipeline 3: N-grid Keyformer Score ====================
    BM = 128
    BN = 64
    current_ttir = _ttir_of_gemm_ngrid(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_ngrid_kernel"
    stages = [
        # Stage 1: GEMM + Scale
        ("scale_ngrid_kernel_c",             lambda: _ttir_of_scale_ngrid_c(M, N, BM, BN, scale, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 2: (GEMM+Scale) + AddMask  
        ("add_mask_ngrid_kernel",          lambda: _ttir_of_addmask_ngrid(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        ("log_neg_ngrid_kernel",                 lambda: _ttir_of_log_neg_ngrid(M, N, BM, BN, batch, heads),       
         [3,4,12,13,14,12,13,14], [2,3,4,5,6,7,8,9]),
        ("add_mask_ngrid_kernel",          lambda: _ttir_of_addmask_ngrid(M, N, BM, BN, batch, heads),       
         [17,19,3,4,12,13,14,12,13,14,12,13,14], [0,1,3,4,5,6,7,8,9,10,11,12,13]),
        ("scale_ngrid_kernel_c",             lambda: _ttir_of_scale_ngrid_c(M, N, BM, BN, 1.0 / tau, batch, heads),  
         [20,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        ("softmax_recompute_ngrid_kernel", lambda: _ttir_of_softmax_recompute_ngrid(M, N, BM, BN, batch, heads),       
         [21,3,4,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10]),
        # Note: gemm_loopk fusion disabled due to dependency issues - using torch.matmul instead
        ("sum_h2o_kernel", lambda: _ttir_of_sum(M, N, BM, BN, batch, heads),  
         [22,3,4,12,13,14], [0,2,3,4,5,6]),
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
    fused_path, fused_ttir_text = opt_kernels_in_ttir(fused_path, direction="1", mask_arg_index=16)
    # Compile kernel with optimization parameters
    options = {
        'num_warps': num_warps,
        'num_stages': num_stages,
    }
    compiled3 = triton_compile(fused_path, options=options)
    grid2 = (triton.cdiv(N, BN), batch * heads, 1)
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled3.asm['ttir'])
        combined_path = f.name
        print(f"Fused2 module written to: {combined_path}")   

    def fused_launcher():
        compiled[grid](*args)
        compiled2[grid1](*args2)
        compiled3[grid2](*args3)

    return fused_launcher

def main():
    batch = 1
    heads = 32
    M = 2048
    N = 2048
    K = 128
    scale = 1.0 / math.sqrt(K)
    tau = 1.0
    
    print(f"Problem size: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")

    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    exp_rand = 1 + torch.randn(batch, heads, M, N, dtype=torch.float16, device=DEVICE).abs()
    
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
    
    # KF buffers
    kf_logits = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    kf_logits_scaled = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    softmax_sum_out = torch.zeros(batch, heads, M, 1, device=DEVICE, dtype=torch.float16)
    kf_score_fused = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.float16)
    
    # Args for Pipeline 1
    args = [
        Q, Kmat, scores_fused,
        M, N, K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale_out_fused,
        Mask, mask_out_fused,
        softmax_out_fused,
        V, Out_fused, K
    ]

    # Args for Pipeline 2: LogNeg + Add + Scale + Softmax + Sum
    args2 = [
        Q, Kmat, scores_fused,
        M, N, K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale_out_fused,
        Mask, mask_out_fused,
        exp_rand, kf_logits,
        mask_out_fused,
        kf_logits_scaled,
        softmax_sum_out, 
        softmax_sum_out.stride(0), softmax_sum_out.stride(1),
    ]
    
    args3 = [
        Q, Kmat, scores_fused,
        M, N, K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale_out_fused,
        Mask, mask_out_fused,
        exp_rand, kf_logits,
        mask_out_fused,
        kf_logits_scaled,
        softmax_out_fused, softmax_sum_out, 
        softmax_sum_out.stride(0), softmax_sum_out.stride(1),
        kf_score_fused, 
        kf_score_fused.stride(0), kf_score_fused.stride(1)    
    ]  
    
    BM = 64
    BN = 64
    launcher = build_and_compile_fused_kernel(args, args2, args3, M, N, K, BM, BN, scale, tau, batch, heads)
    
    # softmax_sum_out = torch.zeros(batch, heads, M, 1, device=DEVICE, dtype=torch.float32)
    # softmax_out_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    softmax_out_test = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    # Out_fused = torch.zeros(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Out_baseline = torch.zeros(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    # Out_h2o_fused = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.float16)  # Changed from M to N
    Out_h2o_baseline = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.float16)
    
    def torch_baseline_launcher():
        # torch_attention_ref returns (O, h2o_score)
        attn_out, h2o_out, sm_out, scaled = torch_keyformer_ref(Q, Kmat, V, Mask, exp_rand, scale, tau)
        Out_baseline.copy_(attn_out)
        Out_h2o_baseline.copy_(h2o_out)
        softmax_out_test.copy_(sm_out)
    
    print("\n" + "="*60)
    print("Validating H2O Scores")
    print("="*60)
    validate_correctness(
        torch_baseline_launcher, 
        launcher, 
        [Out_baseline , Out_h2o_baseline, ], 
        [Out_fused, kf_score_fused, ],
        rtol=1e-1, 
        atol=1e-1
    )
    benchmark_performance(torch_baseline_launcher, launcher)

if __name__ == "__main__":
    main()
