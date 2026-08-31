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
from tilefusion.ops.mask_any import _ttir_of_mask_any, mask_any_triton
from tilefusion.ops.softmax_store_sum import _ttir_of_softmax_store, _ttir_of_softmax_recompute, _ttir_of_softmax_recompute_ngrid
from tilefusion.ops.gemm_ngrid_loopm_kernel import _ttir_of_gemm_ngrid
from tilefusion.ops.elementwize_ngrid_loopm import _ttir_of_addmask_ngrid, _ttir_of_scale_ngrid

def torch_attention_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor, scale: float, corm_mask: torch.Tensor) -> tuple:
    """Pure-Torch reference for attention with Corm scoring:
    scores = softmax((Q @ K^T) * scale + Mask, dim=-1) @ V
    corm_score = any(probs >= corm_mask, dim=M)
    
    Shapes:
        Q: [batch, heads, M, K], Kmat: [batch, heads, K, N], V: [batch, heads, N, K], Mask: [batch, heads, M, N]
    Returns: 
        O: [batch, heads, M, K] with dtype == Q.dtype (output attention)
        corm_score: [batch, heads, N] (bool)
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
    # probs = torch.softmax(scores, dim=-1)  # softmax over N dimension: [batch, heads, M, N]
    
    # 2. 计算分子 (exp)
    # numerators.shape = [B, H, M, N]
    numerators = torch.exp(scores)

    # 3. 计算分母 (sum) - 这就是你想要的中间结果
    # denominators.shape = [B, H, M, 1]
    denominators = torch.sum(numerators, dim=-1, keepdim=True)

    # 4. 计算最终概率 (div)
    # probs.shape = [B, H, M, N]
    probs = numerators / denominators
    
    # Compute output
    O = probs @ v  # [batch, heads, M, N] @ [batch, heads, N, K] -> [batch, heads, M, K]
    
    # Compute Corm score
    corm_score = probs >= corm_mask
    corm_score = corm_score.any(dim=2)  # [batch, heads, M, N] -> [batch, heads, N]
    
    return O, corm_score, denominators

def build_and_compile_fused_kernel(args, args2, M, N, K, BM, BN, scale, batch, heads, num_warps=4, num_stages=2):
    """
    Build and compile a fused attention kernel with given block sizes.
    """
    # Build TTIR for Q @ K^T with 4D support
    current_ttir = _ttir_of_gemm(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_mgrid_kernel"

    # Define fusion stages with 4D support
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
        print(f"Fused module written to: {combined_path}")
        
        
    current_ttir = _ttir_of_gemm_ngrid(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_ngrid_kernel"
    stages = [
        # Stage 1: GEMM + Scale
        ("scale_ngrid_kernel",             lambda: _ttir_of_scale_ngrid(M, N, BM, BN, scale, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 2: (GEMM+Scale) + AddMask  
        ("add_mask_ngrid_kernel",          lambda: _ttir_of_addmask_ngrid(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        # Stage 3: (GEMM+Scale+AddMask) + Softmax
        ("softmax_recompute_ngrid_kernel",           lambda: _ttir_of_softmax_recompute_ngrid(M, N, BM, BN, batch, heads),       
         [18,3,4,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10]),
        # Stage 4: Softmax + MaskAny
        ("mask_any_kernel", lambda: _ttir_of_mask_any(M, N, BM, BN, batch, heads),  
         [19,3,4,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10]), # Map x_ptr, M, N, stride_xb, stride_xh, stride_xm
    ]

    fused_path = None
    for consumer_kernel_name, ttir_fn, prod_out_idx, cons_in_idx in stages:
        consumer_ttir = ttir_fn()
        combined = build_combined_module(current_ttir, consumer_ttir, current_producer_name, consumer_kernel_name)
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
            f.write(combined)
            combined_path = f.name
            print(f"Fused2 module written to: {combined_path}") 
        fused_path, fused_ttir_text = fuse_kernels_in_ttir(
            combined_path,
            producer_kernel_name=current_producer_name,
            consumer_kernel_name=consumer_kernel_name,
            producer_output_arg_idx=prod_out_idx,
            consumer_input_arg_idx=cons_in_idx,
        )
        current_ttir = fused_ttir_text
        current_producer_name = f"{current_producer_name}_{consumer_kernel_name}"
    fused_path, fused_ttir_text = opt_kernels_in_ttir(fused_path, direction="1")
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
        print(f"Fused2 module written to: {combined_path}")    
        
    def fused_launcher():
        compiled[grid](*args)
        compiled2[grid](*args2)

    return fused_launcher


def run_attention_with_config(config, args, args2, Q, Kmat, V, Mask, scale, M, N, K, batch, heads, warmup_iters=10, bench_iters=100):
    """
    Run the fused attention kernel with a specific configuration and benchmark it.
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
    K = 128      # K dimension (head_dim)
    scale = 1.0 / math.sqrt(128)

    print(f"Problem size: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")

    # Prepare input data (4D tensors)
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    causal_mask_bool = torch.triu(
        torch.ones(M, N, dtype=torch.bool, device=DEVICE),
        diagonal=1
    )  # shape: (M, N)
    causal_mask_float = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    # Step 3: 扩展到 (batch, heads, M, N)
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
    
    # Prepare Corm Mask
    corm_mask = torch.ones(M, N, dtype=torch.float16, device=DEVICE)
    for i in range(M):
        corm_mask[i] /= (i + 1)
    corm_mask_expanded = corm_mask.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)

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
    
    Out_corm_fused = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.bool)
    Out_corm_baseline = torch.zeros(batch, heads, N, device=DEVICE, dtype=torch.bool)
    
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
    
    args2 = [
        Q, Kmat, scores_fused,
        M, N, K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale_out_fused,  #15
        scale,
        Mask, mask_out_fused, #17
        softmax_out_fused,      #19
        softmax_sum_out,        #20
        softmax_sum_out.stride(0), softmax_sum_out.stride(1),
        corm_mask_expanded, # y_ptr
        Out_corm_fused,     # z_ptr
        # corm_mask_expanded.stride(0), corm_mask_expanded.stride(1), corm_mask_expanded.stride(2), 
        Out_corm_fused.stride(0), Out_corm_fused.stride(1), 
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
        fused_path = "/tmp/tmpcen0ubfp.ttir"
        options = {
            'num_warps': num_warps,
            'num_stages': num_stages,
        }
        compiled = triton_compile(fused_path, options=options)
        grid = (triton.cdiv(M, BM), batch * heads, 1)
    else :
        fused_launcher = build_and_compile_fused_kernel(args, args2, M, N, K, BM, BN, scale, batch, heads, num_warps, num_stages)
    
    def torch_baseline_launcher():
        # torch_attention_ref returns (O, corm_score)
        attn_out, corm_out, sm_out = torch_attention_ref(Q, Kmat, V, Mask, scale, corm_mask_expanded)
        Out_baseline.copy_(attn_out)
        Out_corm_baseline.copy_(corm_out)
        softmax_out_test.copy_(sm_out)
        
    # fused_launcher()
    # Benchmark and validate
    if not USE_NCU:
        # Validate Corm scores
        print("\n" + "="*60)
        print("Validating Corm Scores")
        print("="*60)
        validate_correctness(
            torch_baseline_launcher, 
            fused_launcher, 
            [Out_baseline, softmax_out_test, Out_corm_baseline,], 
            [Out_fused, softmax_sum_out, Out_corm_fused,],
            rtol=1e-1, 
            atol=1e-1
        )
        benchmark_performance(torch_baseline_launcher, fused_launcher)
    else :
        fused_launcher()

if __name__ == "__main__":
    main()
