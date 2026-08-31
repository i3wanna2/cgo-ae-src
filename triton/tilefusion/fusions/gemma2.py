import math
import time
import tempfile
import torch
import torch.nn.functional as F
import triton
from triton.compiler import compile as triton_compile

# Use local module imports (same style as other demos)
from tilefusion.ops.gemm_mgrid_kernel import _ttir_of_gemm
from tilefusion.ops.gemm_mgrid_loopk_kernel import _ttir_of_gemm_loopk
from tilefusion.ops.softmax_blockM import _ttir_of_softmax
from tilefusion.utils.utils import DEVICE, build_combined_module, benchmark_performance, validate_correctness
from tilefusion.core.compiler import fuse_kernels_in_ttir, opt_kernels_in_ttir
from tilefusion.ops.elementwise import _ttir_of_addmask, _ttir_of_scale_c, _ttir_of_tanh
from tilefusion.core.autotuner import autotune_kernel


def torch_attention_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor, scale: float, logit_softcapping: float) -> torch.Tensor:
    """Pure-Torch reference for Gemma2 attention with logit softcapping:
    scores = (Q @ K^T) / scale
    scores = tanh(scores / logit_softcapping) * logit_softcapping
    scores = scores + mask
    probs = softmax(scores, dim=-1) @ V
    
    Shapes:
        Q: [batch, heads, M, K], Kmat: [batch, heads, K, N], V: [batch, heads, N, K], Mask: [batch, heads, M, N]
    Returns: O: [batch, heads, M, K] with dtype == Q.dtype
    """
    assert Q.is_cuda and Kmat.is_cuda and V.is_cuda and Mask.is_cuda, "torch ref expects CUDA tensors"
    scores = torch.matmul(Q, Kmat.transpose(-2, -1)) / scale  # [batch, heads, M, N]
    scores = scores / logit_softcapping
    scores = torch.tanh(scores)
    scores = scores * logit_softcapping
    scores = scores + Mask
    probs = F.softmax(scores.float(), dim=-1)
    O = torch.matmul(probs.to(Q.dtype), V)  # [batch, heads, M, K]
    return O


def get_autotune_configs(block_sizes=None, stages=None, warps=None):
    """
    Generate configurations for autotuning.
    
    Args:
        block_sizes: List of (BM, BN) tuples to test. Default: [(32,32), (32,64), (32,128), (64,32), (64,64), (64,128), (128,32), (128,64), (128,128)]
        stages: List of num_stages values to test. Default: [2, 3, 4, 5]
        warps: List of num_warps values to test. Default: [4, 8]
    
    Returns:
        List of configuration dictionaries
    """
    if block_sizes is None:
        block_sizes = [(BM, BN) for BM in [32, 64, 128] for BN in [32, 64, 128]]
    if stages is None:
        stages = [2, 3, 4, 5]
    if warps is None:
        warps = [4, 8]
    
    configs = []
    for BM, BN in block_sizes:
        for num_stages in stages:
            for num_warps in warps:
                configs.append({
                    'BM': BM,
                    'BN': BN,
                    'num_stages': num_stages,
                    'num_warps': num_warps,
                })
    return configs


def build_and_compile_fused_kernel(M, N, K, BM, BN, scale, logit_softcapping, batch, heads, num_warps=4, num_stages=2):
    """
    Build and compile a fused Gemma2 attention kernel with logit softcapping.
    
    Args:
        M, N, K: Problem dimensions (per head)
        BM, BN: Block sizes
        scale: Scale factor for attention (1/sqrt(head_dim))
        logit_softcapping: Softcapping value for tanh operation
        batch: Batch size
        heads: Number of attention heads
        num_warps: Number of warps per CTA (default: 4)
        num_stages: Number of pipeline stages (default: 2)
    
    Returns:
        Tuple of (compiled_kernel, grid)
    """
    # Build TTIR for Q @ K^T with 4D support
    current_ttir = _ttir_of_gemm(M, N, K, BM, BN, batch, heads)
    current_producer_name = "gemm_mgrid_kernel"

    # Define fusion stages with 4D support
    # Gemma2 attention: GEMM -> Scale1 -> Tanh -> Scale2 -> AddMask -> Softmax -> GEMM_loopk
    stages = [
        # Stage 1: GEMM + Scale (divide by scale)
        ("scale_kernel_c",             lambda: _ttir_of_scale_c(M, N, BM, BN, 1.0/scale, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 2: (GEMM+Scale1) + Scale (divide by logit_softcapping)
        ("scale_kernel_c",             lambda: _ttir_of_scale_c(M, N, BM, BN, 1.0/logit_softcapping, batch, heads),  
         [15,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 3: (GEMM+Scale1+Scale2) + Tanh
        ("tanh_kernel",              lambda: _ttir_of_tanh(M, N, BM, BN, batch, heads),
         [16,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 4: (GEMM+Scale1+Scale2+Tanh) + Scale (multiply by logit_softcapping)
        ("scale_kernel_c",             lambda: _ttir_of_scale_c(M, N, BM, BN, logit_softcapping, batch, heads),  
         [17,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 5: (GEMM+Scale1+Scale2+Tanh+Scale3) + AddMask  
        ("add_mask_kernel",          lambda: _ttir_of_addmask(M, N, BM, BN, batch, heads),       
         [18,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        # Stage 6: (GEMM+Scale1+Scale2+Tanh+Scale3+AddMask) + Softmax
        ("softmax_kernel",           lambda: _ttir_of_softmax(M, N, BM, BN, batch, heads),       
         [20,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 7: (Full pipeline) + GEMM_loopk
        ("gemm_mgrid_loopk_kernel", lambda: _ttir_of_gemm_loopk(M, K, N, BM, BN, batch, heads),  
         [21,3,4,12,13,14,9,10,11,6,7,8], [0,3,5,6,7,8,9,10,11,12,13,14]),
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
    
    fused_path, fused_ttir_text = opt_kernels_in_ttir(fused_path, mask_arg_index=19)
    
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
    return compiled, grid


def run_attention_with_config(config, Q, Kmat, V, Mask, scale, logit_softcapping, M, N, K, batch, heads, warmup_iters=10, bench_iters=100):
    """
    Run the fused attention kernel with a specific configuration and benchmark it.
    
    Args:
        config: Configuration dict with 'BM', 'BN', 'num_stages', 'num_warps'
        Q, Kmat, V, Mask: Input tensors (4D: [batch, heads, M, K/N])
        scale: Scale factor
        logit_softcapping: Softcapping value
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
    compiled, grid = build_and_compile_fused_kernel(M, N, K, BM, BN, scale, logit_softcapping, batch, heads, num_warps, num_stages)
    
    # Allocate output buffers (4D: [batch, heads, M, N/K])
    scores_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale1_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale2_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    tanh_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale3_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    mask_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    softmax_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Out_fused = torch.empty(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    
    # Prepare kernel arguments (4D strides)
    args = [
        Q, Kmat, scores_fused,
        M, N, K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale1_out_fused,
        1.0/scale,
        scale2_out_fused,
        1.0/logit_softcapping,
        tanh_out_fused,
        scale3_out_fused,
        logit_softcapping,
        Mask, mask_out_fused,
        softmax_out_fused,
        V, Out_fused,
        V.stride(0), V.stride(1), V.stride(2),
        Out_fused.stride(0), Out_fused.stride(1), Out_fused.stride(2),
    ]
    
    # Warmup
    for _ in range(warmup_iters):
        compiled[grid](*args)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(bench_iters):
        compiled[grid](*args)
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
    scale = math.sqrt(K)  # Scale by sqrt(head_dim)
    logit_softcapping = 50.0  # Gemma2 typical value

    print(f"Problem size: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")
    print(f"Scale: {scale}, Logit softcapping: {logit_softcapping}")

    # Prepare input data (4D tensors)
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    
    # Create causal mask with proper diagonal offset
    # mask = torch.triu(torch.full((M, N), -torch.inf), diagonal=N - M + 1)
    causal_mask_bool = torch.triu(
        torch.ones(M, N, dtype=torch.bool, device=DEVICE),
        diagonal=N - M + 1
    )
    causal_mask_float = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    
    # Expand to (batch, heads, M, N)
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
  
    # Allocate output buffers (4D)
    scores_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale1_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale2_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    tanh_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale3_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    mask_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    softmax_out_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Out_fused = torch.empty(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Out_baseline = torch.empty(batch, heads, M, K, device=DEVICE, dtype=torch.float16)

    DEBUG = True   
    USE_PATH = False
    
    if not DEBUG:
        best_config, best_output, best_time, all_valid = autotune_kernel(
            get_autotune_configs(),
            run_attention_with_config,
            (Q, Kmat, V, Mask, scale, logit_softcapping, M, N, K, batch, heads)
        )
        # Build and compile kernel with best config
        BM = best_config['BM']
        BN = best_config['BN']
        num_warps = best_config.get('num_warps', 4)
        num_stages = best_config.get('num_stages', 2)
    else:
        BM = 64
        BN = 64
        num_warps = 4
        num_stages = 3

    if USE_PATH:
        # For debugging: load from a specific TTIR file
        fused_path = "/tmp/tmpxxxxxxx.ttir"
        options = {
            'num_warps': num_warps,
            'num_stages': num_stages,
        }
        compiled = triton_compile(fused_path, options=options)
        grid = (triton.cdiv(M, BM), batch * heads, 1)
    else:
        compiled, grid = build_and_compile_fused_kernel(M, N, K, BM, BN, scale, logit_softcapping, batch, heads, num_warps, num_stages)

    args = [
        Q, Kmat, scores_fused,
        M, N, K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale1_out_fused,
        scale2_out_fused,
        tanh_out_fused,
        scale3_out_fused,
        Mask, mask_out_fused,
        softmax_out_fused,
        V, Out_fused, K
    ]

    # Fused launcher
    def fused_launcher():
        compiled[grid](*args)
    
    # Baseline launcher (only torch reference)
    def torch_baseline_launcher():
        Out_baseline.copy_(torch_attention_ref(Q, Kmat, V, Mask, scale, logit_softcapping))
    
    # Benchmark and validate
    print("\nBenchmarking...")
    benchmark_performance(torch_baseline_launcher, fused_launcher)
    
    print("\nValidating correctness...")
    validate_correctness(
        torch_baseline_launcher, 
        fused_launcher, 
        Out_baseline, 
        Out_fused,
        rtol=1e-3, 
        atol=1e-2
    )

if __name__ == "__main__":
    main()
