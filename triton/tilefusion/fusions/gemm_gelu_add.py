import math
import time
import tempfile
import torch
import torch.nn.functional as F
import triton
from triton.compiler import compile as triton_compile

# Use local module imports (same style as other demos)
from tilefusion.ops.gemm_mgrid_kernel import _ttir_of_gemm,launch_gemm_mgrid
from tilefusion.ops.gemm_mgrid_loopk_kernel import _ttir_of_gemm_loopk
from tilefusion.ops.softmax_blockM import _ttir_of_softmax
from tilefusion.utils.utils import DEVICE, build_combined_module, benchmark_performance, validate_correctness
from tilefusion.core.compiler import fuse_kernels_in_ttir,opt_kernels_in_ttir
from tilefusion.ops.elementwise import _ttir_of_addmask, _ttir_of_scale, add_mask_triton
from tilefusion.ops.relu import _ttir_of_relu,relu_triton, _ttir_of_gelu, gelu_triton
# from tilefusion.core.autotuner import autotune_kernel
# from flash_attn.flash_attn_interface import flash_attn_func
from tilefusion.ops.gems_attn import scaled_dot_product_attention


def sdpa_torch_attention_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor, scale: float) -> torch.Tensor:
    """Reference with internal layout expected by SDPA:
    Expects Q: [B,H,M,K], K: [B,H,N,K], V: [B,H,N,K], Mask: [B,H,M,N].
    Returns O: [B,H,M,K].
    """
    return F.scaled_dot_product_attention(Q, Kmat, V, attn_mask=Mask, scale=scale)

# def flash_attn_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor, scale: float) -> torch.Tensor:
#     # flash_attn_func expects [B, seqlen, nheads, headdim] and doesn't use external mask/scale
#     # Input: Q[B,H,M,K], Kmat[B,H,N,K], V[B,H,N,K]
#     # Convert to [B, seqlen, nheads, headdim]
#     out = flash_attn_func(
#         Q.transpose(1, 2),      # [B,H,M,K] -> [B,M,H,K]
#         Kmat.transpose(1, 2),   # [B,H,N,K] -> [B,N,H,K]
#         V.transpose(1, 2),      # [B,H,N,K] -> [B,N,H,K]
#         causal=True,           # Don't use causal mask since torch_ref doesn't
#         softmax_scale=scale     # Use our scale factor 
#     ).transpose(1, 2).contiguous()  # [B,M,H,K] -> [B,H,M,K]
#     return out
    

def torch_attention_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor, scale: float) -> torch.Tensor:
    """Pure-Torch reference for attention:
    scores = softmax((Q @ K^T) * scale + Mask, dim=-1) @ V
    - Computes in fp32 for stability and casts back to Q.dtype (fp16) at the end.
    - All tensors are expected to be on the same CUDA device.
    Shapes:
        Q: [batch, heads, M, K], Kmat: [batch, heads, K, N], V: [batch, heads, N, K], Mask: [batch, heads, M, N]
    Returns: O: [batch, heads, M, K] with dtype == Q.dtype
    """
    assert Q.is_cuda and Kmat.is_cuda and V.is_cuda and Mask.is_cuda, "torch ref expects CUDA tensors"
    q = Q
    v = V
    mask = Mask
    scores = q @ Kmat.transpose(-2, -1)  # [batch, heads, M, K] @ [batch, heads, K, N] -> [batch, heads, M, N]
    # scores = torch.matmul(q,Kmat.transpose(-2, -1))
    scores = gelu_triton(scores)
    scores = scores + mask
    
    # c = q + Kmat
    # d = torch.relu(c)
    # c = add_mask_triton(q,Kmat)
    # d = relu_triton(c)
    # probs = torch.softmax(scores, dim=-1)  # softmax over N dimension
    # O = probs @ v  # [batch, heads, M, N] @ [batch, heads, N, K] -> [batch, heads, M, K]

    return scores


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


def build_and_compile_fused_kernel(M, N, K, BM, BN, scale, batch, heads, num_warps=4, num_stages=2):
    """
    Build and compile a fused attention kernel with given block sizes.
    
    Args:
        M, N, K: Problem dimensions (per head)
        BM, BN, BK: Block sizes
        scale: Scale factor for attention
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

    # current_ttir = _ttir_of_addmask(M, N, BM, BN, batch, heads)
    # current_producer_name = "add_mask_kernel"

    # Define fusion stages with 4D support
    # Verified parameter mappings:
    # For now, only fuse up to softmax (GEMM_loopk has dependency issues)
    stages = [
        # Stage 1: GEMM + Scale
        ("gelu_kernel",             lambda: _ttir_of_gelu(M, N, BM, BN, batch, heads),  
         [2,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # Stage 2: (GEMM+Scale) + AddMask  
        ("add_mask_kernel",          lambda: _ttir_of_addmask(M, N, BM, BN, batch, heads),       
         [15,3,4,12,13,14,12,13,14,12,13,14], [0,3,4,5,6,7,8,9,10,11,12,13]),
        # Stage 3: (GEMM+Scale+AddMask) + Softmax
        # ("softmax_kernel",           lambda: _ttir_of_softmax(M, N, BM, BN, batch, heads),       
        #  [18,3,4,12,13,14,12,13,14], [0,2,3,4,5,6,7,8,9]),
        # # Note: gemm_loopk fusion disabled due to dependency issues - using torch.matmul instead
        # ("gemm_mgrid_loopk_kernel", lambda: _ttir_of_gemm_loopk(M, K, N, BM, BN, batch, heads),  
        #  [19,3,4,12,13,14,9,10,11,6,7,8], [0,3,5,6,7,8,9,10,11,12,13,14]),
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
    return compiled, grid


def run_attention_with_config(config, Q, Kmat, V, Mask, scale, M, N, K, batch, heads, warmup_iters=10, bench_iters=100):
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
    compiled, grid = build_and_compile_fused_kernel(M, N, K, BM, BN, scale, batch, heads, num_warps, num_stages)
    
    # Allocate output buffers (4D: [batch, heads, M, N/K])
    scores_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    mask_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    softmax_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Out_fused = torch.empty(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    
    # Prepare kernel arguments (4D strides)
    args = [
        Q, Kmat, scores_fused,
        M, N, 
        # K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale_out_fused,
        # scale,
        # Mask, mask_out_fused,
        # Mask.stride(0), Mask.stride(1), Mask.stride(2),
        # mask_out_fused.stride(0), mask_out_fused.stride(1), mask_out_fused.stride(2),
        # softmax_out_fused,
        # softmax_out_fused.stride(0), softmax_out_fused.stride(1), softmax_out_fused.stride(2),
        # V, Out_fused,
        # V.stride(0), V.stride(1), V.stride(2),
        # Out_fused.stride(0), Out_fused.stride(1), Out_fused.stride(2),
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
  
    # Allocate output buffers (4D)
    scores_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    scale_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    mask_out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    softmax_out_fused = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Out_fused = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Out_baseline = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)

    DEBUG = True   
    USE_PATH = False
    USE_NCU = False
    if not DEBUG:
        best_config, best_output, best_time, all_valid = autotune_kernel(
            get_autotune_configs(),
            run_attention_with_config,
            (Q, Kmat, V, Mask, scale, M, N, K, batch, heads)
        )
    # Build and compile kernel with best config
        BM = best_config['BM']
        BN = best_config['BN']
        num_warps = best_config.get('num_warps', 4)
        num_stages = best_config.get('num_stages', 2)
    else :
        BM = 64
        BN = 64
        num_warps = 4
        num_stages = 3


    if USE_PATH:
        # /tmp/tmpuqzib7_f.ttir  tt.divis
        # /tmp/tmpfwhmmdij.ttir
        fused_path = "/tmp/tmpapowhta8.ttir"
        options = {
            'num_warps': num_warps,
            'num_stages': num_stages,
        }
        compiled = triton_compile(fused_path, options=options)
        grid = (triton.cdiv(M, BM), batch * heads, 1)
    else :
        compiled, grid = build_and_compile_fused_kernel(M, N, K, BM, BN, scale, batch, heads,num_warps,num_stages)


    args = [
        Q, Kmat, scores_fused,
        M, N,
        K,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale_out_fused,
        # scale,
        Mask, Out_fused,
        # Mask.stride(0), Mask.stride(1), Mask.stride(2),
        # mask_out_fused.stride(0), mask_out_fused.stride(1), mask_out_fused.stride(2),
        # softmax_out_fused,
        # softmax_out_fused.stride(0), softmax_out_fused.stride(1), softmax_out_fused.stride(2),
        # V, Out_fused, K
        # V.stride(0), V.stride(1), V.stride(2),
        # Out_fused.stride(0), Out_fused.stride(1), Out_fused.stride(2),
    ]

    # Fused launcher
    def fused_launcher():
        compiled[grid](*args)
        # res = triton.testing.do_bench(lambda: compiled[grid](*args))
        # print(f"Fused kernel time: {res} ms")
    # Baseline launchers
    # def flash_attn_launcher():
    #     Out_baseline.copy_(flash_attn_ref(Q, Kmat, V, Mask, scale))
    
    def torch_baseline_launcher():
        Out_baseline.copy_(torch_attention_ref(Q, Kmat, V, Mask, scale))
   
    # def gems_launcher():
    #     Out_baseline.copy_(scaled_dot_product_attention(Q, Kmat, V, is_causal=True, scale=scale))
    
    # fused_launcher()
    # Benchmark and validate
    
    if not USE_NCU:
        benchmark_performance(torch_baseline_launcher, fused_launcher)
        validate_correctness(
            torch_baseline_launcher, 
            fused_launcher, 
            Out_baseline, 
            Out_fused,
            rtol=1e-3, 
            atol=1e-2
        )
    else :
        flash_attn_launcher()
    # print(scale_out_fused)
    
if __name__ == "__main__":
    main()
