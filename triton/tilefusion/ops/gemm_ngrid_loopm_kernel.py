import torch
import triton
import triton.language as tl

from tilefusion.utils.utils import _next_pow2, DEVICE


@triton.jit
def gemm_ngrid_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_ah, stride_am, stride_ak,
    stride_bb, stride_bh, stride_bn, stride_bk, # 注意：这里是 K, N
    stride_cb, stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    GEMM: C[B, H, M, N] = A[B, H, M, K] @ B[B, H, K, N]
    Grid: 2D split along (N, batch*heads)
    Kernel loops internally over M tiles
    K handled in a single tl.dot
    """
    pid_n = tl.program_id(axis=0)  # Grid is over N
    pid_bh = tl.program_id(axis=1) # Grid is over Batch*Heads

    # Batch/Head offsets
    batch_head_offset_a = pid_bh * stride_ah
    batch_head_offset_b = pid_bh * stride_bh
    batch_head_offset_c = pid_bh * stride_ch

    # Pointers for B tile (KxN) - this is constant for this program
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    B_tile_ptrs = (B_ptr + batch_head_offset_b + 
                   offs_k[:, None] * stride_bk +  # [K, 1]
                   offs_n[None, :] * stride_bn)  # [1, N]

    # Load B tile once
    # b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    # B_tile = tl.load(B_tile_ptrs, mask=b_mask, other=0.0)

    # Sweep M dimension inside the kernel
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        
        # Pointers for A tile (MxK)
        A_tile_ptrs = (A_ptr + batch_head_offset_a + 
                       offs_m[:, None] * stride_am +  # [M, 1]
                       offs_k[None, :] * stride_ak)  # [1, K]
                       

        B_tile = tl.load(B_tile_ptrs)
        A_tile = tl.load(A_tile_ptrs)
        
        # Compute C tile
        C_acc = tl.dot(A_tile, B_tile)
        
        # Store C tile
        C_tile_ptrs = (C_ptr + batch_head_offset_c + 
                       offs_m[:, None] * stride_cm +  # [M, 1]
                       offs_n[None, :] * stride_cn)  # [1, N]

        tl.store(C_tile_ptrs, C_acc)


def launch_gemm_ngrid(A: torch.Tensor, B: torch.Tensor, BLOCK_M: int = 128, BLOCK_N: int = 128):
    """
    Launcher for gemm_ngrid_kernel (Grid over N)
    A[B, H, M, K], B[B, H, K, N]
    """
    assert A.ndim == 4 and B.ndim == 4
    batch, heads, M, K = A.shape
    batch2, heads2, K2, N = B.shape
    assert batch == batch2 and heads == heads2
    assert K == K2

    C = torch.empty((batch, heads, M, N), device=A.device, dtype=A.dtype)

    stride_ab, stride_ah, stride_am, stride_ak = A.stride()
    stride_bb, stride_bh, stride_bk, stride_bn = B.stride()
    stride_cb, stride_ch, stride_cm, stride_cn = C.stride()

    # --- 关键区别：Grid 划分 ---
    grid = (
        triton.cdiv(N, BLOCK_N), # Grid 划分 N
        batch * heads,
    )
    # --- ------------------- ---

    def _next_pow2(x: int) -> int:
        return 1 if x <= 1 else 1 << (x - 1).bit_length()

    BLOCK_K = max(16, _next_pow2(int(K))) # 修复：16是tl.dot的最小要求

    gemm_ngrid_kernel[grid](
        A, B, C,
        M, N, K,
        stride_ab, stride_ah, stride_am, stride_ak,
        stride_bb, stride_bh, stride_bn, stride_bk,
        stride_cb, stride_ch, stride_cm, stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return C


def _ttir_of_gemm_ngrid(M: int, N: int, K: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    A = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    B = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    C = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(N, BN), # 新的 Grid
        batch * heads,
    )
    BK = _next_pow2(int(K))
    triton_kernel = gemm_ngrid_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return triton_kernel.asm['ttir']

# ========================================================================
# 测试
# ========================================================================
if __name__ == "__main__":

    print("Running tests on CUDA device.")
    # 定义测试形状
    B, H, M, K, N = 1, 32, 4096, 128, 4096
    
    # 块大小
    BM, BN = 64, 64

    print(f"Testing GEMM B={B}, H={H}, M={M}, K={K}, N={N}")
    
    # 创建输入张量
    A = torch.randn(B, H, M, K, device=DEVICE, dtype=torch.float16)
    B_tensor = torch.randn(B, H, K, N, device=DEVICE, dtype=torch.float16)
    
    # 1. PyTorch (Eager) 作为参考基准
    C_torch = torch.matmul(A, B_tensor)
    

    # 3. 测试新版 (Grid over N)
    C_triton_ngrid = launch_gemm_ngrid(A, B_tensor, BLOCK_M=BM, BLOCK_N=BN)

    torch.testing.assert_close(C_triton_ngrid, C_torch, rtol=1e-2, atol=1e-3, msg="N-Grid Kernel mismatch")
    print("✅ New Kernel (gemm_ngrid_kernel) PASSED")

    print("\nAll tests passed! Both kernels produce identical results.")

    # # 可选：打印 TTIR
    # print("\n--- TTIR for M-Grid Kernel ---")
    # print(_ttir_of_gemm_mgrid(M, N, K, BM, BN, B, H))
    # print("\n--- TTIR for N-Grid Kernel ---")
    # print(_ttir_of_gemm_ngrid(M, N, K, BM, BN, B, H))
