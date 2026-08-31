import torch
import triton
import triton.language as tl

from tilefusion.utils.utils import _next_pow2, DEVICE
# GEMM: C[B, H, M, N] = A[B, H, M, K] @ B[B, H, K, N]
# Grid: 2D split along (M, batch*heads)
# Kernel loops internally over N tiles; K handled in a single tl.dot (no K-loop)

@triton.jit
def gemm_mgrid_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_ah, stride_am, stride_ak,
    stride_bb, stride_bh, stride_bn, stride_bk,
    stride_cb, stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads index

    # Calculate batch and head offset
    batch_head_offset_a = pid_bh * stride_ah  # Assuming contiguous heads within batch
    batch_head_offset_b = pid_bh * stride_bh
    batch_head_offset_c = pid_bh * stride_ch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    # m_mask = offs_m[:, None] < M
    # Tile pointers with batch/head offset
    A_tile_ptrs = A_ptr + batch_head_offset_a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    # Sweep N dimension inside the kernel
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        B_tile_ptrs = B_ptr + batch_head_offset_b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        # Mask loads to honor true M, N, K bounds even if BLOCK_K > K (pow2 rounding)
        # a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        # b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        A_tile = tl.load(A_tile_ptrs,)
        B_tile = tl.load(B_tile_ptrs,)
        # Compute C tile; use IEEE input precision for deterministic matching with torch.mm
        C_acc = tl.dot(A_tile, B_tile)
        # Store with batch/head offset
        C_tile_ptrs = C_ptr + batch_head_offset_c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        # c_mask = m_mask & (offs_n[None, :] < N)
        tl.store(C_tile_ptrs, C_acc)


def launch_gemm_mgrid(A: torch.Tensor, B: torch.Tensor, BLOCK_M: int = 128, BLOCK_N: int = 128):
    """Launch GEMM with 2D grid split along (M, batch*heads); single-shot K via tl.dot.

    Expects 4D tensors A[batch, heads, M, K] and B[batch, heads, K, N].
    Chooses BLOCK_K as next power-of-two >= max(16, K) to satisfy tl.arange constraints,
    while masking loads in K to support arbitrary runtime K. No explicit K tiling; we
    compute the full K contribution in one tl.dot per N tile.
    """
    assert A.ndim == 4 and B.ndim == 4
    batch, heads, M, K = A.shape
    batch2, heads2, K2, N = B.shape
    assert batch == batch2 and heads == heads2, "Batch/heads dimension mismatch"
    assert K == K2, "Inner dimension mismatch"

    # Output tensor (same dtype/device as A)
    C = torch.empty((batch, heads, M, N), device=A.device, dtype=A.dtype)

    # Strides (in elements)
    stride_ab = A.stride(0)
    stride_ah = A.stride(1)
    stride_am = A.stride(2)
    stride_ak = A.stride(3)
    
    stride_bb = B.stride(0)
    stride_bh = B.stride(1)
    stride_bk = B.stride(2)
    stride_bn = B.stride(3)
    
    stride_cb = C.stride(0)
    stride_ch = C.stride(1)
    stride_cm = C.stride(2)
    stride_cn = C.stride(3)

    # Launch grid: 2D along (M, batch*heads)
    grid = (
        triton.cdiv(M, BLOCK_M),
        batch * heads,
    )

    # Choose power-of-two BLOCK_K >= max(16, K)
    def _next_pow2(x: int) -> int:
        return 1 if x <= 1 else 1 << (x - 1).bit_length()

    BLOCK_K = max(128, _next_pow2(int(K)))

    gemm_mgrid_kernel[grid](
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


def _ttir_of_gemm(M: int, N: int, K: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    A = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    B = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    C = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    BK = _next_pow2(int(K))
    triton_kernel = gemm_mgrid_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )
    return triton_kernel.asm['ttir']