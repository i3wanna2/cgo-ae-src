import torch
import triton
import triton.language as tl

from tilefusion.utils.utils import _next_pow2, DEVICE

@triton.jit
def gemm_mgrid_loopk_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_ah, stride_am, stride_ak,
    stride_bb, stride_bh, stride_bk, stride_bn,
    stride_cb, stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr, # BLOCK_N 将是 >= N 的2的幂
    BLOCK_K: tl.constexpr, # K 维的分块大小
):
    # Grid 在 M 维和 batch*heads 维上划分
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads
    
    # 计算当前 batch 和 head 的偏移
    batch_head_offset_a = pid_bh * stride_ah
    batch_head_offset_b = pid_bh * stride_bh
    batch_head_offset_c = pid_bh * stride_ch

    # -- N 维不再循环，一次性计算 --
    # M 和 N 的偏移量在 K 循环外确定
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # -- 初始化累加器 --
    # 累加器现在覆盖 (BLOCK_M, BLOCK_N) 的整个区域
    C_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # -- K 维的内部循环 --
    for k0 in range(0, K, BLOCK_K):
        # K 维偏移量
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # -- 计算 A 和 B 的 tile 指针（包含 batch 和 head 偏移）--
        A_tile_ptrs = A_ptr + batch_head_offset_a + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B_tile_ptrs = B_ptr + batch_head_offset_b + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # -- 加载 A 和 B 的分块 (tile) --
        # 使用 mask 来处理边缘情况
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        A_tile = tl.load(A_tile_ptrs)
        B_tile = tl.load(B_tile_ptrs)

        # -- 计算并累加 --
        C_acc += tl.dot(A_tile, B_tile)
    
    # -- K 维循环结束后，将最终结果写回（包含 batch 和 head 偏移）--
    # 此时 C_acc 包含了 C 的一个 (BLOCK_M, N) tile 的完整结果
    C_out = C_acc.to(A_ptr.dtype.element_ty)
    
    C_tile_ptrs = C_ptr + batch_head_offset_c + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, C_out)


def _ttir_of_gemm_loopk(M: int, N: int, K: int, BM: int, BK: int, batch: int = 1, heads: int = 1):
    A = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    B = torch.randn(batch, heads, K, N, device=DEVICE, dtype=torch.float16)
    C = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (triton.cdiv(M, BM), batch * heads)
    BN = _next_pow2(int(N))
    triton_kernel = gemm_mgrid_loopk_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )

    return triton_kernel.asm['ttir']

def launch_gemm_mgrid_loopk(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_m: int = 64,
    block_k: int = 32,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """GEMM C = A @ B with loop over K inside kernel.

    Expects:
      A: [B, H, M, K]
      B: [B, H, K, N]
      C: [B, H, M, N]

    Notes:
      The underlying kernel in ops does not mask loads, so keep sizes aligned for now.
    """
    assert a.is_cuda and b.is_cuda
    assert a.dim() == 4 and b.dim() == 4
    batch, heads, m, k = a.shape
    batch2, heads2, k2, n = b.shape
    assert batch == batch2 and heads == heads2 and k == k2

    # Keep it safe for current use-cases (seqlen=4096, head_dim=128) without OOB.
    assert k % block_k == 0, f"K={k} must be divisible by block_k={block_k}"
    assert m % block_m == 0, f"M={m} must be divisible by block_m={block_m}"

    block_n = triton.next_power_of_2(n)
    assert block_n == n, f"N={n} must be a power-of-two for current kernel (got block_n={block_n})"

    if out is None:
        c = torch.empty((batch, heads, m, n), device=a.device, dtype=a.dtype)
    else:
        c = out
        assert c.is_cuda and c.device == a.device
        assert c.dtype == a.dtype
        assert c.shape == (batch, heads, m, n)

    grid = (triton.cdiv(m, block_m), batch * heads)
    gemm_mgrid_loopk_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a.stride(3),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b.stride(3),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        c.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return c

@triton.jit
def gemm_ngrid_loopk_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_ab, stride_ah, stride_am, stride_ak,
    stride_bb, stride_bh, stride_bk, stride_bn,
    stride_cb, stride_ch, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr, 
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)       
    pid_bh = tl.program_id(axis=1)     

    batch_head_offset_a = pid_bh * stride_ah
    batch_head_offset_b = pid_bh * stride_bh
    batch_head_offset_c = pid_bh * stride_ch

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        A_tile_ptrs = A_ptr + batch_head_offset_a + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B_tile_ptrs = B_ptr + batch_head_offset_b + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        A_tile = tl.load(A_tile_ptrs, mask=a_mask, other=0.0)
        B_tile = tl.load(B_tile_ptrs, mask=b_mask, other=0.0)

        C_acc += tl.dot(A_tile, B_tile)

    C_out = C_acc.to(A_ptr.dtype.element_ty)
    C_tile_ptrs = C_ptr + batch_head_offset_c + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    tl.store(C_tile_ptrs, C_out, mask=c_mask)


def _ttir_of_gemm_grid_n(M: int, N: int, K: int, BN: int, BK: int, batch: int = 1, heads: int = 1):
    A = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    B = torch.randn(batch, heads, K, N, device=DEVICE, dtype=torch.float16)
    C = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    
    grid = (triton.cdiv(N, BN), batch * heads)
    BM = _next_pow2(M) 
    
    triton_kernel = gemm_ngrid_loopk_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
    )

    return triton_kernel.asm['ttir']