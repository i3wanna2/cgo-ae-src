import triton
import triton.language as tl
import torch

from tilefusion.utils.utils import DEVICE

@triton.jit
def softmax_kernel_stable(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads index

    # Calculate batch and head offset
    # For a flattened [batch, heads] indexing, we need to compute the actual 2D offset
    batch_head_offset_x = pid_bh * stride_xh  # Assuming contiguous heads within batch
    batch_head_offset_y = pid_bh * stride_yh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m[:, None] < M

    # Online softmax stats across N tiles (use fp32 accumulators)
    m_i = tl.full((BLOCK_M, 1), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    # First pass: compute row-wise max and denom
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = m_mask & (offs_n[None, :] < N)
        x_ptrs = x_ptr + batch_head_offset_x + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        x = tl.load(x_ptrs,)
        x = x.to(tl.float32)
        m_curr = tl.max(x, axis=1, keep_dims=True)
        m_new = tl.maximum(m_i, m_curr)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=1, keep_dims=True)
        m_i = m_new

    # Second pass: write normalized probabilities
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = m_mask & (offs_n[None, :] < N)
        x_ptrs = x_ptr + batch_head_offset_x + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        y_ptrs = y_ptr + batch_head_offset_y + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
        x_in = tl.load(x_ptrs, )
        x = x_in.to(tl.float32)
        y = tl.exp(x - m_i) / l_i
        y = tl.where(mask, y, 0.0)
        # Convert result back to the input dtype (e.g., fp16) for storage
        tl.store(y_ptrs, y.to(x_in.dtype))


@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)

    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m[:, None] < M

    # Precompute 1 / ln(2) ≈ 1.44269504089
    # Because: exp(x) = exp2(x / ln(2)) = exp2(x * (1/ln(2)))
    INV_LN2: tl.constexpr = 1.4426950408889634  # 1 / log(2)

    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    # --- Pass 1: Compute denominator ---
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = m_mask & (offs_n[None, :] < N)
        x_ptrs = x_ptr + batch_head_offset_x + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        x = tl.load(x_ptrs).to(tl.float32)
        
        # Convert exp(x) → exp2(x * INV_LN2)
        x_scaled = x * INV_LN2
        numerators = tl.math.exp2(x_scaled)
        # l_i += tl.sum(tl.where(mask, numerators, 0.0), axis=1, keep_dims=True)
        l_i += tl.sum(numerators, axis=1, keep_dims=True)

    # --- Pass 2: Normalize and store ---
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = m_mask & (offs_n[None, :] < N)
        x_ptrs = x_ptr + batch_head_offset_x + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        y_ptrs = y_ptr + batch_head_offset_y + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
        
        x = tl.load(x_ptrs).to(tl.float32)
        x_scaled = x * INV_LN2
        y = tl.math.exp2(x_scaled) / l_i
        # y = tl.where(mask, y, 0.0)
        tl.store(y_ptrs, y.to(tl.float16))

def softmax_triton(x: torch.Tensor, block_m: int = 64, block_n: int = 128) -> torch.Tensor:
    """Row-wise softmax over last dim for 4D tensor using Triton.

    Args:
        x: [batch, heads, M, N] CUDA tensor (float16/float32)
    Returns:
        y: [batch, heads, M, N] CUDA tensor (same dtype as x; kernel computes in fp32 and converts back)
    """
    assert x.is_cuda, "Input must be CUDA tensor"
    assert x.dim() == 4, "Input must be 4D [batch, heads, M, N]"
    batch, heads, M, N = x.shape
    y = torch.empty((batch, heads, M, N), device=x.device, dtype=x.dtype)

    grid = (
        triton.cdiv(M, block_m),
        batch * heads,
    )
    softmax_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return y


def _ttir_of_softmax(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    triton_kernel = softmax_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']
