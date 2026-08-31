import triton
import triton.language as tl
import torch
from tilefusion.utils.utils import DEVICE

# ==================== H2O Sum Reduction Kernel ====================
@triton.jit
def sum_h2o_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum kernel for H2O scoring: sum over M dimension.
    
    Input:  [batch, heads, M, N] (attention probabilities)
    Output: [batch, heads, N]    (per-token importance scores)
    
    Computes: output[b, h, n] = sum over m of input[b, h, m, n]
    """
    pid_n = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads index
    
    # Calculate batch and head offset
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n[None, :] < N
    
    # Accumulate sum over all M blocks
    sum_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask = (offs_m[:, None] < M) & n_mask
        
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        x = tl.load(x_ptrs).to(tl.float32)
        
        # Sum over M dimension (axis=0)
        sum_acc += tl.sum(x, axis=0)
    
    # Store output
    y_ptrs = y_ptr + batch_head_offset_y + offs_n * stride_yn
    tl.store(y_ptrs, sum_acc.to(tl.float16), mask=offs_n < N)


def sum_h2o_triton(x: torch.Tensor, block_m: int = 64, block_n: int = 128) -> torch.Tensor:
    """Sum over M dimension for H2O scoring.
    
    Args:
        x: [batch, heads, M, N] CUDA tensor (float16)
        block_m: Block size for M dimension
        block_n: Block size for N dimension
    
    Returns:
        y: [batch, heads, N] CUDA tensor (float16) - sum over M
    """
    assert x.is_cuda, "Input must be CUDA tensor"
    assert x.dim() == 4, "Input must be 4D [batch, heads, M, N]"
    
    batch, heads, M, N = x.shape
    y = torch.empty((batch, heads, N), device=x.device, dtype=x.dtype)
    
    grid = (
        triton.cdiv(N, block_n),
        batch * heads,
    )
    
    sum_h2o_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return y


def _ttir_of_sum(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    """Generate TTIR for H2O sum kernel.
    
    Input:  [batch, heads, M, N]
    Output: [batch, heads, N]
    """
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, N, device=DEVICE, dtype=torch.float16)
    
    grid = (
        triton.cdiv(N, BN),
        batch * heads,
    )
    
    triton_kernel = sum_h2o_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    
    return triton_kernel.asm['ttir']


# ==================== Sum Reduction Kernel (dim=1) ====================
@triton.jit
def sum_dim1_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_ym, stride_yn,
    num_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum kernel for dimension 1 (heads): sum over heads dimension.
    
    Input:  [batch, heads, M, N]
    Output: [batch, M, N]
    
    Computes: output[b, m, n] = sum over h of input[b, h, m, n]
    """
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)  # batch index
    
    # Calculate batch offset
    batch_offset_x = pid_b * stride_xb
    batch_offset_y = pid_b * stride_yb
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m[:, None] < M
    
    for start_n in range(0, N, BLOCK_N):
        n = start_n + offs_n
        mask = m_mask & (n[None, :] < N)
        
        # Accumulate sum over all heads
        sum_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        
        for h in range(num_heads):
            head_offset = h * stride_xh
            x_ptrs = x_ptr + batch_offset_x + head_offset + offs_m[:, None] * stride_xm + n[None, :] * stride_xn
            x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
            sum_acc += x
        
        # Store output
        y_ptrs = y_ptr + batch_offset_y + offs_m[:, None] * stride_ym + n[None, :] * stride_yn
        tl.store(y_ptrs, sum_acc.to(tl.float16), mask=mask)


def sum_dim1_triton(x: torch.Tensor, block_m: int = 64, block_n: int = 128) -> torch.Tensor:
    """Sum over heads dimension (dim=1).
    
    Args:
        x: [batch, heads, M, N] CUDA tensor (float16)
        block_m: Block size for M dimension
        block_n: Block size for N dimension
    
    Returns:
        y: [batch, M, N] CUDA tensor (float16) - sum over heads
    """
    assert x.is_cuda, "Input must be CUDA tensor"
    assert x.dim() == 4, "Input must be 4D [batch, heads, M, N]"
    
    batch, heads, M, N = x.shape
    y = torch.empty((batch, M, N), device=x.device, dtype=x.dtype)
    
    grid = (
        triton.cdiv(M, block_m),
        batch,
    )
    
    sum_dim1_kernel[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2),
        num_heads=heads,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return y


# ==================== Sum Reduction Kernel (dim=1) — single-loop version ====================
@triton.jit
def sum_dim1_kernel_1loop(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_ym, stride_yn,
    num_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRID_N: tl.constexpr,
):
    """按 heads 维做归约，仅保留一层循环（遍历 heads），M/N 由并行网格覆盖。

    输入:  x_ptr 指向 [B, H, M, N]
    输出:  y_ptr 指向 [B, M, N]
    网格:  axis0 切 M，axis1 同时切 batch 与 N block（batch*GRID_N）
    """
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)

    b = pid_bn // GRID_N
    n_blk = pid_bn % GRID_N

    # 批次偏移
    batch_offset_x = b * stride_xb
    batch_offset_y = b * stride_yb

    # 计算当前 tile 的 M/N 下标
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_mn = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # FP32 累加
    sum_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # 唯一循环：沿 heads 维做归约
    for h in range(num_heads):
        head_offset = h * stride_xh
        x_ptrs = (
            x_ptr
            + batch_offset_x
            + head_offset
            + offs_m[:, None] * stride_xm
            + offs_n[None, :] * stride_xn
        )
        x = tl.load(x_ptrs, mask=mask_mn, other=0.0).to(tl.float32)
        sum_acc += x

    # 写回
    y_ptrs = y_ptr + batch_offset_y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, sum_acc.to(tl.float16), mask=mask_mn)


def sum_dim1_triton_1loop(x: torch.Tensor, block_m: int = 64, block_n: int = 128) -> torch.Tensor:
    """对 dim=1 (heads) 做归约的“一层循环”版本：
    - 线程网格并行覆盖 M/N（无显式 N 循环）
    - 内部仅对 heads 做一层循环并在 fp32 上累加

    Args:
        x: [B, H, M, N] (CUDA, fp16)
        block_m, block_n: tile 大小
    Returns:
        y: [B, M, N] (fp16)
    """
    assert x.is_cuda and x.dim() == 4, "Input must be CUDA 4D tensor [B,H,M,N]"
    assert x.dtype in (torch.float16, torch.bfloat16), "Expect fp16/bf16 input"

    batch, heads, M, N = x.shape
    y = torch.empty((batch, M, N), device=x.device, dtype=x.dtype)

    grid_n = triton.cdiv(N, block_n)
    grid = (
        triton.cdiv(M, block_m),
        batch * grid_n,
    )

    sum_dim1_kernel_1loop[grid](
        x, y,
        M, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2),
        num_heads=heads,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GRID_N=grid_n,
    )
    return y


def sum_triton(
    x: torch.Tensor,
    dim: int,
    block_m: int = 64,
    block_n: int = 128,
    keepdim: bool = False,
) -> torch.Tensor:
    """Generic sum reduction on a 4D tensor using Triton kernels.

    Supports:
      - dim=1: reduce over heads dimension, output shape [B, M, N]
      - dim=2: reduce over M dimension (H2O-style), output shape [B, H, N]

    Args:
        x: [batch, heads, M, N] CUDA tensor (float16)
        dim: dimension to reduce (1 for heads, 2 for M)
        block_m: block size along M (used when relevant)
        block_n: block size along N

    Returns:
        Tensor reduced along the specified dimension.
    """
    assert x.is_cuda, "Input must be CUDA tensor"
    assert x.dim() == 4, "Input must be 4D [batch, heads, M, N]"
    if dim == 1:
        out = sum_dim1_triton(x, block_m=block_m, block_n=block_n)
        if keepdim:
            # x: [B, H, M, N] -> reduce dim=1 -> [B, M, N]
            # keepdim=True -> [B, 1, M, N]
            B, M, N = out.shape
            out = out.view(B, 1, M, N)
        return out
    elif dim == 2:
        out = sum_h2o_triton(x, block_m=block_m, block_n=block_n)
        if keepdim:
            # x: [B, H, M, N] -> reduce dim=2 -> [B, H, N]
            # keepdim=True -> [B, H, 1, N]
            B, H, N = out.shape
            out = out.view(B, H, 1, N)
        return out
    else:
        raise NotImplementedError(f"sum_triton currently supports dim in { [1, 2] }, got {dim}")


def _ttir_of_sum_dim1(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 4):
    """Generate TTIR for sum dim=1 kernel.
    
    Input:  [batch, heads, M, N]
    Output: [batch, M, N]
    """
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, M, N, device=DEVICE, dtype=torch.float16)
    
    grid = (
        triton.cdiv(M, BM),
        batch,
    )
    
    triton_kernel = sum_dim1_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2),
        num_heads=heads,
        BLOCK_M=BM, BLOCK_N=BN,
    )
    
    return triton_kernel.asm['ttir']

@triton.jit
def bmh_broadcast_kernel(src_ptr, dst_ptr,
                         H, M, N,
                         stride_b_src, stride_m_src, stride_n_src,
                         stride_b_dst, stride_h_dst, stride_m_dst, stride_n_dst,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Loop over N dimension in chunks of BLOCK_N
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]

        src_ptrs = (src_ptr
                    + pid_b * stride_b_src
                    + offs_m[:, None] * stride_m_src
                    + offs_n[None, :] * stride_n_src)
        vals = tl.load(src_ptrs, mask=mask, other=0.0)

        for h in range(H):
            dst_ptrs = (dst_ptr
                        + pid_b * stride_b_dst
                        + h * stride_h_dst
                        + offs_m[:, None] * stride_m_dst
                        + offs_n[None, :] * stride_n_dst)
            tl.store(dst_ptrs, vals, mask=mask)

def broadcast_bmn_to_bhmn(x, heads, block_m=64, block_n=64):
    B, M, N = x.shape
    out = torch.empty(B, heads, M, N, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, block_m), B)
    bmh_broadcast_kernel[grid](
        x, out,
        heads, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=block_m, BLOCK_N=block_n
    )
    return out

def _ttir_of_broadcast(M, N, BM, BN, batch, heads):
    """Generate TTIR for broadcast [B,M,N] -> [B,H,M,N]"""
    import torch
    import tempfile
    
    # Create dummy inputs with correct shapes
    x = torch.randn(batch, M, N, device='cuda', dtype=torch.float16)
    out = torch.empty(batch, heads, M, N, device='cuda', dtype=torch.float16)
    
    # Get the grid
    grid = (triton.cdiv(M, BM), batch)
    
    # Trigger compilation
    triton_kernel = bmh_broadcast_kernel[grid](
        x, out,
        heads, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_M=BM, BLOCK_N=BN
    )
    
    return triton_kernel.asm['ttir']
# ==================== H2O Sum and Square Reduction Kernel ====================
@triton.jit
def sum_and_square_kernel(
    x_ptr, y_sum_ptr, y_sq_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_ysumb, stride_ysumh, stride_ysumn,
    stride_ysqb, stride_ysqh, stride_ysqn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sum and Square Sum kernel for ROCO scoring: sum over M dimension.
    
    Input:  [batch, heads, M, N] (attention probabilities)
    Output 1: [batch, heads, N]    (sum of probs)
    Output 2: [batch, heads, N]    (sum of probs^2)
    """
    pid_n = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads index
    
    # Calculate batch and head offset
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_ysum = pid_bh * stride_ysumh
    batch_head_offset_ysq = pid_bh * stride_ysqh
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n[None, :] < N
    
    # Accumulate sum over all M blocks
    sum_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    sq_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask = (offs_m[:, None] < M) & n_mask
        
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        x = tl.load(x_ptrs).to(tl.float32)
        
        # Sum over M dimension (axis=0)
        sum_acc += tl.sum(x, axis=0)
        sq_acc += tl.sum(x * x, axis=0)
    
    # Store output
    ysum_ptrs = y_sum_ptr + batch_head_offset_ysum + offs_n * stride_ysumn
    ysq_ptrs = y_sq_ptr + batch_head_offset_ysq + offs_n * stride_ysqn
    
    tl.store(ysum_ptrs, sum_acc.to(tl.float16), )
    tl.store(ysq_ptrs, sq_acc.to(tl.float16), )

def _ttir_of_sum_and_square(M, N, BM, BN, batch, heads):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y_sum = torch.empty(batch, heads, N, device=DEVICE, dtype=torch.float16)
    Y_sq = torch.empty(batch, heads, N, device=DEVICE, dtype=torch.float16)
    
    grid = (triton.cdiv(N, BN), batch * heads)
    
    triton_kernel = sum_and_square_kernel[grid](
        X, Y_sum, Y_sq,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y_sum.stride(0), Y_sum.stride(1), Y_sum.stride(2),
        Y_sq.stride(0), Y_sq.stride(1), Y_sq.stride(2),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']
