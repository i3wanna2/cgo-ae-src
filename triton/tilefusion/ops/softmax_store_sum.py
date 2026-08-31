import torch
import triton
import triton.language as tl
import math # 需要 math.log(2)

# 检查是否有可用的 CUDA 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("CUDA not available, running on CPU (Triton kernels will not execute)")


@triton.jit
def softmax_fwd_kernel(
    # --- Pointers ---
    x_ptr, y_ptr, lse_ptr,
    # --- Dimensions ---
    M, N,
    # --- Strides ---
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    stride_lseb, stride_lseh, stride_lsem,
    # --- Constexprs ---
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Kernel 1: 计算 Softmax(X) (使用 exp2) 并保存 Sum-Exp (LSE)。
    - Y = exp2(X * C) / sum(exp2(X * C))
    - lse_ptr (L) = sum(exp2(X * C))
    """
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # Batch * Heads 索引
    INV_LN2: tl.constexpr = 1.4426950408889634  # 1 / log(2)

    # --- 计算 offsets ---
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    batch_head_offset_lse = pid_bh * stride_lseh

    # 当前 M-block 的行 offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M  # 保护 M 维度

    # --- Pass 1: 计算 l_i (sum-exp) ---
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        x_ptrs = x_ptr + batch_head_offset_x + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        x = tl.load(x_ptrs).to(tl.float32)
        
        # Convert exp(x) → exp2(x * INV_LN2)
        x_scaled = x * INV_LN2
        numerators = tl.math.exp2(x_scaled)
        # l_i += tl.sum(tl.where(mask, numerators, 0.0), axis=1, keep_dims=True)
        l_i += tl.sum(numerators, axis=1, keep_dims=True)

    
    # 存储 l_i (sum-exp)
    lse_ptrs = lse_ptr + batch_head_offset_lse + (offs_m[:, None] * stride_lsem)
    tl.store(lse_ptrs, l_i, mask=m_mask[:, None])

    # --- Pass 2: 计算 Y = exp2(X * C) / l_i ---
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        x_ptrs = x_ptr + batch_head_offset_x + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        y_ptrs = y_ptr + batch_head_offset_y + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
        
        x = tl.load(x_ptrs).to(tl.float32)
        x_scaled = x * INV_LN2
        y = tl.math.exp2(x_scaled) / l_i
        # y = tl.where(mask, y, 0.0)
        tl.store(y_ptrs, y.to(tl.float16))


def _ttir_of_softmax_store(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    # LSE 张量是 fwd kernel 所必需的
    LSE = torch.empty((batch, heads, M, 1), device=DEVICE, dtype=torch.float32) 
    
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    
    # 修正：调用 softmax_fwd_kernel 并传入 LSE 及其 strides
    triton_kernel = softmax_fwd_kernel[grid]( 
        X, Y, LSE, 
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2), 
        BLOCK_M=BM, BLOCK_N=BN,
    )
    # print(triton_kernel.asm['ttir'])
    return triton_kernel.asm['ttir']

@triton.jit
def softmax_reduce_kernel(
    x_ptr, lse_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_lseb, stride_lseh, stride_lsem,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Kernel 1: 计算 Softmax(X) (使用 exp2) 并保存 Sum-Exp (LSE)。
    - Y = exp2(X * C) / sum(exp2(X * C))
    - lse_ptr (L) = sum(exp2(X * C))
    """
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # Batch * Heads 索引
    INV_LN2: tl.constexpr = 1.4426950408889634  # 1 / log(2)

    # --- 计算 offsets ---
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_lse = pid_bh * stride_lseh

    # 当前 M-block 的行 offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M  # 保护 M 维度

    # --- Pass 1: 计算 l_i (sum-exp) ---
    l_i = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        x_ptrs = x_ptr + batch_head_offset_x + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        x = tl.load(x_ptrs).to(tl.float32)
        
        # Convert exp(x) → exp2(x * INV_LN2)
        x_scaled = x * INV_LN2
        numerators = tl.math.exp2(x_scaled)
        # l_i += tl.sum(tl.where(mask, numerators, 0.0), axis=1, keep_dims=True)
        l_i += tl.sum(numerators, axis=1, keep_dims=True)

    lse_ptrs = lse_ptr + batch_head_offset_lse + (offs_m[:, None] * stride_lsem)
    tl.store(lse_ptrs, l_i, mask=m_mask[:, None])

def _ttir_of_softmax_reduce_kernel(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    LSE = torch.empty((batch, heads, M, 1), device=DEVICE, dtype=torch.float32) 
    
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )

    triton_kernel = softmax_reduce_kernel[grid]( 
        X, LSE, 
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2), 
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']

@triton.jit
def softmax_recompute_kernel(
    x_ptr, y_ptr, lse_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    stride_lseb, stride_lseh, stride_lsem,
    # --- Constexprs ---
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1) # Batch * Heads 索引
    INV_LN2: tl.constexpr = 1.4426950408889634  # 1 / log(2)

    # --- 计算 offsets ---
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M # 保护 M 维度

    # Batch 和 Head 的基地址
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    batch_head_offset_lse = pid_bh * stride_lseh

    lse_ptrs = lse_ptr + batch_head_offset_lse + (offs_m[:, None] * stride_lsem)
    l_i = tl.load(lse_ptrs, )

    # --- 循环遍历 N 维度 ---
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n[None, :] < N
        mask = m_mask[:, None] & n_mask # 完整的 (BLOCK_M, BLOCK_N) mask

        # 加载 X [BLOCK_M, BLOCK_N]
        x_ptrs = x_ptr + batch_head_offset_x + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        x_in = tl.load(x_ptrs, )

        x = x_in.to(tl.float32)
        x_scaled = x * INV_LN2
        y = tl.math.exp2(x_scaled) / l_i

        y_ptrs = y_ptr + batch_head_offset_y + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
        tl.store(y_ptrs, y.to(x_in.dtype),)

def softmax_recompute_triton(x: torch.Tensor, lse: torch.Tensor, block_m: int = 64, block_n: int = 128) -> torch.Tensor:
    """
    Wrapper for Kernel 2: 从 X 和 LSE 重新计算 Softmax。
    (已修复：移除了 max_val)
    """
    assert x.is_cuda and x.dim() == 4, "Input must be 4D CUDA tensor"
    batch, heads, M, N = x.shape
    
    assert lse.shape == (batch, heads, M, 1)
    
    y = torch.empty_like(x)
    
    # (已修复：Grid 改为 2D 以匹配 kernel 2 的新逻辑)
    grid = (
        triton.cdiv(M, block_m),
        batch * heads
    )

    softmax_recompute_kernel[grid](
        x, y, lse,
        M, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return y

def _ttir_of_softmax_recompute(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    LSE = torch.empty((batch, heads, M, 1), device=DEVICE, dtype=torch.float32) 
    
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    
    triton_kernel = softmax_recompute_kernel[grid]( 
        X, Y, LSE, 
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2), 
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']


@triton.jit
def softmax_recompute_ngrid_kernel(
    # --- Pointers ---
    x_ptr, y_ptr, lse_ptr, 
    # --- Dimensions ---
    M, N,
    # --- Strides ---
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    stride_lseb, stride_lseh, stride_lsem,
    # --- Constexprs ---
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(axis=0) # Grid 划分 N
    pid_bh = tl.program_id(axis=1)
    INV_LN2: tl.constexpr = 1.4426950408889634

    # --- 计算 offsets ---
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n[None, :] < N # [1, BLOCK_N] mask
    
    offs_m_base = tl.arange(0, BLOCK_M)

    # Batch 和 Head 的基地址
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    batch_head_offset_lse = pid_bh * stride_lseh

    # --- 循环遍历 M 维度 ---
    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + offs_m_base
        # m_mask = offs_m < M # [BLOCK_M] mask
        
        # mask = m_mask[:, None] & n_mask # [BLOCK_M, BLOCK_N] mask
        lse_ptrs = lse_ptr + batch_head_offset_lse + (offs_m[:, None] * stride_lsem)
        # l_i = tl.load(lse_ptrs, mask=m_mask[:, None], other=float("inf"))
        l_i = tl.load(lse_ptrs,)
        x_ptrs = x_ptr + batch_head_offset_x + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
        x_in = tl.load(x_ptrs,)
        x = x_in.to(tl.float32)
        x_scaled = x * INV_LN2
        y = tl.math.exp2(x_scaled) / l_i
        y_ptrs = y_ptr + batch_head_offset_y + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
        tl.store(y_ptrs, y.to(x_in.dtype))

def launch_softmax_recompute_ngrid(x: torch.Tensor, lse: torch.Tensor, block_m: int = 64, block_n: int = 64) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 4, "Input must be 4D CUDA tensor"
    batch, heads, M, N = x.shape
    assert lse.shape == (batch, heads, M, 1)
    
    y = torch.empty_like(x)
    
    grid = (
        triton.cdiv(N, block_n), # Grid 划分 N
        batch * heads
    )

    softmax_recompute_ngrid_kernel[grid](
        x, y, lse,
        M, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return y

def _ttir_of_softmax_recompute_ngrid(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    LSE = torch.empty((batch, heads, M, 1), device=DEVICE, dtype=torch.float32) 
    
    grid = (
        triton.cdiv(N, BN), # Grid 划分 N
        batch * heads,
    )
    
    triton_kernel = softmax_recompute_ngrid_kernel[grid]( 
        X, Y, LSE,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2), 
        BLOCK_M=BM, BLOCK_N=BN,
    )
    # print(triton_kernel.asm['ttir'])
    return triton_kernel.asm['ttir']
