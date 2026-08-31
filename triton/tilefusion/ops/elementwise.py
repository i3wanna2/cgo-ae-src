import triton
import triton.language as tl
import torch
from tilefusion.utils.utils import DEVICE

@triton.jit
def scale_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    scale,  # runtime scalar factor
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads index
    
    # Calculate batch and head offset
    batch_head_offset_x = pid_bh * stride_xh  # Assuming contiguous heads within batch
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m[:, None] < M
    for start_n in range(0, N, BLOCK_N):
        n = start_n + offs_n
        mask = m_mask & (n[None, :] < N)
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + n[None, :] * stride_yn
        x = tl.load(x_ptrs)
        # Runtime scalar scale (broadcasted)
        y = x * scale
        tl.store(y_ptrs, y)


@triton.jit
def add_mask_kernel(
    x_ptr, m_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads index
    
    # Calculate batch and head offset
    batch_head_offset_x = pid_bh * stride_xh  # Assuming contiguous heads within batch
    batch_head_offset_m = pid_bh * stride_mh
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m[:, None] < M
    for start_n in range(0, N, BLOCK_N):
        n = start_n + offs_n
        mask = m_mask & (n[None, :] < N)
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + n[None, :] * stride_xn
        m_ptrs = m_ptr + batch_head_offset_m + offs_m[:, None] * stride_mm + n[None, :] * stride_mn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + n[None, :] * stride_yn
        x = tl.load(x_ptrs)
        mm = tl.load(m_ptrs)
        y = x + mm
        tl.store(y_ptrs, y)
        
def add(x: torch.Tensor, y: torch.Tensor, *, block_m: int = 64, block_n: int = 64, out: torch.Tensor | None = None) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.shape == y.shape and x.dim() == 4
    batch, heads, m, n = x.shape
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.is_cuda and out.shape == x.shape and out.dtype == x.dtype and out.device == x.device

    grid = (triton.cdiv(m, block_m), batch * heads)
    add_mask_kernel[grid](
        x,
        y,
        out,
        m,
        n,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out
        

def log_neg(x: torch.Tensor, *, block_m: int = 64, block_n: int = 64, out: torch.Tensor | None = None) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 4
    batch, heads, m, n = x.shape
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.is_cuda and out.shape == x.shape and out.dtype == x.dtype and out.device == x.device

    grid = (triton.cdiv(m, block_m), batch * heads)
    log_neg_kernel[grid](
        x,
        out,
        m,
        n,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out


def _ttir_of_scale(M: int, N: int, BM: int, BN: int, scale: float, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    triton_kernel = scale_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        scale,
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']

@triton.jit
def scale_kernel_c(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    scale  : tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads index
    
    # Calculate batch and head offset
    batch_head_offset_x = pid_bh * stride_xh  # Assuming contiguous heads within batch
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m[:, None] < M
    for start_n in range(0, N, BLOCK_N):
        n = start_n + offs_n
        mask = m_mask & (n[None, :] < N)
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + n[None, :] * stride_yn
        x = tl.load(x_ptrs)
        # Runtime scalar scale (broadcasted)
        y = x * scale
        tl.store(y_ptrs, y)

def div(x: torch.Tensor, *, scale: float, block_m: int = 64, block_n: int = 64, out: torch.Tensor | None = None) -> torch.Tensor:
    assert x.is_cuda and x.dim() == 4, "x must be [B,H,M,N] CUDA tensor"
    batch, heads, m, n = x.shape
    if out is None:
        out = torch.empty_like(x)
    else:
        assert out.is_cuda and out.shape == x.shape and out.dtype == x.dtype and out.device == x.device

    grid = (triton.cdiv(m, block_m), batch * heads)
    scale_kernel_c[grid](
        x,
        out,
        m,
        n,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        scale=scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out

def _ttir_of_scale_c(M: int, N: int, BM: int, BN: int, scale: float, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    triton_kernel = scale_kernel_c[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        scale=scale,
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']


@triton.jit
def log_neg_kernel(
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
    batch_head_offset_x = pid_bh * stride_xh  # Assuming contiguous heads within batch
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m[:, None] < M
    for start_n in range(0, N, BLOCK_N):
        n = start_n + offs_n
        mask = m_mask & (n[None, :] < N)
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + n[None, :] * stride_yn
        x = tl.load(x_ptrs)
        x_fp32 = x.to(tl.float32)
        y = -tl.log(x_fp32)
        tl.store(y_ptrs, y.to(tl.float16))

def _ttir_of_log_neg(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    triton_kernel = log_neg_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']

def _ttir_of_addmask(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Mask = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    triton_kernel = add_mask_kernel[grid](
        X, Mask, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Mask.stride(0), Mask.stride(1), Mask.stride(2), Mask.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x + y
    tl.store(output_ptr + offsets, out, mask=mask)

@triton.jit
def div_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = x / y
    tl.store(output_ptr + offsets, out, mask=mask)

def _ttir_of_add(n=1024):
    x = torch.randn(n, device=DEVICE, dtype=torch.float16)
    y = torch.randn(n, device=DEVICE, dtype=torch.float16)
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    triton_kernel = add_kernel[grid](x, y, z, n, BLOCK_SIZE=1024)
    return triton_kernel.asm['ttir']

def _ttir_of_div(n=1024):
    x = torch.randn(n, device=DEVICE, dtype=torch.float16)
    y = torch.randn(n, device=DEVICE, dtype=torch.float16)
    z = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    triton_kernel = div_kernel[grid](x, y, z, n, BLOCK_SIZE=1024)
    return triton_kernel.asm['ttir']


@triton.jit
def tanh_kernel(
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
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m[:, None] < M
    for start_n in range(0, N, BLOCK_N):
        n = start_n + offs_n
        mask = m_mask & (n[None, :] < N)
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + n[None, :] * stride_yn
        x_in = tl.load(x_ptrs)
        x = x_in.to(tl.float32)
        # Compute tanh using: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        # For numerical stability, use: tanh(x) = 2 / (1 + exp(-2x)) - 1
        # exp_neg_2x = tl.exp(-2.0 * x)
        # y = 2.0 / (1.0 + exp_neg_2x) - 1.0
        y = tl.inline_asm_elementwise(
            asm='tanh.approx.f32 $0, $1;',
            constraints=('=r,r'),
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(y_ptrs, y.to(x_in.dtype))


def _ttir_of_tanh(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    triton_kernel = tanh_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']


@triton.jit
def tanh_ngrid_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # --- 1. Grid Indexing ---
    pid_n = tl.program_id(axis=0)  # N 维度并行
    pid_bh = tl.program_id(axis=1) # Batch * Heads
    
    # --- 2. Offsets Calculation ---
    # N 维度是固定的 (由 pid_n 决定)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n[None, :] < N
    
    # M 维度的基准 offset (将在循环中滑动)
    offs_m_base = tl.arange(0, BLOCK_M)

    # Batch/Head Offsets
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    
    # --- 3. Loop over M dimension ---
    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + offs_m_base
        mask_m = offs_m[:, None] < M
        
        # 这里的 mask 需要同时保护 M (行) 和 N (列)
        mask = mask_m & mask_n
        
        # 指针计算逻辑不变: base + m*stride_m + n*stride_n
        # 注意：现在 offs_n 是固定的，offs_m 在变
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        
        x_in = tl.load(x_ptrs, mask=mask)
        x = x_in.to(tl.float32)
        
        # --- Compute Tanh (Inline ASM) ---
        y = tl.inline_asm_elementwise(
            asm='tanh.approx.f32 $0, $1;',
            constraints=('=r,r'),
            args=[x],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        
        tl.store(y_ptrs, y.to(x_in.dtype), mask=mask)


def _ttir_of_tanh_ngrid(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    
    # --- Grid Configuration for N-Grid ---
    # Grid 维度 0 对应 N 的分块数量
    grid = (
        triton.cdiv(N, BN), 
        batch * heads,
    )
    
    triton_kernel = tanh_ngrid_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BM, 
        BLOCK_N=BN,
    )
    
    return triton_kernel.asm['ttir']

# print(_ttir_of_tanh_ngrid(M=128, N=128, BM=32, BN=32))