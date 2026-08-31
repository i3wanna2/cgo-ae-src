import triton
import triton.language as tl
import torch
from tilefusion.utils.utils import DEVICE


@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(x.dtype.is_int_unsigned(), "floating-point value must be passed as bits")
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def fpval_to_key(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) != 0, fm, tm)


@triton.jit
def key_to_fpval(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) == 0, fm, tm)


# stable top-k tie-breaks to value with smaller index
@triton.jit
def indx_to_key(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def key_to_indx(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def streaming_topk_striden(
    X, stride_xm, stride_xn, n_expts_tot, offs_m, mask_m,
    N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Like streaming_topk but supports arbitrary stride along N (last dim)."""
    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    if x_nbits < 16:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_ultype: tl.constexpr = tl.dtype(f"uint{y_nbits}")
    x_dtype: tl.constexpr = X.dtype.element_ty

    loop_iterations: tl.constexpr = N_EXPTS_PAD // BLOCK_N - 1
    offs_x_n = loop_iterations * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_x_n[None, :] < n_expts_tot

    # first iteration
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :] * stride_xn
    x = tl.load(X_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
    acc = tl.topk(x, N_EXPTS_ACT, dim=1)

    # subsequent iterations
    for _i in (tl.static_range if loop_iterations <= 4 else range)(loop_iterations):
        acc = tl.bitonic_merge(acc)
        X_ptrs -= BLOCK_N * stride_xn
        offs_x_n -= BLOCK_N
        x = tl.load(X_ptrs, mask=mask_m, other=float("-inf"))
        x = fpval_to_key(x.to(x_utype, bitcast=True))
        x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
        acc = tl.maximum(acc, tl.topk(x, N_EXPTS_ACT, dim=1))

    acc = (acc << (y_nbits - 16)) | (acc >> 16)
    acc = tl.sort(acc, dim=1, descending=True)
    y_indices_raw = (acc >> (y_nbits - 16)).to(tl.uint32)
    y_indices = key_to_indx(y_indices_raw, N_EXPTS_PAD)
    y_values_raw = acc.to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    return y_values, y_indices


@triton.jit
def topk_forward_bhmn(
    X, Yv, Yi, 
    B, H, M, N,  # shapes
    stride_xb, stride_xh, stride_xm, stride_xn,  # input base + strides for (B,H,M,N)
    stride_yb, stride_yh, stride_ym, stride_yn,  # output base + strides for (B,H,M,k)
    BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (grid_m, grid_bh) where grid_m = ceil_div(M, BLOCK_M), grid_bh = B*H
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # derive (b, h) from pid_bh
    b = pid_bh // H
    h = pid_bh % H

    # row indices within M for this program id
    m0 = pid_m * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)
    mask_m = (b < B) & (h < H) & (offs_m[:, None] < M)

    # constants and safety
    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)

    # base ptr for this (b,h) tile
    X_bh_base = X + b * stride_xb + h * stride_xh

    # compute top-k along N using general-stride version
    y_values, y_indices = streaming_topk_striden(
        X_bh_base, stride_xm, stride_xn, N, offs_m, mask_m, N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N
    )

    # write outputs
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    Y_bh_base = Yv + b * stride_yb + h * stride_yh
    Yi_bh_base = Yi + b * stride_yb + h * stride_yh
    Yv_ptrs = Y_bh_base + offs_m[:, None] * stride_ym + offs_y_n[None, :] * stride_yn
    Yi_ptrs = Yi_bh_base + offs_m[:, None] * stride_ym + offs_y_n[None, :] * stride_yn
    tl.store(Yv_ptrs, y_values, mask=mask_m)
    tl.store(Yi_ptrs, y_indices, mask=mask_m)


def topk_bhmn(x: torch.Tensor, k: int, dim: int = -1, BLOCK_M: int = 32, BLOCK_N: int = 32):
    """
    Compute top-k along the last dimension (N) for a 4D input tensor shaped (B, H, M, N).

    - No provided indices, no softmax, no bitmatrix; just values and indices.
    - Returns (values, indices) with shape (B, H, M, k).

    Constraints:
    - N must be < 2**16 (due to 16-bit index packing in the streaming algorithm).
    - Works best when the last dimension is contiguous.
    """
    assert x.ndim == 4, "input must be 4D (B, H, M, N)"
    assert dim in (-1, 3), "only top-k along the last dimension is supported"
    B, H, M, N = x.shape
    assert N < (1 << 16), "N must be < 65536"
    device = x.device

    # helpers
    cdiv = lambda a, b: (a + b - 1) // b
    n_cols_pad = cdiv(N, BLOCK_N) * BLOCK_N

    # allocate outputs directly in 4D
    y_vals = torch.empty((B, H, M, k), dtype=x.dtype, device=device)
    y_indx = torch.empty((B, H, M, k), dtype=torch.int32, device=device)

    # launch with 2D grid: (m_blocks, B*H)
    grid = (cdiv(M, BLOCK_M), B * H)
    topk_forward_bhmn[grid](
        x, y_vals, y_indx,
        B, H, M, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y_vals.stride(0), y_vals.stride(1), y_vals.stride(2), y_vals.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,
    )
    return y_vals, y_indx


def _ttir_of_topk_bhmn(
    M: int,
    N: int,
    K: int,
    BLOCK_M: int,
    BLOCK_N: int,
    B: int,
    H: int,
):
    """Generate TTIR for the TopK kernel over (B,H,M,N) along the last dim.

    Shapes:
        - Input X:  [B, H, M, N]
        - Output Y: [B, H, M, K]

    Returns:
        - TTIR text (str) compiled from `_topk_forward_bhmn` for the given params.
    """
    assert N < (1 << 16), "N must be < 65536 due to 16-bit packed indices"
    device = torch.device(DEVICE)

    # Helper for padded N used by the streaming algorithm
    cdiv = lambda a, b: (a + b - 1) // b
    n_cols_pad = cdiv(N, BLOCK_N) * BLOCK_N

    # Allocate dummy tensors on device to trigger compilation
    X = torch.randn((B, H, M, N), device=device, dtype=torch.float16)
    Yv = torch.empty((B, H, M, K), device=device, dtype=torch.float16)
    Yi = torch.empty((B, H, M, K), device=device, dtype=torch.int32)

    grid = (cdiv(M, BLOCK_M), B * H)
    compiled = topk_forward_bhmn[grid](
        X,
        Yv, Yi,
        B, H, M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Yv.stride(0), Yv.stride(1), Yv.stride(2), Yv.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=K,
    )
    return compiled.asm["ttir"]