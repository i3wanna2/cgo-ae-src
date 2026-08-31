import triton
import triton.language as tl
import torch
from tilefusion.utils.utils import DEVICE

# ==================== Mask Any Kernel ====================
@triton.jit
def mask_any_kernel(
    x_ptr, y_ptr, z_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    stride_zb, stride_zh, stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Mask any kernel: compare two inputs and reduce over M dimension.
    
    Input:  x [batch, heads, M, N] and y [batch, heads, M, N]
    Output: z [batch, heads, N] (any(x >= y) over M)
    
    Computes: output[b, h, n] = any over m of (input_x[b, h, m, n] >= input_y[b, h, m, n])
    """
    pid_n = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads index
    
    # Calculate batch and head offset
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    batch_head_offset_z = pid_bh * stride_zh
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n[None, :] < N
    
    # Accumulate any over all M blocks
    any_acc = tl.zeros([BLOCK_N], dtype=tl.int8)
    
    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask = (offs_m[:, None] < M) & n_mask
        
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        
        # x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        # y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs,)
        y = tl.load(y_ptrs,)
        # Compare: x >= y
        cond = x >= y
        
        # Any over M dimension (axis=0)
        block_any = tl.max(cond, axis=0).to(tl.int8)
        any_acc = any_acc | block_any
    
    # Store output
    z_ptrs = z_ptr + batch_head_offset_z + offs_n * stride_zn
    tl.store(z_ptrs, any_acc,)


def mask_any_triton(x: torch.Tensor, y: torch.Tensor, block_m: int = 64, block_n: int = 128) -> torch.Tensor:
    """Compare x >= y and reduce over M dimension using any operation.
    
    Args:
        x: [batch, heads, M, N] CUDA tensor (float16)
        y: [batch, heads, M, N] CUDA tensor (float16)
        block_m: Block size for M dimension
        block_n: Block size for N dimension
    
    Returns:
        z: [batch, heads, N] CUDA tensor (bool) - any(x >= y) over M
    """
    assert x.is_cuda and y.is_cuda, "Input must be CUDA tensors"
    assert x.dim() == 4 and y.dim() == 4, "Inputs must be 4D [batch, heads, M, N]"
    assert x.shape == y.shape, "Input shapes must match"
    
    batch, heads, M, N = x.shape
    z = torch.empty((batch, heads, N), device=x.device, dtype=torch.int8)
    
    grid = (
        triton.cdiv(N, block_n),
        batch * heads,
    )
    
    mask_any_kernel[grid](
        x, y, z,
        M, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        z.stride(0), z.stride(1), z.stride(2),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return z.to(torch.bool)


def _ttir_of_mask_any(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    """Generate TTIR for mask_any kernel.
    
    Input:  x [batch, heads, M, N] and y [batch, heads, M, N]
    Output: z [batch, heads, N]
    """
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Z = torch.empty(batch, heads, N, device=DEVICE, dtype=torch.bool)
    
    grid = (
        triton.cdiv(N, BN),
        batch * heads,
    )
    
    triton_kernel = mask_any_kernel[grid](
        X, Y, Z,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        Z.stride(0), Z.stride(1), Z.stride(2),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    
    return triton_kernel.asm['ttir']