import triton
import triton.language as tl
import torch
from tilefusion.utils.utils import DEVICE

@triton.jit
def mul_kernel(
    x_ptr, y_ptr, z_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    stride_zb, stride_zh, stride_zm, stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads index
    
    # Calculate batch and head offset
    batch_head_offset_x = pid_bh * stride_xh  # Assuming contiguous heads within batch
    batch_head_offset_y = pid_bh * stride_yh
    batch_head_offset_z = pid_bh * stride_zh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m[:, None] < M
    for start_n in range(0, N, BLOCK_N):
        n = start_n + offs_n
        mask = m_mask & (n[None, :] < N)
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + n[None, :] * stride_yn
        z_ptrs = z_ptr + batch_head_offset_z + offs_m[:, None] * stride_zm + n[None, :] * stride_zn
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        y = tl.load(y_ptrs, mask=mask, other=0.0)
        # Element-wise multiplication
        z = x * y
        tl.store(z_ptrs, z, mask=mask)


def _ttir_of_mul(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Z = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    triton_kernel = mul_kernel[grid](
        X, Y, Z,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        Z.stride(0), Z.stride(1), Z.stride(2), Z.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']
