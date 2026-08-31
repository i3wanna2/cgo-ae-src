import torch
import triton
import triton.language as tl
import atexit
from tilefusion.utils.utils import DEVICE


# ========================================================================
# 逐元素操作: Scale (Grid N, Loop M) - (新版)
# ========================================================================

@triton.jit
def scale_ngrid_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    scale,  # runtime scalar factor
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + offs_m_base
        
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        
        x = tl.load(x_ptrs)
        y = x * scale
        tl.store(y_ptrs, y)

def launch_scale_ngrid(X: torch.Tensor, scale: float, BLOCK_M: int = 64, BLOCK_N: int = 64):
    batch, heads, M, N = X.shape
    Y = torch.empty_like(X)
    grid = (
        triton.cdiv(N, BLOCK_N), # Grid 划分 N
        batch * heads,
    )
    scale_ngrid_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return Y

def _ttir_of_scale_ngrid(M: int, N: int, BM: int, BN: int, scale: float, batch: int = 1, heads: int = 1):
    
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(N, BN), # Grid 划分 N
        batch * heads,
    )
    triton_kernel = scale_ngrid_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        scale,
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']

@triton.jit
def scale_ngrid_kernel_c(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    scale : tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + offs_m_base
        
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        
        x = tl.load(x_ptrs)
        y = x * scale
        tl.store(y_ptrs, y)

def _ttir_of_scale_ngrid_c(M: int, N: int, BM: int, BN: int, scale: float, batch: int = 1, heads: int = 1):
    
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(N, BN), # Grid 划分 N
        batch * heads,
    )
    triton_kernel = scale_ngrid_kernel_c[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        scale=scale,
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']

# ========================================================================
# 逐元素操作: Add Mask (Grid N, Loop M) - (新版)
# ========================================================================

@triton.jit
def add_mask_ngrid_kernel(
    x_ptr, m_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_m = pid_bh * stride_mh
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + offs_m_base

        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        m_ptrs = m_ptr + batch_head_offset_m + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        
        x = tl.load(x_ptrs, )
        mm = tl.load(m_ptrs, )
        y = x + mm
        tl.store(y_ptrs, y)

def launch_add_mask_ngrid(X: torch.Tensor, Mask: torch.Tensor, BLOCK_M: int = 64, BLOCK_N: int = 64):
    batch, heads, M, N = X.shape
    Y = torch.empty_like(X)
    grid = (
        triton.cdiv(N, BLOCK_N), # Grid 划分 N
        batch * heads,
    )
    add_mask_ngrid_kernel[grid](
        X, Mask, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Mask.stride(0), Mask.stride(1), Mask.stride(2), Mask.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return Y

def _ttir_of_addmask_ngrid(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Mask = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(N, BN), # Grid 划分 N
        batch * heads,
    )
    triton_kernel = add_mask_ngrid_kernel[grid](
        X, Mask, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Mask.stride(0), Mask.stride(1), Mask.stride(2), Mask.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']


# ========================================================================
# 测试
# ========================================================================
if __name__ == "__main__":
    print("Running tests on CUDA device.")
    # 定义测试形状
    B, H, M, K, N = 4, 16, 256, 64, 128
    
    # 块大小
    BM, BN = 64, 64

    # --- GEMM 测试 ---
    print(f"\nTesting GEMM B={B}, H={H}, M={M}, K={K}, N={N}")
    
    A_gemm = torch.randn(B, H, M, K, device=DEVICE, dtype=torch.float16)
    B_gemm = torch.randn(B, H, K, N, device=DEVICE, dtype=torch.float16)
    
    C_torch_gemm = torch.matmul(A_gemm, B_gemm)

    # --- Scale 测试 ---
    print(f"\nTesting Scale B={B}, H={H}, M={M}, N={N}")
    A_elem = torch.randn(B, H, M, N, device=DEVICE, dtype=torch.float16)
    scale_factor = 5.0
    
    C_torch_scale = A_elem * scale_factor
    C_scale_ngrid = launch_scale_ngrid(A_elem, scale_factor, BLOCK_M=BM, BLOCK_N=BN)

    torch.testing.assert_close(C_scale_ngrid, C_torch_scale, rtol=1e-2, atol=1e-3, msg="N-Grid Scale Kernel mismatch")
    print("✅ New Kernel (scale_ngrid_kernel) PASSED")
    print("--- All Scale tests passed ---")

    # --- AddMask 测试 ---
    print(f"\nTesting AddMask B={B}, H={H}, M={M}, N={N}")
    Mask_elem = torch.randn(B, H, M, N, device=DEVICE, dtype=torch.float16)

    C_torch_add = A_elem + Mask_elem
    C_add_ngrid = launch_add_mask_ngrid(A_elem, Mask_elem, BLOCK_M=BM, BLOCK_N=BN)
    
    torch.testing.assert_close(C_add_ngrid, C_torch_add, rtol=1e-2, atol=1e-3, msg="N-Grid AddMask Kernel mismatch")
    print("✅ New Kernel (add_mask_ngrid_kernel) PASSED")
    print("--- All AddMask tests passed ---")


    print("\nAll tests passed!")

# ========================================================================
# 逐元素操作: Log Neg (Grid N, Loop M) - y = -log(x)
# ========================================================================

@triton.jit
def log_neg_ngrid_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + offs_m_base
        
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        
        x = tl.load(x_ptrs)
        x_fp32 = x.to(tl.float32)
        y = -tl.log(x_fp32)
        tl.store(y_ptrs, y.to(tl.float16))

def _ttir_of_log_neg_ngrid(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (triton.cdiv(N, BN), batch * heads)
    
    triton_kernel = log_neg_ngrid_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']

# ========================================================================
# 逐元素操作: Add (Grid N, Loop M) - z = x + y
# ========================================================================

@triton.jit
def add_ngrid_kernel(
    x_ptr, y_ptr, z_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    stride_zb, stride_zh, stride_zm, stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)
    
    batch_head_offset_x = pid_bh * stride_xh
    batch_head_offset_y = pid_bh * stride_yh
    batch_head_offset_z = pid_bh * stride_zh
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)

    for start_m in range(0, M, BLOCK_M):
        offs_m = start_m + offs_m_base
        
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        z_ptrs = z_ptr + batch_head_offset_z + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
        
        x = tl.load(x_ptrs)
        y = tl.load(y_ptrs)
        z = x + y
        tl.store(z_ptrs, z)

def _ttir_of_add_ngrid(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Z = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (triton.cdiv(N, BN), batch * heads)
    
    triton_kernel = add_ngrid_kernel[grid](
        X, Y, Z,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        Z.stride(0), Z.stride(1), Z.stride(2), Z.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']
