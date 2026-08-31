import triton
import triton.language as tl
import torch
from tilefusion.utils.utils import DEVICE

@triton.jit
def relu_kernel(
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
        x = tl.load(x_ptrs, )
        # ReLU activation: max(0, x)
        y = tl.maximum(x, 0.0)
        tl.store(y_ptrs, y,)


def _ttir_of_relu(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    triton_kernel = relu_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']

def relu_triton(X: torch.Tensor, BM: int = 128, BN: int = 128) -> torch.Tensor:

    # 获取维度信息
    batch, heads, M, N = X.shape
    
    # 2. 分配输出张量
    Y = torch.empty_like(X)
    
    # 3. 定义 Grid
    # 根据你的 kernel 定义：axis 0 是 M 的分块，axis 1 是 Batch * Heads
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    
    # 4. 启动 Kernel
    # 注意：Triton 会自动处理 torch.Tensor 到指针的转换
    relu_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    
    return Y


@triton.jit
def gelu_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_bh = tl.program_id(axis=1)  # batch * heads index
    
    # 计算 Batch 和 Head 的偏移
    batch_head_offset_x = pid_bh * stride_xh 
    batch_head_offset_y = pid_bh * stride_yh
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    m_mask = offs_m[:, None] < M
    
    # 沿着 N 维度循环处理
    for start_n in range(0, N, BLOCK_N):
        n = start_n + offs_n
        mask = m_mask & (n[None, :] < N)
        
        # 指针计算
        x_ptrs = x_ptr + batch_head_offset_x + offs_m[:, None] * stride_xm + n[None, :] * stride_xn
        y_ptrs = y_ptr + batch_head_offset_y + offs_m[:, None] * stride_ym + n[None, :] * stride_yn
        
        x = tl.load(x_ptrs, ).to(tl.float32)
        
        # --- GeLU 算法实现 ---
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        # Triton 里的 tl.extra.cuda.libdevice.tanh 或者简单的 tl.sigmoid 变体也可以
        # 这里使用标准近似公式
        inner = 0.79788456 * (x + 0.044715 * x * x * x) # 0.7978... 是 sqrt(2/pi)
        gelu_out = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
        
        tl.store(y_ptrs, gelu_out,)

def gelu_triton(X: torch.Tensor, BM: int = 64, BN: int = 64) -> torch.Tensor:
    batch, heads, M, N = X.shape
    Y = torch.empty_like(X)
    
    grid = (
        triton.cdiv(M, BM),
        batch * heads,
    )
    
    gelu_kernel[grid](
        X, Y,
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        BLOCK_M=BM, BLOCK_N=BN,
    )
    
    return Y

def _ttir_of_gelu(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    grid = (triton.cdiv(M, BM), batch * heads)
    
    # 编译并获取 TTIR
    triton_kernel = gelu_kernel[grid](
        X, torch.empty_like(X),
        M, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        X.stride(0), X.stride(1), X.stride(2), X.stride(3), # 借用 stride
        BLOCK_M=BM, BLOCK_N=BN,
    )
    return triton_kernel.asm['ttir']