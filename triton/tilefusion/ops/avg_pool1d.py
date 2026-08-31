import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from tilefusion.utils.utils import DEVICE

# --- Kernel 定义 ---
@triton.jit
def avg_pool1d_kernel(
    x_ptr,
    output_ptr,
    N,
    stride_xb, stride_xh, stride_xn,
    stride_ob, stride_oh, stride_on,
    KERNEL_SIZE: tl.constexpr,
    PADDING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Grid 索引
    pid_bh = tl.program_id(1)  # Batch * Head 扁平索引
    pid_col = tl.program_id(0)  # N 维度分块索引
    
    # 2. 计算当前 Block 的列偏移
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3. 计算基地址偏移
    x_batch_head_offset = pid_bh * stride_xh
    out_batch_head_offset = pid_bh * stride_oh
    
    # 4. 累加逻辑
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # 静态循环展开
    for k in range(KERNEL_SIZE):
        # 计算滑窗对应的输入列坐标
        target_col_idx = col_offsets + k - PADDING
        
        # 边界检查: avg_pool 的 padding 逻辑是越界补 0
        load_mask = (target_col_idx >= 0) & (target_col_idx < N)
        
        # 计算具体加载地址
        load_ptr = x_ptr + x_batch_head_offset + target_col_idx * stride_xn
        
        # 加载数据，mask 之外补 0.0
        val = tl.load(load_ptr, mask=load_mask, other=0.0)
        accumulator += val

    # 5. 计算均值 (对应 PyTorch count_include_pad=True)
    avg_val = accumulator / KERNEL_SIZE
    
    # 6. 存储结果
    out_ptr = output_ptr + out_batch_head_offset + col_offsets * stride_on
    store_mask = col_offsets < N
    tl.store(out_ptr, avg_val.to(tl.float16), mask=store_mask)


# --- Python 调用包装 ---
def triton_avg_pool1d(x, output, kernel_size=5, stride=1, padding=2, BN=64):
    """
    Triton implementation of 1D average pooling.
    
    Args:
        x: Input tensor [B, H, N]
        output: Output tensor [B, H, N] (pre-allocated)
        kernel_size: Pooling kernel size
        stride: Stride (only stride=1 supported)
        padding: Padding size
        BN: Block size for N dimension
    """
    assert stride == 1, "Only stride=1 is supported"
    assert x.shape == output.shape, "Input and output shapes must match"
    
    B, H, N = x.shape
    
    # Grid: (N方向分块数, B*H 总行数)
    grid = (triton.cdiv(N, BN), B * H)
    
    avg_pool1d_kernel[grid](
        x, output, 
        N,
        x.stride(0), x.stride(1), x.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        KERNEL_SIZE=kernel_size,
        PADDING=padding,
        BLOCK_SIZE=BN,
    )
    
    return output

def _ttir_of_avg_pool1d(N: int, BN: int, kernel_size: int, padding: int, batch: int, heads: int,):
    """
    Generate TTIR for avg_pool1d kernel.
    
    Args:
        B: Batch size
        H: Number of heads
        N: Sequence length
        kernel_size: Pooling kernel size
        padding: Padding size
        BN: Block size for N dimension
    
    Returns:
        TTIR string
    """
    X = torch.randn(batch, heads, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, N, device=DEVICE, dtype=torch.float16)
    grid = (triton.cdiv(N, BN), batch * heads)
    
    triton_kernel = avg_pool1d_kernel[grid](
        X, Y,
        N,
        X.stride(0), X.stride(1), X.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        KERNEL_SIZE=kernel_size,
        PADDING=padding,
        BLOCK_SIZE=BN,
    )
    return triton_kernel.asm['ttir']


# --- 正确性验证 ---
def verify_correctness():
    torch.manual_seed(42)
    
    # 配置参数
    B, H, N = 1, 32, 4096
    KERNEL = 5
    PAD = 2
    
    print(f"Testing shape: ({B}, {H}, {N}), Kernel={KERNEL}, Pad={PAD}")
    
    # 1. 准备数据 (CUDA)
    x = torch.randn((B, H, N), device='cuda', dtype=torch.float16)
    
    # 2. PyTorch 原生实现 (基准)
    # PyTorch avg_pool1d 需要 (Batch, C, L) 格式
    # 我们将 (B*H) 视为 Batch, 1 视为 Channel
    x_torch = x.view(-1, 1, N) 
    
    y_torch = F.avg_pool1d(
        x_torch, 
        kernel_size=KERNEL, 
        padding=PAD, 
        stride=1, 
        count_include_pad=True # 这一步很关键！Triton 代码目前对应这种模式
    ).view(B, H, N)
    
    # 3. Triton 实现
    y_triton = torch.empty_like(x)
    triton_avg_pool1d(x, y_triton, kernel_size=KERNEL, padding=PAD)
    
    # 4. 误差分析
    diff = torch.abs(y_torch - y_triton)
    max_diff = diff.max().item()
    
    if torch.allclose(y_torch, y_triton, atol=1e-5):
        print(f"✅ Pass! Max Diff: {max_diff:.8f}")
    else:
        print(f"❌ Fail! Max Diff: {max_diff:.8f}")
        # 打印出错位置的值方便调试
        idx = torch.argmax(diff)
        print(f"Mismatch at flat index {idx.item()}:")
        print(f"  PyTorch: {y_torch.flatten()[idx].item()}")
        print(f"  Triton:  {y_triton.flatten()[idx].item()}")

if __name__ == "__main__":
    verify_correctness()