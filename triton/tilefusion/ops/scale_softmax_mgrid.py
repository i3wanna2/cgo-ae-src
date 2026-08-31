import torch
import triton
import triton.language as tl
from tilefusion.utils.utils import DEVICE, _next_pow2


@triton.jit
def add_mask_kernel_mgrid(
    x_ptr, m_ptr, y_ptr,
    H, N,  # H: heads, N: sequence length
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Add mask kernel with grid (Batch, M, Head_blocks)
    
    Computes y = x + m element-wise.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_hb = tl.program_id(2)
    
    # 计算 head 范围
    off_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = off_h < H
    
    # 基地址
    X_base = x_ptr + pid_b * stride_xb + pid_m * stride_xm
    M_base = m_ptr + pid_b * stride_mb + pid_m * stride_mm
    Y_base = y_ptr + pid_b * stride_yb + pid_m * stride_ym
    
    # 循环处理 N 维度
    for start_n in range(0, N, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < N
        mask = mask_h[:, None] & mask_n[None, :]
        
        # 加载 x 和 m
        X_ptrs = X_base + off_h[:, None] * stride_xh + off_n[None, :] * stride_xn
        M_ptrs = M_base + off_h[:, None] * stride_mh + off_n[None, :] * stride_mn
        Y_ptrs = Y_base + off_h[:, None] * stride_yh + off_n[None, :] * stride_yn
        
        x = tl.load(X_ptrs, )
        m = tl.load(M_ptrs, )
        y = x + m
        
        tl.store(Y_ptrs, y,)


@triton.jit
def scale_kernel_mgrid(
    input_ptr, output_ptr,
    H, N,  # H: heads, N: sequence length
    stride_ib, stride_ih, stride_im, stride_in,
    stride_ob, stride_oh, stride_om, stride_on,
    scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Scale kernel with grid (Batch, M, Head_blocks)"""
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_hb = tl.program_id(2)
    
    # 计算 head 范围
    off_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = off_h < H
    
    # 基地址
    In_base = input_ptr + pid_b * stride_ib + pid_m * stride_im
    Out_base = output_ptr + pid_b * stride_ob + pid_m * stride_om
    
    # 循环处理 N 维度
    for start_n in range(0, N, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < N
        
        # 加载 [BLOCK_H, BLOCK_N]
        In_ptrs = In_base + off_h[:, None] * stride_ih + off_n[None, :] * stride_in
        data = tl.load(In_ptrs, )
        
        # Scale
        scaled = data * scale
        
        # 存储
        Out_ptrs = Out_base + off_h[:, None] * stride_oh + off_n[None, :] * stride_on
        tl.store(Out_ptrs, scaled,)


@triton.jit
def softmax_kernel_mgrid(
    input_ptr, output_ptr,
    H, N,
    stride_ib, stride_ih, stride_im, stride_in,
    stride_ob, stride_oh, stride_om, stride_on,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Softmax kernel with grid (Batch, M, Head_blocks)
    
    Computes softmax over N dimension. Uses exp2 optimization with only 2 loops.
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_hb = tl.program_id(2)
    
    # 计算 head 范围
    off_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = off_h < H
    
    # 基地址
    In_base = input_ptr + pid_b * stride_ib + pid_m * stride_im
    Out_base = output_ptr + pid_b * stride_ob + pid_m * stride_om
    
    # Precompute 1 / ln(2) for exp2 optimization
    INV_LN2: tl.constexpr = 1.4426950408889634
    
    # 第一遍：计算 denominator (使用 exp2)
    l_i = tl.zeros([BLOCK_H, 1], dtype=tl.float32)
    
    for start_n in range(0, N, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < N
        
        In_ptrs = In_base + off_h[:, None] * stride_ih + off_n[None, :] * stride_in
        data = tl.load(In_ptrs, ).to(tl.float32)
        
        # exp(x) = exp2(x * INV_LN2)
        x_scaled = data * INV_LN2
        numerators = tl.math.exp2(x_scaled)
        l_i += tl.sum(numerators, axis=1, keep_dims=True)
    
    # 第二遍：归一化并存储
    for start_n in range(0, N, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < N
        
        In_ptrs = In_base + off_h[:, None] * stride_ih + off_n[None, :] * stride_in
        data = tl.load(In_ptrs,).to(tl.float32)
        
        x_scaled = data * INV_LN2
        y = tl.math.exp2(x_scaled) / l_i
        
        Out_ptrs = Out_base + off_h[:, None] * stride_oh + off_n[None, :] * stride_on
        tl.store(Out_ptrs, y.to(tl.float16), )


def launch_add_mask_mgrid(
    x: torch.Tensor, 
    m: torch.Tensor, 
    BLOCK_H: int = 64, 
    BLOCK_N: int = 64
):
    """Launch add_mask kernel with mgrid layout
    
    Args:
        x: [B, H, M, N] tensor
        m: [B, H, M, N] tensor (mask to add)
        BLOCK_H: block size for heads dimension
        BLOCK_N: block size for N dimension
    
    Returns:
        y: [B, H, M, N] tensor, y = x + m
    """
    assert x.ndim == 4
    assert m.ndim == 4
    B, H, M, N = x.shape
    
    y = torch.empty_like(x)
    
    grid = (B, M, triton.cdiv(H, BLOCK_H))
    
    add_mask_kernel_mgrid[grid](
        x, m, y,
        H, N,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        m.stride(0), m.stride(1), m.stride(2), m.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
    )
    
    return y


def launch_scale_mgrid(input: torch.Tensor, scale: float, BLOCK_H: int = 64, BLOCK_N: int = 64):
    """Launch scale kernel with mgrid layout
    
    Args:
        input: [B, H, M, N] tensor
        scale: scalar to multiply
        BLOCK_H: block size for heads dimension
        BLOCK_N: block size for N dimension
    
    Returns:
        output: [B, H, M, N] tensor
    """
    assert input.ndim == 4
    B, H, M, N = input.shape
    
    output = torch.empty_like(input)
    
    grid = (B, M, triton.cdiv(H, BLOCK_H))
    
    scale_kernel_mgrid[grid](
        input, output,
        H, N,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale=scale,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
    )
    
    return output


def launch_softmax_mgrid(input: torch.Tensor, BLOCK_H: int = 64, BLOCK_N: int = 64):
    """Launch softmax kernel with mgrid layout
    
    Args:
        input: [B, H, M, N] tensor
        BLOCK_H: block size for heads dimension
        BLOCK_N: block size for N dimension
    
    Returns:
        output: [B, H, M, N] tensor with softmax applied over N dimension
    """
    assert input.ndim == 4
    B, H, M, N = input.shape
    
    output = torch.empty_like(input)
    
    grid = (B, M, triton.cdiv(H, BLOCK_H))
    
    softmax_kernel_mgrid[grid](
        input, output,
        H, N,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
    )
    
    return output


def _ttir_of_add_mask(M: int, TopK: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    """Generate TTIR for add_mask kernel"""
    x_tensor = torch.randn(batch, heads, M, TopK, device=DEVICE, dtype=torch.float16)
    m_tensor = torch.randn(batch, heads, M, TopK, device=DEVICE, dtype=torch.float16)
    y_tensor = torch.empty(batch, heads, M, TopK, device=DEVICE, dtype=torch.float16)
    
    grid = (batch, M, triton.cdiv(heads, BM))
    
    triton_kernel = add_mask_kernel_mgrid[grid](
        x_tensor, m_tensor, y_tensor,
        heads, TopK,
        x_tensor.stride(0), x_tensor.stride(1), x_tensor.stride(2), x_tensor.stride(3),
        m_tensor.stride(0), m_tensor.stride(1), m_tensor.stride(2), m_tensor.stride(3),
        y_tensor.stride(0), y_tensor.stride(1), y_tensor.stride(2), y_tensor.stride(3),
        BLOCK_H=BM,
        BLOCK_N=BN,
    )
    
    return triton_kernel.asm['ttir']


def _ttir_of_scale_mgrid(M: int, TopK: int, BM: int, BN: int, batch: int = 1, heads: int = 1, scale: float = 0.5):
    """Generate TTIR for scale kernel"""
    input_tensor = torch.randn(batch, heads, M, TopK, device=DEVICE, dtype=torch.float16)
    output_tensor = torch.empty(batch, heads, M, TopK, device=DEVICE, dtype=torch.float16)
    
    grid = (batch, M, triton.cdiv(heads, BM))
    
    triton_kernel = scale_kernel_mgrid[grid](
        input_tensor, output_tensor,
        heads, TopK,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        output_tensor.stride(0), output_tensor.stride(1), output_tensor.stride(2), output_tensor.stride(3),
        scale=scale,
        BLOCK_H=BM,
        BLOCK_N=BN,
    )
    
    return triton_kernel.asm['ttir']


def _ttir_of_softmax_mgrid(M: int, TopK: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    """Generate TTIR for softmax kernel"""
    input_tensor = torch.randn(batch, heads, M, TopK, device=DEVICE, dtype=torch.float16)
    output_tensor = torch.empty(batch, heads, M, TopK, device=DEVICE, dtype=torch.float16)
    
    grid = (batch, M, triton.cdiv(heads, BM))
    
    triton_kernel = softmax_kernel_mgrid[grid](
        input_tensor, output_tensor,
        heads, TopK,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        output_tensor.stride(0), output_tensor.stride(1), output_tensor.stride(2), output_tensor.stride(3),
        BLOCK_H=BM,
        BLOCK_N=BN,
    )
    
    return triton_kernel.asm['ttir']

@triton.jit
def softmax_reduce_mgrid_kernel(
    input_ptr, lse_ptr,
    H, N,
    stride_ib, stride_ih, stride_im, stride_in,      # Input strides
    stride_lb, stride_lh, stride_lm, stride_ln,      # LSE strides
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Softmax Reduce Kernel based on mgrid structure.
    Grid: (Batch, M, Head_blocks)
    Output: LSE (LogSumExp) or Sum(Exp) depending on logic. 
    Here consistent with your reference: Sum(Exp2(x * inv_ln2))
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # 对应单个 M
    pid_hb = tl.program_id(2) # 对应 Head Block

    # 1. 计算 Head 维度的偏移
    off_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = off_h < H

    # 2. 计算基地址 (Batch 和 M 是固定的)
    # Input: [Batch, Heads, M, N]
    In_base = input_ptr + pid_b * stride_ib + pid_m * stride_im
    
    # LSE: [Batch, Heads, M, 1]
    # 注意：LSE 的 M 维度通常在 Head 后面或者前面，取决于 Layout，这里使用通用 stride 计算
    Lse_base = lse_ptr + pid_b * stride_lb + pid_m * stride_lm

    # Constant
    INV_LN2: tl.constexpr = 1.4426950408889634

    # 3. 循环计算 Sum(Exp)
    # accumulator shape: [BLOCK_H, 1]
    l_i = tl.zeros([BLOCK_H, 1], dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < N

        # Load Input Block: [BLOCK_H, BLOCK_N]
        # 注意 mask 处理：需要同时 mask H 和 N
        In_ptrs = In_base + off_h[:, None] * stride_ih + off_n[None, :] * stride_in
        data = tl.load(In_ptrs, ).to(tl.float32)

        # Compute Exp2
        x_scaled = data * INV_LN2
        numerators = tl.math.exp2(x_scaled)
        
        # Accumulate: sum over N dim (axis=1)
        l_i += tl.sum(numerators, axis=1, keep_dims=True)

    # 4. 存储结果
    # LSE ptrs shape: [BLOCK_H, 1]
    # stride_ln 通常为 1 (因为最后一维是 1)
    Lse_ptrs = Lse_base + off_h[:, None] * stride_lh 
    tl.store(Lse_ptrs, l_i, )


def _ttir_of_softmax_reduce_mgrid(M: int, TopK: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    """Generate TTIR for softmax reduce kernel (mgrid version)"""
    # Input: (B, H, M, N)
    input_tensor = torch.randn(batch, heads, M, TopK, device=DEVICE, dtype=torch.float16)
    
    # Output (LSE): (B, H, M, 1) -> 这里的 1 是为了保持维度对齐，方便 view
    lse_tensor = torch.empty(batch, heads, M, 1, device=DEVICE, dtype=torch.float32)
    
    # Grid: (Batch, M, Head_blocks) - 保持和你原版 mgrid 一致
    grid = (batch, M, triton.cdiv(heads, BM))
    
    triton_kernel = softmax_reduce_mgrid_kernel[grid](
        input_tensor, lse_tensor,
        heads, TopK,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        lse_tensor.stride(0), lse_tensor.stride(1), lse_tensor.stride(2), lse_tensor.stride(3),
        BLOCK_H=BM,
        BLOCK_N=BN,
    )
    
    return triton_kernel.asm['ttir']

@triton.jit
def softmax_recompute_mgrid_kernel(
    # --- Pointers ---
    x_ptr, y_ptr, lse_ptr, 
    # --- Dimensions ---
    H, N, # M 由 grid 隐含
    # --- Strides ---
    stride_xb, stride_xh, stride_xm, stride_xn,
    stride_yb, stride_yh, stride_ym, stride_yn,
    stride_lseb, stride_lseh, stride_lsem, # stride_lsen 默认为1
    # --- Constexprs ---
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Grid: (Batch, M, triton.cdiv(Heads, BLOCK_H))
    逻辑: 一个 PID 处理具体的某一行 (pid_m)，以及一组 Heads。
          并在 Kernel 内部循环处理所有的 N。
    """
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)      # 对应具体的某一行 M
    pid_hb = tl.program_id(2)     # 对应 Head Block

    INV_LN2: tl.constexpr = 1.4426950408889634

    # --- 1. 计算 Head 维度的偏移 (Block H) ---
    off_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = off_h < H

    # --- 2. 计算基地址 ---
    # 由于 pid_m 确定了具体哪一行，这里直接加上 stride_xm
    # X/Y: [Batch, Heads, M, N] -> [Batch, Heads, pid_m, ...]
    X_base = x_ptr + pid_b * stride_xb + pid_m * stride_xm
    Y_base = y_ptr + pid_b * stride_yb + pid_m * stride_ym
    
    # LSE: [Batch, Heads, M, 1] -> [Batch, Heads, pid_m, 1]
    LSE_base = lse_ptr + pid_b * stride_lseb + pid_m * stride_lsem

    # --- 3. 加载 LSE (Denominator) ---
    # LSE 对于同一行的不同 N 是共享的，所以在循环外加载
    # Shape: [BLOCK_H, 1]
    lse_ptrs = LSE_base + off_h[:, None] * stride_lseh
    # mask=mask_h[:, None] 保护 Head 越界
    l_i = tl.load(lse_ptrs, ) 

    # --- 4. 循环遍历 N 维度 ---
    # 每次处理 [BLOCK_H, BLOCK_N] 的块
    for start_n in range(0, N, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < N
        
        # Load X: [BLOCK_H, BLOCK_N]
        x_ptrs = X_base + off_h[:, None] * stride_xh + off_n[None, :] * stride_xn
        x_in = tl.load(x_ptrs, )
        
        # Compute Softmax: exp2(x * inv_ln2) / sum
        x = x_in.to(tl.float32)
        x_scaled = x * INV_LN2
        # 注意: l_i 已经是 [BLOCK_H, 1], 会自动广播到 [BLOCK_H, BLOCK_N]
        y = tl.math.exp2(x_scaled) / l_i 
        
        # Store Y: [BLOCK_H, BLOCK_N]
        y_ptrs = Y_base + off_h[:, None] * stride_yh + off_n[None, :] * stride_yn
        tl.store(y_ptrs, y.to(x_in.dtype), )


def _ttir_of_softmax_recompute_mgrid(M: int, N: int, BM: int, BN: int, batch: int = 1, heads: int = 1):
    """
    生成 mgrid 版本 softmax recompute 的 TTIR
    Grid: (Batch, M, Head_Blocks)
    注意: 这里的 BM 对应的是 Kernel 中的 BLOCK_H (Head Block Size)，而不是 M 的分块。
          因为 Grid 中 M 是完全展开的 (M dimension size is M).
    """
    X = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    Y = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    LSE = torch.empty((batch, heads, M, 1), device=DEVICE, dtype=torch.float32)
    
    # 按照你的要求构造 Grid
    grid = (
        batch, 
        M, 
        triton.cdiv(heads, BM) # 这里 BM 当作 Block_Heads 使用
    )
    
    triton_kernel = softmax_recompute_mgrid_kernel[grid](
        X, Y, LSE,
        heads, N,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        LSE.stride(0), LSE.stride(1), LSE.stride(2),
        BLOCK_H=BM, # 将传入的 BM 参数映射给 BLOCK_H
        BLOCK_N=BN,
    )
    
    return triton_kernel.asm['ttir']


def test_scale_softmax():
    """Test scale, softmax, and add_mask kernels"""
    torch.manual_seed(0)
    B, H, M, N = 1, 128, 4096, 2048
    scale = 0.5
    
    # 测试 add_mask
    print("Testing add_mask kernel...")
    x = torch.randn(B, H, M, N, device=DEVICE, dtype=torch.float16)
    m = torch.randn(B, H, M, N, device=DEVICE, dtype=torch.float16)
    output_triton = launch_add_mask_mgrid(x, m)
    output_ref = x + m
    
    torch.testing.assert_close(output_triton, output_ref, rtol=1e-3, atol=1e-3)
    print("✓ Add mask kernel passed")
    
    # 测试 scale
    print("\nTesting scale kernel...")
    input_data = torch.randn(B, H, M, N, device=DEVICE, dtype=torch.float16)
    output_triton = launch_scale_mgrid(input_data, scale)
    output_ref = input_data * scale
    
    torch.testing.assert_close(output_triton, output_ref, rtol=1e-3, atol=1e-3)
    print("✓ Scale kernel passed")
    
    # 测试 softmax
    print("\nTesting softmax kernel...")
    input_data = torch.randn(B, H, M, N, device=DEVICE, dtype=torch.float16)
    output_triton = launch_softmax_mgrid(input_data)
    output_ref = torch.softmax(input_data.float(), dim=-1).to(torch.float16)
    
    torch.testing.assert_close(output_triton, output_ref, rtol=1e-2, atol=1e-2)
    print("✓ Softmax kernel passed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_scale_softmax()
