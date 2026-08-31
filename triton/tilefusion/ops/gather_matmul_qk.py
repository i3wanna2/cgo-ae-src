import torch
import triton
import triton.language as tl
from tilefusion.utils.utils import DEVICE, _next_pow2

@triton.jit
def gather_gemm_dot_kernel(
    Q_ptr, K_ptr, Indices_ptr, Out_ptr,
    # 维度参数
    H, TopK,        # Indices 的长度 (Selected TopK)
    # Strides
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kg, stride_kn, stride_kd,
    stride_ib, stride_ig, stride_im, stride_ik,
    stride_ob, stride_oh, stride_om, stride_ok,
    # Meta parameters
    BLOCK_H: tl.constexpr,   # 每次处理的 Head 数 (例如 16 or 32)
    BLOCK_N: tl.constexpr,   # TopK 维度的分块大小 (例如 32 or 64)
    BLOCK_D1: tl.constexpr,  # 第一部分 Head Dim (128)
    BLOCK_D2: tl.constexpr,  # 第二部分 Head Dim (16)
):
    # 1. Grid 解析: (Batch, SeqLen, Head_Blocks)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_hb = tl.program_id(2)

    # 2. 计算 Heads 和 Group 索引
    off_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = off_h < H

    # 3. 计算各个指针的基地址 (简化版)
    Q_base = Q_ptr + pid_b * stride_qb + pid_m * stride_qm
    K_base = K_ptr + pid_b * stride_kb 
    Idx_base = Indices_ptr + pid_b * stride_ib + pid_m * stride_im
    Out_base = Out_ptr + pid_b * stride_ob + pid_m * stride_om

    # 4. 加载 Q 的两部分
    # Q1: [BLOCK_H, BLOCK_D1] - 前 128 维
    off_d1 = tl.arange(0, BLOCK_D1)
    Q1_ptrs = Q_base + off_h[:, None] * stride_qh + off_d1[None, :] * stride_qd
    q1 = tl.load(Q1_ptrs, mask=mask_h[:, None], other=0.0)
    
    # Q2: [BLOCK_H, BLOCK_D2] - 后 16 维
    off_d2 = BLOCK_D1 + tl.arange(0, BLOCK_D2)
    Q2_ptrs = Q_base + off_h[:, None] * stride_qh + off_d2[None, :] * stride_qd
    q2 = tl.load(Q2_ptrs, mask=mask_h[:, None], other=0.0)

    # 5. 循环处理 TopK 维度
    for start_n in range(0, TopK, BLOCK_N):
        off_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = off_n < TopK

        # 加载 indices [BLOCK_N]
        Idx_ptrs = Idx_base + off_n * stride_ik
        idx = tl.load(Idx_ptrs,)
        
        # K1: [BLOCK_D1, BLOCK_N] - 前 128 维
        K1_ptrs = K_base + off_d1[:, None] * stride_kd + idx[None, :] * stride_kn
        k1 = tl.load(K1_ptrs,)
        
        # K2: [BLOCK_D2, BLOCK_N] - 后 16 维
        K2_ptrs = K_base + off_d2[:, None] * stride_kd + idx[None, :] * stride_kn
        k2 = tl.load(K2_ptrs,)
        
        # 两个 dot 相加
        # dot1: [BLOCK_H, BLOCK_D1] @ [BLOCK_D1, BLOCK_N] -> [BLOCK_H, BLOCK_N]
        # dot2: [BLOCK_H, BLOCK_D2] @ [BLOCK_D2, BLOCK_N] -> [BLOCK_H, BLOCK_N]
        out1 = tl.dot(q1, k1)
        out2 = tl.dot(q2, k2)
        out = out1 + out2

        # 存储结果
        Out_ptrs = Out_base + off_h[:, None] * stride_oh + off_n[None, :] * stride_ok
        tl.store(Out_ptrs, out,)

def generate_safe_mla_indices(B, H, Sq, TopK, device="cuda"):
    indices = torch.zeros(B, H, Sq, TopK, dtype=torch.long, device=device)
    for t in range(Sq):
        valid_len = t + 1
        if valid_len <= TopK:
            selection = torch.arange(valid_len, device=device)
            indices[:, :, t, :valid_len] = selection
        else:
            noise = torch.rand(B, H, valid_len, device=device)
            _, selected_indices = torch.topk(noise, k=TopK, dim=-1)
            indices[:, :, t, :] = selected_indices
    return indices

def run_gather_gemm():
    # 配置
    B, H, M, D = 1, 128, 4096, 144  # D = 128 + 16
    G = 1            
    TopK = 2048
    N = 4096
    
    # 构造数据
    torch.manual_seed(0)
    q = torch.randn((B, H, M, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((B, G, N, D), device=DEVICE, dtype=torch.float16)
    # 这里的 indices 生成逻辑保持你原有的
    indices = generate_safe_mla_indices(B, G, M, TopK, device=DEVICE) 
    
    out_triton = torch.empty((B, H, M, TopK), device=DEVICE, dtype=torch.float16)

    # Kernel 配置
    BLOCK_H = 64
    BLOCK_N = 64
    BLOCK_D1 = 128  # 第一个 dot 处理 128 维
    BLOCK_D2 = 16   # 第二个 dot 处理 16 维

    grid = (B, M, triton.cdiv(H, BLOCK_H))

    gather_gemm_dot_kernel[grid](
        q, k, indices, out_triton,
        H, TopK,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        out_triton.stride(0), out_triton.stride(1), out_triton.stride(2), out_triton.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, BLOCK_D1=BLOCK_D1, BLOCK_D2=BLOCK_D2
    )
    
    for b in range(B):
        for h in range(H):
            # 1. 准备数据
            g_idx = h // (H // G)
            curr_q = q[b, h, :, :].unsqueeze(1) # Shape: [M, 1, D]
            curr_indices = indices[b, g_idx, :, :]  # Shape: [M, TopK]

            curr_k_group = k[b, g_idx, :, :]
            gathered_k = curr_k_group[curr_indices] # Shape: [M, TopK, D]

            # =========================================================
            # 修改点：使用 matmul
            # [M, 1, D] @ [M, D, TopK] -> [M, 1, TopK]
            # =========================================================
            # 关键：gathered_k 需要转置最后两个维度 (.transpose(-1, -2))
            ref_out = torch.matmul(curr_q, gathered_k.transpose(-1, -2)) 

            # matmul 出来的结果是 [M, 1, TopK]，需要把中间那个 1 去掉，变成 [M, TopK]
            ref_out = ref_out.squeeze(1)

            # 模拟截断 (保持你原有的逻辑)
            ref_out = ref_out.to(torch.float16).float()

            # 验证
            tri_out = out_triton[b, h, :, :].float()
            torch.testing.assert_close(ref_out, tri_out, rtol=1e-2, atol=1e-2, equal_nan=True)


def _ttir_of_gather_mm_qk(M: int, N: int, TopK: int, D: int, BLOCK_H: int = 64, BLOCK_N: int = 64, BLOCK_D1: int = 128, BLOCK_D2: int = 16, batch: int = 1, heads: int = 128):
    """Generate TTIR for gather_gemm_dot_kernel
    
    D should be BLOCK_D1 + BLOCK_D2
    """
    G = 1
    
    q = torch.randn((batch, heads, M, D), device=DEVICE, dtype=torch.float16)
    k = torch.randn((batch, G, N, D), device=DEVICE, dtype=torch.float16)
    indices = torch.randint(0, N, (batch, G, M, TopK), device=DEVICE, dtype=torch.int32)
    out = torch.empty((batch, heads, M, TopK), device=DEVICE, dtype=torch.float16)
    
    grid = (batch, M, triton.cdiv(heads, BLOCK_H))
    
    triton_kernel = gather_gemm_dot_kernel[grid](
        q, k, indices, out,
        heads, TopK,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, BLOCK_D1=BLOCK_D1, BLOCK_D2=BLOCK_D2,
    )
    
    return triton_kernel.asm['ttir']


if __name__ == "__main__":
    run_gather_gemm()