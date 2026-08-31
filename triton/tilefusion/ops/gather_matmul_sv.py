import torch
import triton
import triton.language as tl

# ==========================================
# Triton Kernel: Stacked Heads Score @ Shared V
# ==========================================

@triton.jit
def gather_gemm_score_v_stacked_kernel(
    Scores_ptr,      # [B, H, M, TopK]
    V_ptr,           # [B, G, N, D]
    Indices_ptr,     # [B, H, M, TopK] (或 [B, G, M, TopK])
    Out_ptr,         # [B, H, M, D]
    
    # 维度
    H, TopK,         
    
    # Strides
    stride_sb, stride_sh, stride_sm, stride_sk,
    stride_vb, stride_vg, stride_vn, stride_vd,
    stride_ib, stride_ih, stride_im, stride_ik,
    stride_ob, stride_oh, stride_om, stride_od,
    
    # Meta Parameters
    BLOCK_H: tl.constexpr,   # 设为 16 或 32，利用 Head 堆叠
    BLOCK_K: tl.constexpr,   # Reduction 分块 (例如 64)
    BLOCK_D: tl.constexpr,   # 128
):
    # 1. Grid 解析: (Batch, SeqLen_M, Head_Blocks)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_hb = tl.program_id(2)

    # 2. 确定当前 Block 负责的一组 Heads
    off_h = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    off_k = tl.arange(0, BLOCK_K)
    off_d = tl.arange(0, BLOCK_D)
    mask_h = off_h < H

    # 3. 基础指针计算
    S_base = Scores_ptr + pid_b * stride_sb + pid_m * stride_sm
    V_base = V_ptr + pid_b * stride_vb  # G=1, 直接跳过 group 维度
    I_base = Indices_ptr + pid_b * stride_ib + pid_m * stride_im  # 共享 indices，取第一个 head
    O_base = Out_ptr + pid_b * stride_ob + pid_m * stride_om

    # 4. 初始化 Accumulator [BLOCK_H, BLOCK_D]
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    # 5. Reduction Loop (TopK 维度)
    for start_k in range(0, TopK, BLOCK_K):
        curr_k = start_k + off_k
        mask_k = curr_k < TopK
        
        # A. Load Scores [BLOCK_H, BLOCK_K]
        s_ptrs = S_base + off_h[:, None] * stride_sh + curr_k[None, :] * stride_sk
        scores = tl.load(s_ptrs,)
        indices = tl.load(I_base + curr_k * stride_ik,)
        v_ptrs = V_base + off_d[:, None] * stride_vd + indices[None, :] * stride_vn
        v = tl.load(v_ptrs,)  # [BLOCK_D, BLOCK_K]

        acc += tl.dot(scores, tl.trans(v))
    C_out = acc.to(Scores_ptr.dtype.element_ty)
    # 6. Store Result [BLOCK_H, BLOCK_D]
    o_ptrs = O_base + off_h[:, None] * stride_oh + off_d[None, :] * stride_od
    tl.store(o_ptrs, C_out,)


def run_gather_gemm_stacked(scores, v, indices):
    B, H, M, TopK = scores.shape
    _, G, N, D = v.shape
    
    out = torch.empty((B, H, M, D), device=scores.device, dtype=torch.float16)
    
    # ================= 配置 =================
    # 使用 32 个 Head 堆叠。
    # 为什么不用 64? 因为 64*128 的累加器太大，容易爆寄存器。
    # 32 是一个很好的平衡点。
    BLOCK_H = 64  
    BLOCK_K = 64
    BLOCK_D = 128
    
    # Grid: (B, M, H_Blocks) -> 这里的 Grid 大小大大减小了
    grid = (B, M, triton.cdiv(H, BLOCK_H))
    # =======================================
    
    gather_gemm_score_v_stacked_kernel[grid](
        scores, v, indices, out,
        H, TopK,
        # Strides
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # Meta
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D
    )
    
    return out

# ================= 辅助函数 =================
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

# ================= 测试代码 =================
if __name__ == "__main__":
    torch.manual_seed(2024)
    DEVICE = "cuda"
    
    # 1. 参数配置 (G=1 场景)
    B = 1
    G = 1          # Shared V (MHA/MQA)
    H = 128        # Heads
    M = 4096       # Sequence Length
    N = 4096       # KV Cache Length
    TopK = 2048    # Selected K
    D = 128        # Head Dim
    
    print(f"Config: B={B}, H={H}, G={G}, M={M}, D={D}, TopK={TopK}")
    print(f"Strategy: Stacked Heads (BLOCK_H=32). Heads share Indices/V.")
    
    # 2. 数据构造
    scores = torch.randn((B, H, M, TopK), device=DEVICE, dtype=torch.float16)
    # scores = torch.softmax(scores, dim=-1).to(torch.float16)
    
    v = torch.randn((B, G, N, D), device=DEVICE, dtype=torch.float16)
    
    # 关键：必须保证 Indices 是共享的，否则 kernel 的优化前提不成立
    # 生成 [B, G, M, TopK] 的 indices，然后广播到 [B, H, M, TopK]
    indices_per_group = generate_safe_mla_indices(B, G, M, TopK, device=DEVICE)  # [B, G, M, TopK]
    # 将每个 Group 的 indices 复制给该 Group 内的所有 Heads
    H_PER_G = H // G
    indices = indices_per_group.unsqueeze(2).expand(B, G, H_PER_G, M, TopK).reshape(B, H, M, TopK)
    
    # 3. 运行 Triton Kernel
    print("Running Triton Kernel (Stacked Heads)...")
    out_triton = run_gather_gemm_stacked(scores, v, indices_per_group)
    
    # 4. 运行 Reference 验证 (逐 Head 验证，防止 OOM)
    print("Running Verification (Iterative to save memory)...")
    
    passed = True
    max_diff = 0.0
    
    # 不创建巨大的 Tensor，而是直接在循环里对比
    for b in range(B):
        for h in range(H):
            g_idx = h // (H // G)
            
            # 1. 准备当前 Head 的数据
            curr_score = scores[b, h]         # [M, TopK]
            curr_indices = indices[b, h]      # [M, TopK]
            curr_v_group = v[b, g_idx]        # [N, D]
            
            # 2. Gather V (只针对当前 Head)
            # 显存占用: 4096 * 2048 * 128 * 2 bytes ≈ 2 GB (安全)
            gathered_v_head = curr_v_group[curr_indices] # [M, TopK, D]
            
            # 3. Matmul
            # [M, 1, TopK] @ [M, TopK, D] -> [M, 1, D]
            ref_out_head = torch.matmul(curr_score.unsqueeze(1), gathered_v_head).squeeze(1)
            ref_out_head = ref_out_head.to(torch.float16)
            
            # 4. 对比 Triton 结果
            tri_out_head = out_triton[b, h].float() # 转 float32 对比更稳
            ref_out_head_f32 = ref_out_head.float()
            
            # 简单检查
            curr_diff = (tri_out_head - ref_out_head_f32).abs().max().item()
            if curr_diff > max_diff:
                max_diff = curr_diff
                
            try:
                torch.testing.assert_close(tri_out_head, ref_out_head_f32, rtol=1e-2, atol=2e-2)
            except AssertionError:
                print(f"❌ Mismatch at Batch {b}, Head {h}")
                print(f"Max Diff: {curr_diff}")
                passed = False
                break
        
        if not passed: break

    if passed:
        print(f"✅ Test Passed! Max Diff: {max_diff:.6f}")
    else:
        print("❌ Test Failed.")


def _ttir_of_gather_mm_sv(M: int, N: int, TopK: int, D: int, BLOCK_H: int = 64, BLOCK_K: int = 64, BLOCK_D1: int = 128, batch: int = 1, heads: int = 128):
    """Generate TTIR for gather_gemm_score_v_stacked_kernel"""
    DEVICE = "cuda"
    G = 1
    
    scores = torch.randn((batch, heads, M, TopK), device=DEVICE, dtype=torch.float16)
    v = torch.randn((batch, G, N, D), device=DEVICE, dtype=torch.float16)
    indices = torch.randint(0, N, (batch, G, M, TopK), device=DEVICE, dtype=torch.int32)
    out = torch.empty((batch, heads, M, D), device=DEVICE, dtype=torch.float16)
    
    grid = (batch, M, triton.cdiv(heads, BLOCK_H))
    
    triton_kernel = gather_gemm_score_v_stacked_kernel[grid](
        scores, v, indices, out,
        heads, TopK,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D1,
    )
    
    return triton_kernel.asm['ttir']