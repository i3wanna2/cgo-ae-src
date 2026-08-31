import torch
import torch
import torch.nn.functional as F

import math
import time
import tempfile
import torch
import torch.nn.functional as F
import triton
from triton.compiler import compile as triton_compile

# Use local module imports (same style as other demos)
from tilefusion.utils.utils import DEVICE, build_combined_module, benchmark_performance, validate_correctness
from tilefusion.core.compiler import fuse_kernels_in_ttir,opt_kernels_in_ttir, opt_kernels_in_ttir_test
from tilefusion.ops.gather_matmul_qk import _ttir_of_gather_mm_qk
from tilefusion.ops.gather_matmul_sv import _ttir_of_gather_mm_sv
from tilefusion.ops.scale_softmax_mgrid import _ttir_of_scale_mgrid, _ttir_of_softmax_mgrid, _ttir_of_add_mask

def torch_attention_ref(q, k, v, indices, mask, sm_scale):
    """
    Args:
        q:       # [B, H, Sq, 576]  (输入可以是 B-S-H-D)
        k:       # [B, G, Sk, 576] (输入可以是 B-S-G-D)
        v:       # [B, G, Sk, 512] (输入可以是 B-S-G-D)
        indices: # [B, H, Sq, TopK] (稀疏索引，Padding=0 指向 Sink)
        mask :   # [B, H, Sq, Sk] (加法掩码，Padding=-inf 指向 Sink)
        
    Returns:
        o:       [B, Sq, H, 512]
    """
    # 0. 维度定义
    b, h, sq, dim_q = q.shape
    _, g, sk, _ = k.shape
    dim_content = 128
     
    # 2. 处理 GQA (显式重复 KV)
    # 既然要用标准的 Batch-Head 格式，就必须把 G 扩展成 H
    # 这样 Q 和 K 才能一一对应
    n_rep = h // g
    if n_rep > 1:
        # [B, G, Sk, D] -> [B, G, n_rep, Sk, D] -> [B, H, Sk, D]
        k = k.unsqueeze(2).expand(b, g, n_rep, sk, dim_q).reshape(b, h, sk, dim_q)
        v = v.unsqueeze(2).expand(b, g, n_rep, sk, dim_content).reshape(b, h, sk, dim_content)

    # 填值：indices 指向的位置填 0.0
    # 这里的 scatter 直接在最后一维 (dim=3) 操作，非常符合直觉
    mask = mask.scatter(3, indices, 0.0)

    # 3. 计算 Score (Standard Matmul)
    # [B, H, Sq, D] @ [B, H, D, Sk] -> [B, H, Sq, Sk]
    score = torch.matmul(q, k.transpose(-1, -2))
    score *= sm_scale
    # 5. Apply Mask & Softmax
    # 此时 score 和 mask 都是 [B, H, Sq, Sk]，完美对齐
    score = score + mask
    prob = torch.softmax(score, dim=-1)
    # 6. 计算 Output
    # [B, H, Sq, Sk] @ [B, H, Sk, Dv] -> [B, H, Sq, Dv]
    o = torch.matmul(prob, v)

    return o

def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    # assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(
        0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
            1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda").view(1, -1)

    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :1 - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(torch.float16)

def build_and_compile_fused_kernel(args, args2, M, N, K, BM, BN, scale, batch, heads, num_warps=4, num_stages=2):
    topk = 2048
    current_ttir = _ttir_of_gather_mm_qk(M, N, topk, K, BM, BN, batch=batch, heads=heads)
    current_producer_name = "gather_gemm_dot_kernel"

    # Define fusion stages with 4D support
    # Verified parameter mappings:
    # For now, only fuse up to softmax (GEMM_loopk has dependency issues)
    stages = [
        # Stage 1: GEMM + Scale
        ("scale_kernel_mgrid",             lambda: _ttir_of_scale_mgrid(M, topk, BM, BN, batch, heads, scale),  
         [3,4,5,15,16,17,15,16,17], [0,2,3,4,5,6,7,8,9]),
        ("add_mask_kernel_mgrid",           lambda: _ttir_of_add_mask(M, topk, BM, BN, batch, heads),       
         [18,4,5,15,16,17,15,16,17,15,16,17], [0,3,4,5,6,7,8,9,10,11,12,13]),
        # Stage 3: (GEMM+Scale+AddMask) + Softmax
        ("softmax_kernel_mgrid",           lambda: _ttir_of_softmax_mgrid(M, topk, BM, BN, batch, heads),       
         [20,4,5,15,16,17,15,16,17], [0,2,3,4,5,6,7,8,9]),
        # # Note: gemm_loopk fusion disabled due to dependency issues - using torch.matmul instead
        ("gather_gemm_score_v_stacked_kernel", lambda: _ttir_of_gather_mm_sv(M, N, topk, 512, BM, BN, batch=batch, heads=heads),  
         [21,1,2,4,5,15,16,17,9,10,11,12,13,14], [0,1,2,4,5,6,7,8,9,10,11,12,13,14]),
    ]

    fused_path = None
    for consumer_kernel_name, ttir_fn, prod_out_idx, cons_in_idx in stages:
        consumer_ttir = ttir_fn()
        combined = build_combined_module(current_ttir, consumer_ttir, current_producer_name, consumer_kernel_name)
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
            f.write(combined)
            combined_path = f.name
        fused_path, fused_ttir_text = fuse_kernels_in_ttir(
            combined_path,
            producer_kernel_name=current_producer_name,
            consumer_kernel_name=consumer_kernel_name,
            producer_output_arg_idx=prod_out_idx,
            consumer_input_arg_idx=cons_in_idx,
        )
        current_ttir = fused_ttir_text
        current_producer_name = f"{current_producer_name}_{consumer_kernel_name}"
    fused_path, fused_ttir_text = opt_kernels_in_ttir(fused_path, targetaxis=1, direction="0", mask_arg_index=19)
    # Compile kernel with optimization parameters
    options = {
        'num_warps': num_warps,
        'num_stages': num_stages,
    }
    compiled2 = triton_compile(fused_path, options=options)
    grid = (batch, M, triton.cdiv(heads, BM))
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled2.asm['ttir'])
        combined_path = f.name
        print(f"Fused2 module written to: {combined_path}")    
        
    def fused_launcher():
        compiled2[grid](*args2)

    return fused_launcher

def generate_safe_mla_indices(B, H, Sq, TopK, device="cuda"):
    """
    生成符合 Sparse MLA 要求的稀疏索引。
    Shape: [B, H, Sq, TopK]
    
    策略:
    1. Sink Token: 初始化全为 0。当历史长度不足 TopK 时，Padding 部分自动指向第 0 个 token。
    2. Causal: 第 t 步只能从 [0, ..., t] 中采样。
    3. Random: 当历史长度 > TopK 时，随机采样 TopK 个不重复的位置。
    """
    # 1. 初始化: 全 0
    # 这一步至关重要，它保证了所有未被填满的位置（Padding）都指向合法的索引 0
    indices = torch.zeros(B, H, Sq, TopK, dtype=torch.int32, device=device)
    
    # 2. 逐时间步填充 (Loop over Query Sequence)
    # 虽然这里有个 Python 循环，但只循环 Sq 次，内部是 B*H 的向量化操作，速度很快
    for t in range(Sq):
        # 当前合法的 Key 范围是 [0, 1, ..., t]
        # 长度是 t + 1
        valid_len = t + 1
        
        if valid_len <= TopK:
            # === Case A: 历史长度不够 TopK ===
            # 直接选取所有历史 token [0, 1, ..., t]
            # 剩下的位置保持初始化时的 0 (作为重复的 Sink Token)
            
            # 生成: [0, 1, ..., t]
            selection = torch.arange(valid_len, device=device)
            
            # 广播赋值给所有 Batch 和 Head
            # indices[..., :valid_len] = [0, 1, ..., t]
            # indices[..., valid_len:] = 0 (保持原样)
            indices[:, :, t, :valid_len] = selection
            
        else:
            # === Case B: 历史长度超过 TopK ===
            # 需要从 [0, ..., t] 中随机采样 TopK 个不重复的索引
            
            # 技巧: 使用 rand + topk 实现“无放回随机采样”
            # 生成随机噪声: [B, H, t+1]
            noise = torch.rand(B, H, valid_len, device=device)
            
            # 取 TopK 的下标，这等价于随机选择了 TopK 个位置
            # values 不重要，indices 才是我们需要的 random index
            _, selected_indices = torch.topk(noise, k=TopK, dim=-1)
            
            # 赋值
            indices[:, :, t, :] = selected_indices
            
    return indices

def main():
    # Problem sizes with 4D support (batch, heads, M, N/K)
    batch = 1
    heads = 128
    M = 4096    # M dimension (per head)
    N = 4096    # N dimension (per head)
    K = 576      # K dimension (head_dim)
    D = 512      # Value dimension (head_dim_v)
    scale = 1.0 / math.sqrt(576)
    topk = 2048
    print(f"Problem size: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")

    # Prepare input data (4D tensors)
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, 1, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, 1, N, D, device=DEVICE, dtype=torch.float16)
    Mask = torch.full((batch, heads, M, N), fill_value=-torch.inf, device=DEVICE, dtype=torch.float16)
    # Mask_fuse = torch.full((batch, heads, M, N), fill_value=-torch.inf, device=DEVICE, dtype=torch.float16)
    # Allocate output buffers (4D)
    scores_fused = torch.zeros(batch, heads, M, topk, device=DEVICE, dtype=torch.float16)
    scale_out_fused = torch.zeros(batch, heads, M, topk, device=DEVICE, dtype=torch.float16)
    softmax_out_fused = torch.zeros(batch, heads, M, topk, device=DEVICE, dtype=torch.float16)
    Out_fused = torch.zeros(batch, heads, M, D, device=DEVICE, dtype=torch.float16)
    Out_baseline = torch.zeros(batch, heads, M, D, device=DEVICE, dtype=torch.float16)
    Out_baseline2 = torch.zeros(batch, heads, M, D, device=DEVICE, dtype=torch.float16)
    indicis = generate_safe_mla_indices(batch, 1, M, topk, device=DEVICE)
    indicis_expanded = indicis.expand(batch, heads, M, topk).contiguous()
    causal_mask_bool = torch.triu(
        torch.ones(M, topk, dtype=torch.bool, device=DEVICE),
        diagonal=1
    )  # shape: (M, N)
    causal_mask_float = torch.zeros(M, topk, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))

    # Step 3: 扩展到 (batch, heads, M, N)
    Mask_fuse = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
    Add_mask_out = torch.zeros(batch, heads, M, topk, device=DEVICE, dtype=torch.float16)
    args = []
    args2 = [
        Q, Kmat, indicis, scores_fused,
        heads, topk,
        Q.stride(0), Q.stride(1), Q.stride(2),
        Kmat.stride(0), Kmat.stride(1), Kmat.stride(2),
        indicis.stride(0), indicis.stride(1), indicis.stride(2),
        scores_fused.stride(0), scores_fused.stride(1), scores_fused.stride(2),
        scale_out_fused,
        Mask_fuse, Add_mask_out,
        softmax_out_fused,
        Out_fused,
        Out_fused.stride(0), Out_fused.stride(1), Out_fused.stride(2),
    ]

    BM = 32
    BN = 128
    num_warps = 8
    num_stages = 3

    fused_launcher = build_and_compile_fused_kernel(args, args2, M, N, K, BM, BN, scale, batch, heads, num_warps, num_stages)
    
    def torch_baseline_launcher():
        # torch_attention_ref returns (O, h2o_score)
        mask_out = torch_attention_ref(Q, Kmat, Kmat, indicis_expanded, Mask, scale)
        # Out_baseline.copy_(attn_out)
        Out_baseline.copy_(mask_out)
        # softmax_out.copy_(mask_out)

    def ref_sparse_mla():
        res = ref_sparse_mla_fwd_interface(Q.permute(0,2,1,3), 
                                            Kmat.permute(0,2,1,3), 
                                            # V.permute(0,2,1,3),
                                            indicis.permute(0,2,1,3), 
                                            sm_scale=scale, 
                                            is_casual=True)
        Out_baseline2.copy_(res.permute(0,2,1,3))

    # Validate H2O scores
    print("\n" + "="*60)
    print("Validating H2O Scores")
    print("="*60)
    validate_correctness(
        ref_sparse_mla, 
        fused_launcher, 
        [ Out_baseline2], 
        [ Out_fused],
        rtol=1e-1, 
        atol=1e-1
    )
    benchmark_performance(ref_sparse_mla, fused_launcher)


if __name__ == "__main__":
    main()
