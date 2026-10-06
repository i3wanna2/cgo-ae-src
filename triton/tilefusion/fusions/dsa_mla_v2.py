"""
dsa_mla_v2.py

参考 `dsa_mla.py`、`h2o_auto_search_v2.py` 和 `attn_v4.py`，
使用 ComputeGraph + kernel_registry 构建 DSA (Dynamic Sparse Attention) 融合流程。

计算图结构：
  gather_gemm_qk -> scale -> add_mask -> softmax -> gather_gemm_sv

与 dsa_mla.py 不同：
- 使用 ComputeGraph 自动管理 kernel 融合
- Kernel 已在 kernel_registry.py 中注册
- 支持 search_optimal 自动搜索最佳拆分策略
"""

import math
import torch
import torch.nn.functional as F

from tilefusion.core.compute_graph import ComputeGraph
from tilefusion.utils.utils import DEVICE, validate_correctness, benchmark_performance


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    """
    参考实现 (来自 dsa_mla.py)
    
    Args:
        q: [B, M, H, D] Query (注意：维度顺序与 ComputeGraph 不同)
        kv: [B, N, G, D] Key/Value 合并 (G=1)
        indices: [B, M, G, TopK] 稀疏索引 (注意：维度顺序与 ComputeGraph 不同)
        sm_scale: softmax scale
        is_casual: 是否使用 causal mask
        
    Returns:
        o: [B, M, H, Dv] 输出 (注意：维度顺序与 ComputeGraph 不同)
    """
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

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


def generate_safe_mla_indices(B, G, M, TopK, device="cuda"):
    """
    生成符合 Sparse MLA 要求的稀疏索引
    
    策略:
    1. Sink Token: 初始化全为 0
    2. Causal: 第 t 步只能从 [0, ..., t] 中采样
    3. Random: 当历史长度 > TopK 时，随机采样 TopK 个不重复的位置
    """
    indices = torch.zeros(B, G, M, TopK, dtype=torch.int32, device=device)
    
    for t in range(M):
        valid_len = t + 1
        
        if valid_len <= TopK:
            selection = torch.arange(valid_len, device=device, dtype=torch.int32)
            indices[:, :, t, :valid_len] = selection
        else:
            # 使用 randperm 采样 TopK 个
            for b in range(B):
                for g in range(G):
                    perm = torch.randperm(valid_len, device=device)[:TopK]
                    indices[b, g, t, :] = perm.to(torch.int32)
    
    return indices


def main():
    # 问题规模 (与 dsa_mla.py 保持一致)
    batch = 1
    heads = 128
    M = 4096    # Sequence length
    N = 4096    # KV cache length
    K = 576     # Query head dim (128 + 16)
    D = 512     # Value head dim
    TopK = 2048 # Sparse attention top-k
    G = 1       # Number of KV groups (MQA/MLA style)
    
    scale = 1.0 / math.sqrt(576)  # 通常用 128 而非 144
    
    BM = 64
    BN = 64
    num_warps = 8
    num_stages = 3
    
    print("=" * 70)
    print("DSA MLA v2 - Using ComputeGraph + Kernel Registry")
    print("=" * 70)
    print(f"Problem: batch={batch}, heads={heads}, M={M}, N={N}, K={K}, D={D}, TopK={TopK}")
    print(f"Block: BM={BM}, BN={BN}, warps={num_warps}, stages={num_stages}")
    
    # 准备输入数据
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, G, N, K, device=DEVICE, dtype=torch.float16)
    # 注意：V 在 DSA MLA 中就是 K，只是在计算时取前 D(128) 维
    # 所以这里 V 实际上不需要单独传入
    
    # 生成稀疏索引
    indices = generate_safe_mla_indices(batch, G, M, TopK, device=DEVICE)
    
    # Causal mask: 上三角为 -inf
    # Use expand() instead of .contiguous() to save memory
    # expand() creates a VIEW with strides (0, 0, TopK, 1) - NO memory copy
    causal_mask_bool = torch.triu(
        torch.ones(M, TopK, dtype=torch.bool, device=DEVICE),
        diagonal=1
    )
    causal_mask_float = torch.zeros(M, TopK, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)
    # Note: Do NOT call .contiguous() - kernel handles broadcasting via strides
    
    # ========================================================================
    # 构建 ComputeGraph
    # ========================================================================
    print("\nBuilding ComputeGraph...")
    graph = ComputeGraph("dsa_mla_graph")
    
    # 添加输入节点: Q, K, Indices, Mask
    # 注意：V 不需要单独输入，gather_gemm_sv 会直接使用 K 的前 D 维
    graph.add_input("Q", "K", "Indices", "Mask")
    # 节点索引: 0=Q, 1=K, 2=Indices, 3=Mask
    
    # 公共参数
    common_params = {
        "M": M, "N": N, "TopK": TopK, "K": K, "D": D,
        "BM": BM, "BN": BN, "batch": batch, "heads": heads
    }
    
    # 节点定义 (参考 dsa_mla.py 的融合顺序)
    # 5: gather_gemm_qk (Q, K, Indices) -> scores
    graph.add_node(
        "gather_gemm_qk_mgrid",
        inputs={"Q": "Q_ptr", "K": "K_ptr", "Indices": "Indices_ptr"},
        parents=[0, 1, 2],
        **common_params
    )
    
    # 6: scale (scores) -> scaled_scores
    graph.add_node(
        "scale_mgrid",
        inputs={"scores": "input"},
        parents=[4],
        M=M, TopK=TopK, BM=BM, BN=BN, batch=batch, heads=heads, scale=scale
    )
    
    # 7: add_mask (scaled_scores, Mask) -> masked_scores
    graph.add_node(
        "add_mask_mgrid",
        inputs={"output": "input", "Mask": "mask"},
        parents=[5, 3],
        M=M, TopK=TopK, BM=BM, BN=BN, batch=batch, heads=heads
    )
    
    # 8: softmax (masked_scores) -> probs
    graph.add_node(
        "softmax_mgrid",
        inputs={"output": "input"},
        parents=[6],
        M=M, TopK=TopK, BM=BM, BN=BN, batch=batch, heads=heads
    )
    
    # 9: gather_gemm_sv (probs, K, Indices) -> output
    # 注意：这里用 K 而不是 V，因为在 MLA 中 V 就是 K 的前 D 维
    # inputs 的 key 是来源名称（InputNode名称或producer输出名称），value 是 consumer 的参数名称
    graph.add_node(
        "gather_gemm_sv_mgrid",
        inputs={"output": "Scores", "K": "V", "Indices": "Indices"},
        parents=[7, 1, 2],  # 7=softmax输出, 1=K(InputNode), 2=Indices(InputNode)
        **common_params
    )
    
    # 添加输出节点
    graph.add_output("attn_output", source=8, shape=(batch, heads, M, D), device=DEVICE)

    # ========================================================================
    # 编译和搜索
    # ========================================================================
    global_inputs = {
        'Q': Q,
        'K': Kmat,
        'Indices': indices,
        'Mask': Mask,
    }
    
    params_dict = {
        'M': M, 'N': N, 'K': K, 'D': D, 'TopK': TopK,
        'BM': BM, 'BN': BN,
        'batch': batch, 'heads': heads,
        'scale': scale,
    }
    
    print("\nSearching for the best split...")
    best_compiled, best_partition, best_time = graph.compile(
        num_warps=num_warps,
        num_stages=num_stages,
        search_optimal=True,
        global_inputs=global_inputs,
        params_dict=params_dict,
        num_warmup=5,
        num_repeat=20,
        max_splits=1,
        device=DEVICE,
    )
    
    # ========================================================================
    # 验证和性能测试
    # ========================================================================
    print("\n" + "=" * 70)
    print("Correctness check")
    print("=" * 70)
    
    fused_args_list, _, output_tensor, intermediate_tensors = \
        graph._build_fused_args_from_partition(best_partition, global_inputs, params_dict, DEVICE)
    
    Out_baseline = torch.empty(batch, heads, M, D, device=DEVICE, dtype=torch.float16)
    
    def fused_launcher():
        for (compiled, grid), subgraph_args in zip(best_compiled, fused_args_list):
            compiled[grid](*subgraph_args)
    
    def torch_baseline_launcher():
        # ref_sparse_mla_fwd_interface 需要的维度顺序: [B, M, H, D]
        # 我们的数据是 [B, H, M, D]，需要 permute
        res = ref_sparse_mla_fwd_interface(
            Q.permute(0, 2, 1, 3),      # [B, H, M, D] -> [B, M, H, D]
            Kmat.permute(0, 2, 1, 3),   # [B, G, N, D] -> [B, N, G, D]
            indices.permute(0, 2, 1, 3), # [B, G, M, TopK] -> [B, M, G, TopK]
            sm_scale=scale,
            is_casual=True
        )
        # res: [B, M, H, Dv] -> [B, H, M, Dv]
        Out_baseline.copy_(res.permute(0, 2, 1, 3))
    
    # 使用预分配的输出 tensor
    outputs = graph.outputs
    print(f"\nOutputs: {list(outputs.keys())}")
    
    target_output = outputs.get("attn_output")
    if target_output is None:
        # fallback 到旧方式
        for k, t in intermediate_tensors.items():
            if t.shape == (batch, heads, M, D):
                target_output = t
                break
        if target_output is None:
            target_output = output_tensor if isinstance(output_tensor, torch.Tensor) else list(intermediate_tensors.values())[-1]
    
    print("\nPerformance comparison:")
    benchmark_performance(torch_baseline_launcher, fused_launcher)
    
    print("\nCorrectness check:")
    validate_correctness(
        torch_baseline_launcher,
        fused_launcher,
        Out_baseline,
        target_output,
        rtol=1e-1,
        atol=1e-1,
    )
    
    print("\n" + "=" * 70)
    print("DSA MLA v2 done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
