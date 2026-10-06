"""
Corm Attention - 自动搜索最优拆分策略

使用ComputeGraph的search_optimal_partition功能：
1. 构建完整的Corm计算图（包含mask_any）
2. 自动枚举所有拆分方案
3. 编译并benchmark每个方案
4. 选择性能最优的拆分策略
"""

import math
import torch
import torch.nn.functional as F
import time

from tilefusion.core.compute_graph import ComputeGraph
from tilefusion.utils.utils import DEVICE, validate_correctness, benchmark_performance


def torch_attention_ref(Q, Kmat, V, Mask, scale, corm_mask):
    """PyTorch参考实现（含Corm score）"""
    scores = Q @ Kmat.transpose(-2, -1)
    scores = scores * scale
    scores = scores + Mask
    
    # 计算softmax的中间值
    numerators = torch.exp(scores)
    denominators = torch.sum(numerators, dim=-1, keepdim=True)
    probs = numerators / denominators
    
    # Attention输出
    O = probs @ V
    
    # Corm score: any(probs >= corm_mask) over M dimension
    corm_score = (probs >= corm_mask).any(dim=2)  # [batch, heads, M, N] -> [batch, heads, N]
    
    return O, corm_score, denominators


def main():
    # 问题规模
    batch = 1
    heads = 32
    M = 4096
    N = 4096
    K = 128
    scale = 1.0 / math.sqrt(K)
    
    # Block大小
    BM = 64
    BN = 64
    
    # 性能参数
    num_warps = 4
    num_stages = 3
    
    print("=" * 70)
    print("Corm Attention - Auto Search Optimal Partition")
    print("=" * 70)
    print(f"Problem size: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")
    print(f"Block size: BM={BM}, BN={BN}")
    print(f"Performance params: num_warps={num_warps}, num_stages={num_stages}")
    
    # 准备输入数据
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    
    # Causal mask
    causal_mask_bool = torch.triu(
        torch.ones(M, N, dtype=torch.bool, device=DEVICE),
        diagonal=1
    )
    causal_mask_float = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
    
    # Corm mask
    corm_mask = torch.ones(M, N, dtype=torch.float16, device=DEVICE)
    for i in range(M):
        corm_mask[i] /= (i + 1)
    corm_mask_expanded = corm_mask.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
    
    # ========================================================================
    # 构建完整的Corm计算图（不手动拆分）
    # ========================================================================
    print("\n" + "=" * 70)
    print("Building Full Corm Compute Graph")
    print("=" * 70)
    
    # 创建完整计算图（gemm_qk -> scale -> add_mask -> softmax_store -> gemm_pv + mask_any）
    full_graph = ComputeGraph("corm_full_graph")
    
    # 添加输入节点
    full_graph.add_input("Q", "K", "V", "Mask", "CormMask")
    
    common_params_mk = {"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    common_params_m = {"M": M, "N": N, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    common_params_loopk = {"M": M, "N": K, "K": N, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    
    # 节点索引: 0=Q, 1=K, 2=V, 3=Mask, 4=CormMask
    # 5=gemm_qk, 6=scale, 7=add_mask, 8=softmax_store, 9=gemm_pv, 10=mask_any
    full_graph.add_node("gemm_qk", inputs={"Q": "q_ptr", "K": "k_ptr"}, parents=[0, 1], **common_params_mk) \
              .add_node("scale", inputs={"scores": "input"}, parents=[5], **common_params_mk, scale=scale) \
              .add_node("add_mask", inputs={"output": "input", "Mask": "mask_ptr"}, parents=[6, 3], **common_params_m) \
              .add_node("softmax_store", inputs={"output": "input"}, parents=[7], **common_params_m) \
              .add_node("gemm_pv", inputs={"output": "probs", "V": "v_ptr"}, parents=[8, 2], **common_params_loopk) \
              .add_node("mask_any", inputs={"output": "input", "CormMask": "mask"}, parents=[8, 4], **common_params_m) \
              .add_output("attn_output", source=9, shape=(batch, heads, M, K), device=DEVICE) \
              .add_output("corm_score", source=10, shape=(batch, heads, N), dtype=torch.bool, device=DEVICE)
    
    # ========================================================================
    # 准备全局输入和参数字典（用于自动参数构建）
    # ========================================================================
    global_inputs = {
        'Q': Q,
        'K': Kmat,
        'V': V,
        'Mask': Mask,
        'CormMask': corm_mask_expanded
    }
    
    params_dict = {
        'M': M, 'N': N, 'K': K,
        'BM': BM, 'BN': BN,
        'batch': batch, 'heads': heads,
        'scale': scale
    }
    
    # ========================================================================
    # 自动搜索最优拆分（使用自动参数构建）
    # ========================================================================
    print("\n" + "=" * 70)
    print("Starting Automatic Partition Search with Auto Arg Construction")
    print("=" * 70)
    
    best_compiled, best_partition, best_time = full_graph.compile(
        num_warps=num_warps,
        num_stages=num_stages,
        search_optimal=True,
        global_inputs=global_inputs,
        params_dict=params_dict,
        num_warmup=10,
        num_repeat=50,
        max_splits=1,  # 允许最多2个切分点（3个子图）
        device=DEVICE
    )
    
    # 验证
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    corm_output = None
    fused_args_list, _, output_tensor, intermediate_tensors = \
        full_graph._build_fused_args_from_partition(best_partition, global_inputs, params_dict, DEVICE)

    Out_baseline = torch.empty(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    corm_score_baseline = torch.empty(batch, heads, N, device=DEVICE, dtype=torch.bool)
    softmax_sum_baseline = torch.empty(batch, heads, M, 1, device=DEVICE, dtype=torch.float32)
    
    def fused_launcher():
        for (compiled, grid), subgraph_args in zip(best_compiled, fused_args_list):
            compiled[grid](*subgraph_args)

    def torch_baseline_launcher():
        attn_out, corm_out, sm_out = torch_attention_ref(Q, Kmat, V, Mask, scale, corm_mask_expanded)
        Out_baseline.copy_(attn_out)
        corm_score_baseline.copy_(corm_out)
        softmax_sum_baseline.copy_(sm_out)

    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    
    benchmark_performance(
        torch_baseline_launcher,
        fused_launcher
    )
    
    print("\n" + "=" * 70)
    print("Correctness Validation")
    print("=" * 70)
    
    # 使用预分配的输出 tensor
    outputs = full_graph.outputs
    print(f"\nOutputs: {list(outputs.keys())}")
    
    # 验证 attention output
    if "attn_output" in outputs:
        print("\nCheck Attention Output (gemm_pv):")
        validate_correctness(
            torch_baseline_launcher,
            fused_launcher,
            Out_baseline,
            outputs["attn_output"],
            rtol=1e-3,
            atol=1e-2
        )
    else:
        print("\nattn_output not found")

    # 验证 Corm score output
    if "corm_score" in outputs:
        print("\nCheck Corm Score Output (mask_any):")
        corm_output = outputs["corm_score"]
        print(f"  Corm output dtype: {corm_output.dtype}")
        validate_correctness(
            torch_baseline_launcher,
            fused_launcher,
            corm_score_baseline,
            outputs["corm_score"],
            rtol=0,
            atol=0
        )
    else:
        print("\ncorm_score not found")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
