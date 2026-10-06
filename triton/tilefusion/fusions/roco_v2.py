"""
ROCO Attention V2 - 自动搜索最优拆分策略

使用ComputeGraph的search_optimal_partition功能：
1. 构建完整的ROCO计算图（包含Attention Output和两个Score Output）
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


def torch_attention_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, Mask: torch.Tensor, scale: float) -> tuple:
    """Pure-Torch reference for attention with ROCO (Robust Contextual Compression) scoring:
    scores = softmax((Q @ K^T) * scale + Mask, dim=-1) @ V
    roco_score = sum(attention_probs, dim=M) for token importance
    roco_sq_score = sum(attention_probs^2, dim=M) for variance-aware importance
    
    - Computes in fp32 for stability and casts back to Q.dtype (fp16) at the end.
    - All tensors are expected to be on the same CUDA device.
    
    Shapes:
        Q: [batch, heads, M, K], Kmat: [batch, heads, K, N], V: [batch, heads, N, K], Mask: [batch, heads, M, N]
    Returns: 
        O: [batch, heads, M, K] with dtype == Q.dtype (output attention)
        roco_score: [batch, heads, N] (sum of attention weights over sequence dimension)
        roco_sq_score: [batch, heads, N] (sum of squared attention weights)
    """
    assert Q.is_cuda and Kmat.is_cuda and V.is_cuda and Mask.is_cuda, "torch ref expects CUDA tensors"
    q = Q
    v = V
    mask = Mask
    
    # Compute attention scores
    scores = q @ Kmat.transpose(-2, -1)  # [batch, heads, M, K] @ [batch, heads, K, N] -> [batch, heads, M, N]
    scores = scores * scale
    scores = scores + mask
    
    # Compute attention probabilities
    numerators = torch.exp(scores)
    denominators = torch.sum(numerators, dim=-1, keepdim=True)
    probs = numerators / denominators  # [batch, heads, M, N]
    
    # Compute output
    O = probs @ v  # [batch, heads, M, N] @ [batch, heads, N, K] -> [batch, heads, M, K]
    
    # Compute ROCO scores: sum over query sequence dimension (dim=2)
    roco_score = probs.sum(dim=2)  # [batch, heads, M, N] -> [batch, heads, N]
    roco_sq_score = (probs ** 2).sum(dim=2)  # [batch, heads, M, N] -> [batch, heads, N]
    
    return O, roco_score, roco_sq_score, denominators


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
    print("ROCO Attention V2 - Auto Search Optimal Partition")
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
    
    # ========================================================================
    # 构建完整的ROCO计算图
    # ========================================================================
    print("\n" + "=" * 70)
    print("Building Full ROCO Compute Graph")
    print("=" * 70)
    
    # 创建完整计算图
    full_graph = ComputeGraph("roco_full_graph")
    
    # 添加输入节点
    full_graph.add_input("Q", "K", "V", "Mask")
    
    common_params_mk = {"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    common_params_m = {"M": M, "N": N, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    common_params_loopk = {"M": M, "N": K, "K": N, "BM": BM, "BN": BN, "batch": batch, "heads": heads}

    full_graph.add_node("gemm_qk", inputs={"Q": "q_ptr", "K": "k_ptr"}, parents=[0, 1], **common_params_mk) \
              .add_node("scale", inputs={"scores": "input"}, parents=[4], **common_params_mk, scale=scale) \
              .add_node("add_mask", inputs={"output": "input", "Mask": "mask_ptr"}, parents=[5, 3], **common_params_m) \
              .add_node("softmax_store", inputs={"output": "input"}, parents=[6], **common_params_m) \
              .add_node("gemm_pv", inputs={"output": "probs", "V": "v_ptr"}, parents=[7, 2], **common_params_loopk) \
              .add_node("sum_and_square", inputs={"output": "input"}, parents=[7], **common_params_m) \
              .add_output("attn_output", source=8, shape=(batch, heads, M, K), device=DEVICE) \
              .add_output("roco_score", source=9, shape=(batch, heads, N), output_index=0, device=DEVICE) \
              .add_output("roco_sq_score", source=9, shape=(batch, heads, N), output_index=1, device=DEVICE)
    
    # ========================================================================
    # 准备全局输入和参数字典
    # ========================================================================
    global_inputs = {
        'Q': Q,
        'K': Kmat,
        'V': V,
        'Mask': Mask
    }
    
    params_dict = {
        'M': M, 'N': N, 'K': K,
        'BM': BM, 'BN': BN,
        'batch': batch, 'heads': heads,
        'scale': scale
    }
    
    # ========================================================================
    # 自动搜索最优拆分
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
        max_splits=1,  # 允许最多2个切分点
        device=DEVICE
    )
    
    # 验证
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    
    fused_args_list, _, output_tensor, intermediate_tensors = \
        full_graph._build_fused_args_from_partition(best_partition, global_inputs, params_dict, DEVICE)
    
    Out_baseline = torch.empty(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    roco_score_baseline = torch.empty(batch, heads, N, device=DEVICE, dtype=torch.float16)
    roco_sq_score_baseline = torch.empty(batch, heads, N, device=DEVICE, dtype=torch.float16)
    softmax_sum_baseline = torch.empty(batch, heads, M, 1, device=DEVICE, dtype=torch.float32)
    
    def fused_launcher():
        for (compiled, grid), subgraph_args in zip(best_compiled, fused_args_list):
            compiled[grid](*subgraph_args)

    def torch_baseline_launcher():
        attn_out, roco_out, roco_sq_out, sm_out = torch_attention_ref(Q, Kmat, V, Mask, scale)
        Out_baseline.copy_(attn_out)
        roco_score_baseline.copy_(roco_out)
        roco_sq_score_baseline.copy_(roco_sq_out)
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
    
    # 验证 ROCO score output
    if "roco_score" in outputs:
        print("\nCheck ROCO Score Output (sum):")
        validate_correctness(
            torch_baseline_launcher,
            fused_launcher,
            roco_score_baseline,
            outputs["roco_score"],
            rtol=1e-1,
            atol=1e-1
        )
    else:
        print("\nroco_score not found")
        
    # 验证 ROCO squared score output
    if "roco_sq_score" in outputs:
        print("\nCheck ROCO Squared Score Output (sq):")
        validate_correctness(
            torch_baseline_launcher,
            fused_launcher,
            roco_sq_score_baseline,
            outputs["roco_sq_score"],
            rtol=1e-1,
            atol=1e-1
        )
    else:
        print("\nroco_sq_score not found")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
