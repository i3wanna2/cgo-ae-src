"""
Keyformer Attention - 自动搜索最优拆分策略

使用ComputeGraph的search_optimal_partition功能：
1. 构建完整的Keyformer计算图
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


def torch_keyformer_ref(Q, Kmat, V, Mask, exp_rand, scale, tau):
    """PyTorch参考实现（含Keyformer score）"""
    # Standard Attention
    scores = Q @ Kmat.transpose(-2, -1)
    scores = scores * scale
    scores = scores + Mask
    
    probs = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
    O = probs @ V
    
    # Keyformer Score
    gumbels = -torch.log(exp_rand)
    kf_score_scale = (scores.float() + gumbels) / tau
    kf_score_sm = torch.softmax(kf_score_scale, dim=-1)
    kf_score = kf_score_sm.sum(dim=2)  # [batch, heads, M, N] -> [batch, heads, N]
    
    return O, kf_score, kf_score_sm


def main():
    # 问题规模
    batch = 1
    heads = 32
    M = 1024
    N = 1024
    K = 128
    scale = 1.0 / math.sqrt(K)
    tau = 1.0
    
    # Block大小
    BM = 64
    BN = 64
    
    # 性能参数
    num_warps = 4
    num_stages = 3
    
    print("=" * 70)
    print("Keyformer Attention - Auto Search Optimal Partition")
    print("=" * 70)
    print(f"Problem size: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")
    print(f"Block size: BM={BM}, BN={BN}")
    print(f"Keyformer tau: {tau}")
    print(f"Performance params: num_warps={num_warps}, num_stages={num_stages}")
    
    # 准备输入数据
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    exp_rand = 1 + torch.randn(batch, heads, M, N, dtype=torch.float16, device=DEVICE).abs()
    
    # Causal mask
    causal_mask_bool = torch.triu(
        torch.ones(M, N, dtype=torch.bool, device=DEVICE),
        diagonal=1
    )
    causal_mask_float = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
    
    # ========================================================================
    # 构建完整的Keyformer计算图
    # ========================================================================
    print("\n" + "=" * 70)
    print("Building Full Keyformer Compute Graph")
    print("=" * 70)
    
    # Pipeline 1: 标准Attention（M-grid）
    # gemm_qk -> scale -> add_mask -> softmax_store -> gemm_pv
    
    # Pipeline 2: Keyformer Score（N-grid）
    # gemm_qk -> scale -> add_mask -> log_neg (gumbel) -> add -> scale (1/tau) -> softmax_recompute -> sum
    
    # 创建完整计算图
    full_graph = ComputeGraph("keyformer_full_graph")
    
    # 添加输入节点
    full_graph.add_input("Q", "K", "V", "Mask", "ExpRand")
    
    common_params_mk = {"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    common_params_m = {"M": M, "N": N, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    common_params_loopk = {"M": M, "N": K, "K": N, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    
    # 节点索引: 0=Q, 1=K, 2=V, 3=Mask, 4=ExpRand
    # Pipeline 1: 5=gemm_qk, 6=scale, 7=add_mask, 8=softmax_store, 9=gemm_pv
    # Pipeline 2: 10=log_neg (gumbel), 11=add (scores+gumbel), 12=scale_c (1/tau), 13=softmax_recompute, 14=sum
    
    # Pipeline 1: 标准Attention（M-grid）
    full_graph.add_node("gemm_qk", inputs={"Q": "q_ptr", "K": "k_ptr"}, parents=[0, 1], **common_params_mk) \
              .add_node("scale_c", inputs={"scores": "input"}, parents=[5], **common_params_mk, scale=scale) \
              .add_node("add_mask", inputs={"output": "input", "Mask": "mask_ptr"}, parents=[6, 3], **common_params_m) \
              .add_node("softmax", inputs={"output": "input"}, parents=[7], **common_params_m) \
              .add_node("gemm_pv", inputs={"output": "probs", "V": "v_ptr"}, parents=[8, 2], **common_params_loopk)

    # Pipeline 2: Keyformer Score（N-grid）
    # 复用前面的 gemm_qk (5) -> scale (6) -> add_mask (7)
    full_graph.add_node("log_neg", inputs={"ExpRand": "input"}, parents=[4], **common_params_m) \
              .add_node("add_mask", inputs={"output": "input", "output": "mask_ptr"}, parents=[7, 10], **common_params_m) \
              .add_node("scale_c", inputs={"output": "input"}, parents=[11], **common_params_m, scale=1.0/tau) \
              .add_node("softmax", inputs={"output": "input"}, parents=[12], **common_params_m) \
              .add_node("sum_h2o", inputs={"output": "input"}, parents=[13], **common_params_m) \
              .add_output("attn_output", source=9, shape=(batch, heads, M, K)) \
              .add_output("kf_score", source=14, shape=(batch, heads, N))

    # 准备全局输入和参数字典
    global_inputs = {
        'Q': Q,
        'K': Kmat,
        'V': V,
        'Mask': Mask,
        'ExpRand': exp_rand
    }
    
    params_dict = {
        'M': M, 'N': N, 'K': K,
        'BM': BM, 'BN': BN,
        'batch': batch, 'heads': heads,
        'scale': scale,
        'tau': tau
    }
    
    # ========================================================================
    # 自动搜索最优拆分
    # ========================================================================
    print("\n" + "=" * 70)
    print("Starting Automatic Partition Search with Auto Arg Construction")
    print("=" * 70)
    test_configs = [{'BM': 64, 'BN': 64, 'num_warps': 4, 'num_stages': 3}]
        
    best_compiled, best_partition, best_time = full_graph.compile(
        num_warps=num_warps,
        num_stages=num_stages,
        search_optimal=False,
        global_inputs=global_inputs,
        params_dict=params_dict,
        num_warmup=10,
        num_repeat=50,
        max_splits=2,  # 允许最多2个切分点（3个子图）
        device=DEVICE,
        configs=test_configs
    )
    
    # 验证
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    print("Clearing CUDA cache before validation...")
    
    fused_args_list, _, output_tensor, intermediate_tensors = \
        full_graph._build_fused_args_from_partition(best_partition, global_inputs, params_dict, DEVICE)
    
    Out_baseline = torch.empty(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    kf_score_baseline = torch.empty(batch, heads, N, device=DEVICE, dtype=torch.float16)
    kf_score_sm_baseline = torch.empty(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
    
    def fused_launcher():
        for (compiled, grid), subgraph_args in zip(best_compiled, fused_args_list):
            compiled[grid](*subgraph_args)

    def torch_baseline_launcher():
        attn_out, kf_out, kf_sm_out = torch_keyformer_ref(Q, Kmat, V, Mask, exp_rand, scale, tau)
        Out_baseline.copy_(attn_out)
        kf_score_baseline.copy_(kf_out)
        kf_score_sm_baseline.copy_(kf_sm_out)

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
    
    # 使用 get_output_tensors 直接获取输出
    outputs = full_graph.get_output_tensors(intermediate_tensors)
    print(f"\n获取到的输出: {list(outputs.keys())}")
    
    # 验证 attention output
    if "attn_output" in outputs:
        print("\n验证 Attention Output (gemm_pv):")
        validate_correctness(
            torch_baseline_launcher,
            fused_launcher,
            Out_baseline,
            outputs["attn_output"],
            rtol=1e-3,
            atol=1e-2
        )
    else:
        print("\n⚠️  未找到 attn_output")
    
    # 验证 Keyformer score output
    if "kf_score" in outputs:
        print("\n验证 Keyformer Score Output (sum_h2o):")
        validate_correctness(
            torch_baseline_launcher,
            fused_launcher,
            kf_score_baseline,
            outputs["kf_score"],
            rtol=1e-1,
            atol=1e-1
        )
    else:
        print("\n⚠️  未找到 kf_score")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
