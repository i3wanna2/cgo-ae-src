import math
import torch
import torch.nn.functional as F
from tilefusion.core.compute_graph import ComputeGraph
from tilefusion.utils.utils import DEVICE, benchmark_performance, validate_correctness


def torch_attention_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, 
                       Mask: torch.Tensor, scale: float, logit_softcapping: float) -> torch.Tensor:
    """PyTorch参考实现 - Gemma2 attention with logit softcapping"""
    scores = torch.matmul(Q, Kmat.transpose(-2, -1)) / scale
    scores = scores / logit_softcapping
    scores = torch.tanh(scores)
    scores = scores * logit_softcapping
    scores = scores + Mask
    probs = F.softmax(scores.float(), dim=-1)
    O = torch.matmul(probs.to(Q.dtype), V)
    return O


def main():
    # 问题规模
    batch, heads = 1, 32
    M, N, K = 4096, 4096, 128
    scale = math.sqrt(K)  # sqrt(head_dim)
    logit_softcapping = 50.0
    BM, BN = 64, 64
    num_warps, num_stages = 4, 3
    
    print("=" * 70)
    print("Gemma2 Attention - Auto Search Test")
    print("=" * 70)
    print(f"Problem: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")
    print(f"Scale: {scale}, Logit softcapping: {logit_softcapping}")
    print(f"Block: BM={BM}, BN={BN}, warps={num_warps}, stages={num_stages}")
    
    # 输入数据
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    
    # 创建因果掩码 (与Gemma2原始代码一致)
    causal_mask_bool = torch.triu(
        torch.ones(M, N, dtype=torch.bool, device=DEVICE),
        diagonal=N - M + 1
    )
    causal_mask_float = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
    
    # 构建计算图: GEMM -> Scale1 -> Scale2 -> Tanh -> Scale3 -> AddMask -> Softmax -> GEMM_loopk
    graph = ComputeGraph("gemma2_attention")
    graph.add_input("Q", "K", "V", "Mask")
    
    common_params_mk = {"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    common_params_m = {"M": M, "N": N, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    common_params_loopk = {"M": M, "N": K, "K": N, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    
    # 构建计算图节点
    graph.add_node("gemm_qk", inputs={"Q": "q_ptr", "K": "k_ptr"}, parents=[0, 1], **common_params_mk) \
         .add_node("scale_c", inputs={"scores": "input"}, parents=[4], **common_params_m, scale=1.0/scale) \
         .add_node("scale_c", inputs={"output": "input"}, parents=[5], **common_params_m, scale=1.0/logit_softcapping) \
         .add_node("tanh", inputs={"output": "input"}, parents=[6], **common_params_m) \
         .add_node("scale_c", inputs={"output": "input"}, parents=[7], **common_params_m, scale=logit_softcapping) \
         .add_node("add_mask", inputs={"output": "input", "Mask": "mask_ptr"}, parents=[8, 3], **common_params_m) \
         .add_node("softmax", inputs={"output": "input"}, parents=[9], **common_params_m) \
         .add_node("gemm_pv", inputs={"output": "probs", "V": "v_ptr"}, parents=[10, 2], **common_params_loopk) \
         .add_output("attn_output", source=11, shape=(batch, heads, M, K), device=DEVICE)
    
    # 自动搜索
    global_inputs = {'Q': Q, 'K': Kmat, 'V': V, 'Mask': Mask}
    params_dict = {
        'M': M, 'N': N, 'K': K, 
        'BM': BM, 'BN': BN, 
        'batch': batch, 'heads': heads, 
        'scale': scale,
        'logit_softcapping': logit_softcapping
    }
    
    print("\n" + "=" * 70)
    print("Starting Search...")
    print("=" * 70)
    
    best_compiled, best_partition, best_time = graph.compile(
        search_optimal=True,
        global_inputs=global_inputs,
        params_dict=params_dict,
        num_warps=num_warps,
        num_stages=num_stages,
        num_warmup=5,
        num_repeat=20,
        max_splits=1,
        device=DEVICE
    )
    
    # 验证
    print("\n" + "=" * 70)
    print("Validation")
    print("=" * 70)
    
    fused_args_list, _, output_tensor, intermediate_tensors = \
        graph._build_fused_args_from_partition(best_partition, global_inputs, params_dict, DEVICE)
    
    Out_baseline = torch.empty(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
 
    def fused_launcher():
        for (compiled, grid), subgraph_args in zip(best_compiled, fused_args_list):
            compiled[grid](*subgraph_args)

    def torch_baseline_launcher():
        Out_baseline.copy_(torch_attention_ref(Q, Kmat, V, Mask, scale, logit_softcapping))
    
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
    outputs = graph.outputs
    print(f"\nOutputs: {list(outputs.keys())}")
    
    if "attn_output" in outputs:
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
    
    print("\n" + "=" * 70)
    print("Gemma2 Demo Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
