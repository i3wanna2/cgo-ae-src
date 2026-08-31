import math
import torch
import torch.nn.functional as F
from tilefusion.core.compute_graph import ComputeGraph
from tilefusion.utils.utils import DEVICE, benchmark_performance, validate_correctness
# from flash_attn.flash_attn_interface import flash_attn_func

def torch_attention_ref(Q, Kmat, V, Mask, scale):
    """PyTorch参考实现"""
    scores = Q @ Kmat.transpose(-2, -1)
    scores = scores * scale
    scores = scores + Mask
    probs = torch.nn.functional.softmax(scores, dim=-1)
    O = probs @ V
    return O

def torch_attention_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, 
                       Mask: torch.Tensor, scale: float) -> torch.Tensor:
    """PyTorch参考实现"""
    scores = Q @ Kmat.transpose(-2, -1)
    scores = scores * scale
    scores = scores + Mask
    probs = torch.softmax(scores, dim=-1)
    O = probs @ V
    return O


def sdpa_torch_attention_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, 
                             Mask: torch.Tensor, scale: float) -> torch.Tensor:
    """SDPA参考实现"""
    return F.scaled_dot_product_attention(Q, Kmat, V, attn_mask=Mask, scale=scale)


# def flash_attn_ref(Q: torch.Tensor, Kmat: torch.Tensor, V: torch.Tensor, 
#                   Mask: torch.Tensor, scale: float) -> torch.Tensor:
#     """Flash Attention参考实现"""
#     out = flash_attn_func(
#         Q.transpose(1, 2),
#         Kmat.transpose(1, 2),
#         V.transpose(1, 2),
#         causal=True,
#         softmax_scale=scale
#     ).transpose(1, 2).contiguous()
#     return out


def main():
    # 问题规模
    batch, heads = 1, 32
    M, N, K = 4096, 4096, 128
    scale = 1.0 / math.sqrt(K)
    BM, BN = 32, 128
    num_warps, num_stages = 4, 3
    
    print("=" * 70)
    print("H2O Attention - Simplified Auto Search Test")
    print("=" * 70)
    print(f"Problem: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")
    print(f"Block: BM={BM}, BN={BN}, warps={num_warps}, stages={num_stages}")
    
    # 输入数据
    Q = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
    Kmat = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    V = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
    causal_mask_bool = torch.triu(
        torch.ones(M, N, dtype=torch.bool, device=DEVICE),
        diagonal=1
    )
    causal_mask_float = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)
    
    # 构建简化的计算图（不包含sum_h2o）
    graph = ComputeGraph("attention")
    graph.add_input("Q", "K", "V", "Mask")
    
    common_params_mk = {"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    common_params_m = {"M": M, "N": N, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    common_params_loopk = {"M": M, "N": K, "K": N, "BM": BM, "BN": BN, "batch": batch, "heads": heads}
    
    graph.add_node("gemm_qk", inputs={"Q": "q_ptr", "K": "k_ptr"}, parents=[0, 1], **common_params_mk) \
         .add_node("scale", inputs={"scores": "input"}, parents=[4], **common_params_mk, scale=scale) \
         .add_node("add_mask", inputs={"output": "input", "Mask": "mask_ptr"}, parents=[5, 3], **common_params_m) \
         .add_node("softmax", inputs={"output": "input"}, parents=[6], **common_params_m) \
         .add_node("gemm_pv", inputs={"output": "probs", "V": "v_ptr"}, parents=[7, 2], **common_params_loopk) \
         .add_output("attn_output", source=8, shape=(batch, heads, M, K))
    
    # 自动搜索
    global_inputs = {'Q': Q, 'K': Kmat, 'V': V, 'Mask': Mask}
    params_dict = {'M': M, 'N': N, 'K': K, 'BM': BM, 'BN': BN, 'batch': batch, 'heads': heads, 'scale': scale}
    
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

    # def flash_attn_launcher():
    #     Out_baseline.copy_(flash_attn_ref(Q, Kmat, V, Mask, scale))

    def torch_baseline_launcher():
        Out_baseline.copy_(torch_attention_ref(Q, Kmat, V, Mask, scale))
    
    def sdpa_launcher():
        Out_baseline.copy_(sdpa_torch_attention_ref(Q, Kmat, V, Mask, scale))
    

    print("\n" + "=" * 70)
    print("Performance Benchmark")
    print("=" * 70)
    
    benchmark_performance(
        torch_baseline_launcher,
        sdpa_launcher, 
        # flash_attn_launcher,
        fused_launcher
    )
    
    print("\n" + "=" * 70)
    print("Correctness Validation")
    print("=" * 70)
    
    # 使用 get_output_tensors 直接获取输出
    outputs = graph.get_output_tensors(intermediate_tensors)
    attn_output = outputs.get("attn_output", output_tensor)
    
    validate_correctness(
        torch_baseline_launcher,
        fused_launcher,
        Out_baseline,
        attn_output,
        rtol=1e-3,
        atol=1e-2
    )
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
