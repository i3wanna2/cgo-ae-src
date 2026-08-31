import math
import torch

from tilefusion.core.compute_graph import ComputeGraph, ComputeNode
from tilefusion.utils.utils import DEVICE, benchmark_performance, validate_correctness
from dsa_mha_v2 import torch_dsa_attention_ref


def build_dsa_graph(
    name: str,
    *,
    batch: int,
    heads: int,
    M: int,
    N: int,
    K: int,
    BM: int,
    BN: int,
    scale: float,
    topk_k: int,
    scatter_block_m: int,
    scatter_block_k: int,
) -> ComputeGraph:
    """Construct a unified ComputeGraph covering indexer + MHA pipelines."""
    graph = ComputeGraph(name)
    graph.add_input("Q_idx", "K_idx", "W_idx", "Q", "K", "V", "Mask", "IndexMask")

    input_idx = {
        "Q_idx": 0,
        "K_idx": 1,
        "W_idx": 2,
        "Q": 3,
        "K": 4,
        "V": 5,
        "Mask": 6,
        "IndexMask": 7,
    }

    common_mk = {
        "M": M,
        "N": N,
        "K": K,
        "BM": BM,
        "BN": BN,
        "batch": batch,
        "heads": heads,
    }
    common_mn = {
        "M": M,
        "N": N,
        "BM": BM,
        "BN": BN,
        "batch": batch,
        "heads": heads,
    }

    # Indexer pipeline -----------------------------------------------------
    # Indexer GEMM expects tensors named 'Q' and 'K' in metadata
    graph.add_node(
        "gemm_qk",
        inputs={"Q_idx": "Q", "K_idx": "K"},
        parents=[input_idx["Q_idx"], input_idx["K_idx"]],
        **common_mk,
    )
    idx_gemm = len(graph.nodes) - 1

    graph.add_node(
        "relu",
        inputs={"scores": "input"},
        parents=[idx_gemm],
        **common_mn,
    )
    idx_relu = len(graph.nodes) - 1

    graph.add_node(
        "mul",
        inputs={"output": "x", "W_idx": "y"},
        parents=[idx_relu, input_idx["W_idx"]],
        **common_mn,
    )
    idx_mul = len(graph.nodes) - 1

    graph.add_node(
        "sum_dim1",
        inputs={"output": "input"},
        parents=[idx_mul],
        **common_mn,
        num_heads=heads,
    )
    idx_sum = len(graph.nodes) - 1

    graph.add_node(
        "broadcast_bmn_to_bhmn",
        inputs={"sum_out": "src"},
        parents=[idx_sum],
        **common_mn,
    )
    idx_broadcast = len(graph.nodes) - 1

    # add_mask metadata expects 'input' and 'mask' tensor names
    graph.add_node(
        "add_mask",
        inputs={"dst": "input", "Mask": "mask"},
        parents=[idx_broadcast, input_idx["Mask"]],
        **common_mn,
    )
    idx_index_add_mask = len(graph.nodes) - 1

    graph.add_node(
        "topk_bhmn",
        inputs={"output": "input"},
        parents=[idx_index_add_mask],
        M=M,
        N=N,
        K=topk_k,
        BM=BM,
        BN=BN,
        batch=batch,
        heads=heads,
    )
    idx_topk = len(graph.nodes) - 1

    graph.add_node(
        "scatter_mask",
        inputs={"topk_idx": "indices", "IndexMask": "mask_tensor", },
        parents=[idx_topk, input_idx["IndexMask"],],
        M=M,
        N=N,
        K=topk_k,
        BM=scatter_block_m,
        BN=scatter_block_k,
        batch=batch,
        heads=heads,
    )
    idx_scatter = len(graph.nodes) - 1

    # MHA pipeline ---------------------------------------------------------
    # MHA GEMM expects 'Q' and 'K'
    graph.add_node(
        "gemm_qk",
        inputs={"Q": "Q", "K": "K"},
        parents=[input_idx["Q"], input_idx["K"]],
        **common_mk,
    )
    mha_gemm = len(graph.nodes) - 1

    graph.add_node(
        "scale",
        inputs={"scores": "input"},
        parents=[mha_gemm],
        **common_mk,
        scale=scale,
    )
    mha_scale = len(graph.nodes) - 1

    # add_mask in MHA path: 'input' from scale, 'mask' from tilefusion.ops.scatter result
    graph.add_node(
        "add_mask",
        inputs={"output": "input", "mask_tensor": "mask"},
        parents=[mha_scale, idx_scatter],
        **common_mn,
    )
    mha_add_mask = len(graph.nodes) - 1

    graph.add_node(
        "softmax_store",
        inputs={"output": "input"},
        parents=[mha_add_mask],
        **common_mn,
    )
    softmax_node = len(graph.nodes) - 1

    graph.add_node(
        "gemm_pv",
        inputs={"output": "probs", "V": "v_ptr"},
        parents=[softmax_node, input_idx["V"]],
        M=M,
        N=K,
        K=N,
        BM=BM,
        BN=BN,
        batch=batch,
        heads=heads,
    )

    return graph


def main():
    batch = 1
    heads = 32
    M = 4096
    N = 4096
    K = 128
    BM = 64
    BN = 64
    scale = 1.0 / math.sqrt(K)

    topk_requested = 2
    topk_k = min(topk_requested, N)
    scatter_block_m = 32
    scatter_block_k = 32

    num_warps = 4
    num_stages = 3

    print("=" * 70)
    print("DSA Attention - Full Graph Auto Search")
    print("=" * 70)
    print(f"Problem: batch={batch}, heads={heads}, M={M}, N={N}, K={K}, TopK: k={topk_k}")

    torch.manual_seed(0)
    dtype = torch.float16

    Q_mha = torch.randn(batch, heads, M, K, device=DEVICE, dtype=dtype)
    K_mha = torch.randn(batch, heads, N, K, device=DEVICE, dtype=dtype)
    V_mha = torch.randn(batch, heads, N, K, device=DEVICE, dtype=dtype)

    Q_idx = torch.randn(batch, heads, M, K, device=DEVICE, dtype=dtype)
    K_idx = torch.randn(batch, heads, N, K, device=DEVICE, dtype=dtype)
    W_idx = torch.randn(batch, heads, M, N, device=DEVICE, dtype=dtype)

    causal_mask_bool = torch.triu(
        torch.ones(M, N, dtype=torch.bool, device=DEVICE),
        diagonal=1,
    )
    causal_mask = torch.zeros(M, N, dtype=dtype, device=DEVICE)
    causal_mask = causal_mask.masked_fill(causal_mask_bool, float("-inf"))
    Mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)

    index_mask_template = torch.full(
        (batch, heads, M, N),
        fill_value=-torch.inf,
        device=DEVICE,
        dtype=dtype,
    )
    IndexMask = torch.full(
        (batch, heads, M, N),
        fill_value=-torch.inf,
        device=DEVICE,
        dtype=dtype,
    )

    Out_baseline = torch.empty(batch, heads, M, K, device=DEVICE, dtype=dtype)
    out_indicis = torch.empty(batch, heads, M, topk_k, device=DEVICE, dtype=torch.int32)
    def torch_baseline_launcher():
        out, tmp, _ = torch_dsa_attention_ref(
            Q_mha,
            K_mha,
            V_mha,
            Q_idx,
            K_idx,
            W_idx,
            Mask,
            index_mask_template,
            scale,
            topk_k,
        )
        Out_baseline.copy_(out)
        out_indicis.copy_(tmp)

    graph = build_dsa_graph(
        "dsa_full",
        batch=batch,
        heads=heads,
        M=M,
        N=N,
        K=K,
        BM=BM,
        BN=BN,
        scale=scale,
        topk_k=topk_k,
        scatter_block_m=scatter_block_m,
        scatter_block_k=scatter_block_k,
    )

    global_inputs = {
        "Q_idx": Q_idx,
        "K_idx": K_idx,
        "W_idx": W_idx,
        "Q": Q_mha,
        "K": K_mha,
        "V": V_mha,
        "Mask": Mask,
        "IndexMask": IndexMask,
    }

    params_dict = {
        "M": M,
        "N": N,
        "K": K,
        "BM": BM,
        "BN": BN,
        "batch": batch,
        "heads": heads,
        "scale": scale,
        "topk_k": topk_k,
        "scatter_block_m": scatter_block_m,
        "scatter_block_k": scatter_block_k,
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
        num_repeat=10,
        max_splits=5,
        device=DEVICE,
        enable_causal_opt=False,
    )
    print(f"Best partition latency: {best_time:.3f} ms")

    fused_args_list, _, output_tensor, intermediate_tensors = graph._build_fused_args_from_partition(
        best_partition,
        global_inputs,
        params_dict,
        DEVICE,
    )

    def fused_launcher():
        IndexMask.copy_(index_mask_template)
        # idx = 0
        for (compiled, grid), subgraph_args in zip(best_compiled, fused_args_list):
            compiled[grid](*subgraph_args)
            # if idx == 3:
            #     scatter_fused_args = fused_args_list[4]
            #     scatter_fused_args[1] = fused_args_list[3][15]
            #     # break
            # idx+=1
            

    # ========================================================================
    # 按子图检查输出 - 只检查子图边界的输出
    # ========================================================================
    print("\n" + "=" * 70)
    print("Subgraph-Level Debugging")
    print("=" * 70)
    
    print(f"\n📋 Partition structure: {len(best_partition)} subgraphs")
    for sg_idx, subgraph in enumerate(best_partition):
        compute_nodes = [n for n in subgraph if isinstance(n, ComputeNode)]
        if compute_nodes:
            node_names = [n.kernel_name for n in compute_nodes]
            print(f"  Subgraph {sg_idx}: {' -> '.join(node_names)}")
    
    print("\n🔄 运行baseline...")
    torch_baseline_launcher()
    # scatter_fused_args = fused_args_list[4]
    # scatter_fused_args[1] = out_indicis  # 替换 scatter 的 indices 输入为 baseline 结果
    print("🔄 运行fused...")
    IndexMask.copy_(index_mask_template)
    fused_launcher()
    
    # 找到每个子图的最后一个节点并检查输出
    print("\n" + "=" * 70)
    print("Checking Subgraph Output Boundaries")
    print("=" * 70)
    
    print(f"\n📦 所有中间tensor keys:")
    for key in sorted(intermediate_tensors.keys()):
        val = intermediate_tensors[key]
        if torch.is_tensor(val):
            print(f"   {key}: {val.shape}")
        else:
            print(f"   {key}: <{type(val).__name__}>")
    
    # 手动计算baseline的每个子图输出
    # Subgraph 0: gemm_qk -> relu -> mul (indexer前3步)
    scores_idx = Q_idx @ K_idx.transpose(-2, -1)
    relu_out = torch.nn.functional.relu(scores_idx)
    mul_out = relu_out * W_idx
    
    print(f"\n✅ Subgraph 0 output (mul): {mul_out.shape}")
    # 查找fused的mul输出 - 搜索所有包含output且形状匹配的
    sg0_candidates = [(k, v) for k, v in intermediate_tensors.items() 
                      if 'output' in k and v.shape == mul_out.shape and v.dim() == 4]
    print(f"   Candidates with shape {mul_out.shape}: {[k for k, v in sg0_candidates]}")
    
    # **关键调试**：打印W_idx看是否被正确使用
    print(f"\n   🔍 Debugging mul inputs:")
    print(f"      relu_out: min={relu_out.min().item():.6f}, max={relu_out.max().item():.6f}, mean={relu_out.mean().item():.6f}")
    print(f"      W_idx: min={W_idx.min().item():.6f}, max={W_idx.max().item():.6f}, mean={W_idx.mean().item():.6f}")
    print(f"      mul_out (baseline): min={mul_out.min().item():.6f}, max={mul_out.max().item():.6f}, mean={mul_out.mean().item():.6f}")
    
    if sg0_candidates:
        for key, fused_mul in sg0_candidates:
            max_diff = torch.abs(mul_out - fused_mul).max().item()
            print(f"      {key} (fused): min={fused_mul.min().item():.6f}, max={fused_mul.max().item():.6f}, mean={fused_mul.mean().item():.6f}, diff={max_diff:.6e}")
    
    # Subgraph 1: sum_dim1
    sum_out = mul_out.sum(dim=1, keepdim=False)
    print(f"\n✅ Subgraph 1 output (sum_dim1): {sum_out.shape}")
    sg1_candidates = [(k, v) for k, v in intermediate_tensors.items() 
                      if 'sum_out' in k and v.shape == sum_out.shape]
    print(f"   Candidates with shape {sum_out.shape}: {[k for k, v in sg1_candidates]}")
    if sg1_candidates:
        key, fused_sum = sg1_candidates[0]
        max_diff = torch.abs(sum_out - fused_sum).max().item()
        print(f"   Using {key}: Max diff {max_diff:.6e} {'❌' if max_diff > 1e-1 else '✅'}")
    
    # Subgraph 2: 分区可能仅包含 broadcast，也可能包含 broadcast+add_mask（如果策略不同）
    sg2_nodes = [n for n in best_partition[2] if isinstance(n, ComputeNode)] if len(best_partition) > 2 else []
    if sg2_nodes:
        sg2_names = [n.kernel_name for n in sg2_nodes]
    else:
        sg2_names = []

    broadcast_out = sum_out.unsqueeze(1).expand(batch, heads, M, N)
    if len(sg2_nodes) == 1 and sg2_nodes[0].kernel_name == 'broadcast_bmn_to_bhmn':
        # 子图2只有 broadcast，期望输出是 broadcast 的 dst
        expected_sg2 = broadcast_out
        expected_label = 'broadcast'
        # broadcast 输出张量命名为 <id>_dst
        sg2_candidates = [(k, v) for k, v in intermediate_tensors.items()
                          if k.endswith('_dst') and v.shape == expected_sg2.shape]
        # 为后续 topk / scatter 基线仍需一个 add_mask_idx_out（indexer路径语义：广播后再加 Mask）
        add_mask_idx_out = broadcast_out + Mask  # 不参与本子图 diff，只供后续使用
    else:
        # 默认认为包含了 add_mask（旧逻辑）
        add_mask_idx_out = broadcast_out + Mask
        expected_sg2 = add_mask_idx_out
        expected_label = 'add_mask indexer'
        sg2_candidates = [(k, v) for k, v in intermediate_tensors.items()
                          if 'output' in k and v.shape == expected_sg2.shape and v.dim() == 4]

    print(f"\n✅ Subgraph 2 output ({expected_label}): {expected_sg2.shape}")
    print(f"   Subgraph 2 actual nodes: {sg2_names}")
    print(f"   Candidates with shape {expected_sg2.shape}: {[k for k, v in sg2_candidates]}")
    if sg2_candidates:
        # 优先选与期望标签匹配度高的：如果是 broadcast 选 dst；如果是 add_mask 选第一个 output
        key, fused_val = sg2_candidates[0]
        max_diff = torch.abs(expected_sg2 - fused_val).max().item()
        print(f"   Using {key}: Max diff {max_diff:.6e} {'❌' if max_diff > 1e-1 else '✅'}")
    
    # Subgraph 3: (add_mask + topk) fused -> boundary = topk outputs ONLY
    # Baseline: compute topk over add_mask_idx_out (do NOT apply scatter yet)
    res = expected_sg2 + Mask
    topk_vals_baseline, topk_idx_baseline = res.topk(topk_k, dim=-1)
    print(f"\n✅ Subgraph 3 boundary output (topk): vals={topk_vals_baseline.shape}, idx={topk_idx_baseline.shape}")
    # Extra debug: inspect fused args list for this subgraph to ensure inputs built correctly
    try:
        # Heuristic: partition index 3 corresponds to (add_mask+topk) or just topk depending on search.
        # Find subgraph whose last node kernel_name contains 'topk'.
        topk_sg_idx = None
        for idx_sg, sg in enumerate(best_partition):
            nodes = [n for n in sg if isinstance(n, ComputeNode)]
            if nodes and nodes[-1].kernel_name.startswith('topk'):
                topk_sg_idx = idx_sg
                break
        if topk_sg_idx is not None and topk_sg_idx < len(fused_args_list):
            fused_topk_args = fused_args_list[topk_sg_idx]
            print(f"   [DEBUG] Fused Subgraph {topk_sg_idx} arg count: {len(fused_topk_args)}")
            for ai, av in enumerate(fused_topk_args):
                if hasattr(av, 'shape'):
                    print(f"      arg[{ai}] tensor shape={tuple(av.shape)} stride={tuple(av.stride())} dtype={av.dtype} ptr={av.data_ptr()}")
                else:
                    print(f"      arg[{ai}] scalar={av}")
            # Try to locate candidate tensor matching baseline add_mask_idx_out shape as input to topk
            candidates = [av for av in fused_topk_args if hasattr(av,'shape') and tuple(av.shape)==tuple(add_mask_idx_out.shape)]
            if candidates:
                ptrs = [c.data_ptr() for c in candidates]
                print(f"   [DEBUG] candidate add_mask output ptrs in fused args: {ptrs}")
                print(f"   [DEBUG] baseline add_mask_idx_out ptr (temp, not stored) cannot compare directly; using value stats")
                for c in candidates:
                    print(f"         cand ptr={c.data_ptr()} min={c.min().item():.3f} max={c.max().item():.3f}")
    except Exception as e:
        print(f"   [DEBUG] Failed to inspect fused topk args: {e}")
    # Fused candidates: look for *_topk_vals / *_topk_idx tensors
    fused_topk_vals = [(k, v) for k, v in intermediate_tensors.items() if k.endswith('_topk_vals') and v.shape == topk_vals_baseline.shape]
    fused_topk_idx = [(k, v) for k, v in intermediate_tensors.items() if k.endswith('_topk_idx') and v.shape == topk_idx_baseline.shape]
    if fused_topk_vals:
        k_vals, v_vals = fused_topk_vals[0]
        # diff_vals = torch.abs(topk_vals_baseline - v_vals).max().item()
        index_scores_masked = torch.zeros(batch, heads, M, N, device=DEVICE, dtype=torch.float16)
        torch.testing.assert_close(expected_sg2, fused_topk_args[0], rtol=1e-1, atol=1e-1, equal_nan=True)
        torch.testing.assert_close(Mask, fused_topk_args[1], rtol=1e-1, atol=1e-1, equal_nan=True)
        torch.testing.assert_close(expected_sg2 + Mask, fused_topk_args[2], rtol=1e-1, atol=1e-1, equal_nan=True)
        # torch.testing.assert_close(topk_vals_baseline, fused_topk_args[14], rtol=1e-1, atol=1e-1, equal_nan=True)
        # torch.testing.assert_close(topk_idx_baseline, fused_topk_args[15], rtol=1e-1, atol=1e-1, equal_nan=True, check_dtype=False)
        # print(fused_topk_args[14])
        # print(fused_topk_args[15])
        # print(topk_vals_baseline)
        # print(topk_idx_baseline)
        # print(f"   Using {k_vals}: Max diff (vals) {diff_vals:.6e} {'❌' if diff_vals > 1e-1 else '✅'}")
        # Value distribution comparison
        print(f"   Baseline topk vals stats: min={topk_vals_baseline.min().item():.3f} max={topk_vals_baseline.max().item():.3f} mean={topk_vals_baseline.mean().item():.3f}")
        print(f"   Fused    topk vals stats: min={v_vals.min().item():.3f} max={v_vals.max().item():.3f} mean={v_vals.mean().item():.3f}")
        # Sample row 0 head 0: compare sorted values
        try:
            b_row = topk_vals_baseline[0,0,0].float().cpu().numpy()
            f_row = v_vals[0,0,0].float().cpu().numpy()
            print(f"   Sample head0 row0 baseline vals: {b_row.tolist()}")
            print(f"   Sample head0 row0 fused    vals: {f_row.tolist()}")
        except Exception:
            pass
    else:
        print("   ❌ No fused topk_vals tensor found")

    # Subgraph 4: scatter_mask (applies indices). Baseline scatter now.
    # index_mask_baseline = index_mask_template.clone()
    # index_mask_baseline.scatter_(dim=-1, index=topk_idx_baseline.to(torch.int64), value=0.0)
    mtp_mask = torch.full((batch, heads, M, N), fill_value=-torch.inf, device=DEVICE, dtype=torch.float16)
    index_mask_baseline = torch.scatter(mtp_mask, -1, topk_idx_baseline, 0.0)
    print(f"\n✅ Subgraph 4 boundary output (scatter_mask): {index_mask_baseline.shape}")
    scatter_candidates = [(k, v) for k, v in intermediate_tensors.items() if k.endswith('_mask_tensor') and v.shape == index_mask_baseline.shape]
    if scatter_candidates:
        fused_key, fused_mask_tensor = scatter_candidates[0]
        max_diff_scatter = torch.abs(index_mask_baseline - fused_mask_tensor).max().item()
        print(f"   Using {fused_key}: Max diff {max_diff_scatter:.6e} {'❌' if max_diff_scatter > 1e-2 else '✅'}")
    else:
        print("   ❌ No fused scatter mask_tensor found")
    scatter_fused_args = fused_args_list[4]
    for ai, av in enumerate(scatter_fused_args):
        if hasattr(av, 'shape'):
            print(f"      arg[{ai}] tensor shape={tuple(av.shape)} stride={tuple(av.stride())} dtype={av.dtype} ptr={av.data_ptr()}")
        else:
            print(f"      arg[{ai}] scalar={av}")
    # torch.testing.assert_close(index_mask_template, scatter_fused_args[1], rtol=1e-1, atol=1e-1, equal_nan=True)
    torch.testing.assert_close(index_mask_baseline, scatter_fused_args[0], rtol=1e-1, atol=1e-1, equal_nan=True)
    # 最终子图（MHA fused path）只验证边界输出：gemm_pv，不检查中间 add_mask / softmax
    # 基线重建整条 MHA 流程
    scores_mha = Q_mha @ K_mha.transpose(-2, -1)
    scaled = scores_mha * scale
    masked = scaled + index_mask_baseline  # 使用 scatter 后的稀疏 IndexMask
    probs = torch.softmax(masked, dim=-1)
    output_baseline = probs @ V_mha
    last_sg_idx = len(best_partition) - 1
    print(f"\n✅ Subgraph {last_sg_idx} boundary output (gemm_pv): {output_baseline.shape}")
    print(f"   Fused output_tensor: {output_tensor.shape}")
    max_diff = torch.abs(output_baseline - output_tensor).max().item()
    print(f"   Max diff: {max_diff:.6e} {'❌' if max_diff > 1e-1 else '✅'}")

    # -------------------------------------------------------------
    # 深度调试：MHA 子图内部指针与数值链路
    # -------------------------------------------------------------
    mha_subgraph = best_partition[-1]
    mha_nodes = [n for n in mha_subgraph if isinstance(n, ComputeNode)]
    print("\n🔍 MHA子图阶段链路调试")
    # 收集期望baseline阶段结果
    baseline_scores = Q_mha @ K_mha.transpose(-2, -1)
    baseline_scaled = baseline_scores * scale
    baseline_masked = baseline_scaled + IndexMask  # IndexMask 已经被 scatter 更新
    baseline_softmax = torch.softmax(baseline_masked, dim=-1)
    baseline_final = baseline_softmax @ V_mha
    print(f"   baseline_scores: min={baseline_scores.min().item():.3f} max={baseline_scores.max().item():.3f}")
    print(f"   baseline_scaled: min={baseline_scaled.min().item():.3f} max={baseline_scaled.max().item():.3f}")
    print(f"   baseline_masked: min={baseline_masked.min().item():.3f} max={baseline_masked.max().item():.3f}")
    print(f"   baseline_softmax: min={baseline_softmax.min().item():.3f} max={baseline_softmax.max().item():.3f}")
    print(f"   baseline_final: min={baseline_final.min().item():.3f} max={baseline_final.max().item():.3f}")

    # 建立 id -> 输出tensor key 映射，按节点顺序打印
    for node in mha_nodes:
        mid = node.metadata.id
        outs = [t for t in node.metadata.tensors if t.role in ('output','input_output')]
        for t in outs:
            key = f"{mid}_{t.name}"
            if key in intermediate_tensors and torch.is_tensor(intermediate_tensors[key]):
                ten = intermediate_tensors[key]
                print(f"   [{node.kernel_name}] {key}: ptr={ten.data_ptr()} min={ten.min().item():.3f} max={ten.max().item():.3f}")

    # 关键映射核查：scale 输入是否指向 gemm scores；MHA add_mask 输入是否指向 scale 输出；softmax 输入是否指向 add_mask 输出
    def ptr(t):
        return t.data_ptr() if torch.is_tensor(t) else None
    # 查找实际被选作 scale 输入的 tensor（input 参数名）
    scale_node = next((n for n in mha_nodes if n.kernel_name == 'scale'), None)
    gemm_node = next((n for n in mha_nodes if n.kernel_name == 'gemm_qk'), None)
    add_mask_node = next((n for n in mha_nodes if n.kernel_name == 'add_mask'), None)
    softmax_node = next((n for n in mha_nodes if n.kernel_name == 'softmax_store'), None)

    def find_tensor(node, tensor_name):
        if not node: return None
        key = f"{node.metadata.id}_{tensor_name}"
        return intermediate_tensors.get(key)

    gemm_scores = find_tensor(gemm_node, 'scores')
    scale_input = find_tensor(scale_node, 'input')
    scale_output = find_tensor(scale_node, 'output')
    add_mask_input = find_tensor(add_mask_node, 'input')
    add_mask_mask = find_tensor(add_mask_node, 'mask')
    add_mask_output = find_tensor(add_mask_node, 'output')
    softmax_input = find_tensor(softmax_node, 'input')

    print("\n   指针匹配:")
    print(f"      scale.input == gemm.scores ? {ptr(scale_input)==ptr(gemm_scores)}")
    print(f"      add_mask.input == scale.output ? {ptr(add_mask_input)==ptr(scale_output)}")
    print(f"      softmax.input == add_mask.output ? {ptr(softmax_input)==ptr(add_mask_output)}")
    print(f"      add_mask.mask == IndexMask ? {ptr(add_mask_mask)==ptr(IndexMask)}")

    # 若发现不匹配，输出建议
    if scale_input is not None and gemm_scores is not None and ptr(scale_input)!=ptr(gemm_scores):
        print("   ⚠ scale 输入未指向 GEMM scores，可能被影子缓冲覆盖")
    if add_mask_input is not None and scale_output is not None and ptr(add_mask_input)!=ptr(scale_output):
        print("   ⚠ add_mask 输入未指向 scale 输出，存在指针错配")
    if softmax_input is not None and add_mask_output is not None and ptr(softmax_input)!=ptr(add_mask_output):
        print("   ⚠ softmax 输入未指向 add_mask 输出，存在指针错配")
    if add_mask_mask is not None and ptr(add_mask_mask)!=ptr(IndexMask):
        print("   ⚠ add_mask 的 mask 参数未指向 IndexMask，稀疏掩码未生效")


    # print("\n" + "=" * 70)
    # print("Performance Benchmark")
    # print("=" * 70)

    benchmark_performance(
        torch_baseline_launcher,
        fused_launcher,
    )

    print("\n" + "=" * 70)
    print("Correctness Validation")
    print("=" * 70)

    validate_correctness(
        torch_baseline_launcher,
        fused_launcher,
        (Out_baseline,),
        (output_tensor,),
        rtol=1e-1,
        atol=1e-1,
    )

    print("\n" + "=" * 70)
    print("DSA v3 Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
