import math
import torch
import torch.nn.functional as F

from tilefusion.core.compute_graph import ComputeGraph
from tilefusion.utils.utils import DEVICE, benchmark_performance, validate_correctness


def torch_dsa_attention_ref(
	Q_mha: torch.Tensor,
	K_mha: torch.Tensor,
	V_mha: torch.Tensor,
	Q_idx: torch.Tensor,
	K_idx: torch.Tensor,
	W_idx: torch.Tensor,
	Mask: torch.Tensor,
	index_mask: torch.Tensor,
	mha_scale: float,
	index_topk: int,
):
	B, H_mha, M, _ = Q_mha.shape
	H_idx = Q_idx.shape[1]
	N = K_mha.shape[2]

	# Indexer path
	dots = Q_idx @ K_idx.transpose(-2, -1)
	activated = F.relu(dots)
	weighted = activated * W_idx
	index_scores = weighted.sum(dim=1)
	index_scores = index_scores.unsqueeze(1).expand(B, H_mha, M, N).contiguous()
	index_scores_masked = index_scores + Mask

	k = min(index_topk, index_scores_masked.size(-1))
	topk_indices = index_scores_masked.topk(k, dim=-1)[1]
	index_mask_scatter = torch.scatter(index_mask, -1, topk_indices, 0.0)

	# MHA path
	mha_scores = Q_mha @ K_mha.transpose(-2, -1)
	mha_scores = mha_scores * mha_scale
	total_scores = mha_scores + index_mask_scatter
	probs = torch.softmax(total_scores, dim=-1)
	O = probs @ V_mha

	return O, topk_indices, index_mask_scatter


def build_mha_graph(name, M, N, K, BM, BN, batch, heads, scale):
	graph = ComputeGraph(name)
	graph.add_input("Q", "K", "V", "Mask")

	common_params_mk = {
		"M": M,
		"N": N,
		"K": K,
		"BM": BM,
		"BN": BN,
		"batch": batch,
		"heads": heads,
	}
	common_params_m = {
		"M": M,
		"N": N,
		"BM": BM,
		"BN": BN,
		"batch": batch,
		"heads": heads,
	}
	common_params_loopk = {
		"M": M,
		"N": K,
		"K": N,
		"BM": BM,
		"BN": BN,
		"batch": batch,
		"heads": heads,
	}

	graph.add_node("gemm_qk", inputs={"Q": "q_ptr", "K": "k_ptr"}, parents=[0, 1], **common_params_mk) \
		 .add_node("scale", inputs={"scores": "input"}, parents=[4], **common_params_mk, scale=scale) \
		 .add_node("add_mask", inputs={"output": "input", "Mask": "mask_ptr"}, parents=[5, 3], **common_params_m) \
		 .add_node("softmax_store", inputs={"output": "input"}, parents=[6], **common_params_m) \
		 .add_node("gemm_pv", inputs={"output": "probs", "V": "v_ptr"}, parents=[7, 2], **common_params_loopk)

	return graph


def main():
	batch = 1
	heads = 32
	M = 4096
	N = 4096
	K = 128
	scale = 1.0 / math.sqrt(K)

	BM = 64
	BN = 64
	num_warps = 4
	num_stages = 3

	print("=" * 70)
	print("DSA Attention - Auto Search MHA")
	print("=" * 70)
	print(f"Problem: batch={batch}, heads={heads}, M={M}, N={N}, K={K}")
	print(f"Block: BM={BM}, BN={BN}, warps={num_warps}, stages={num_stages}")

	# Inputs for MHA
	Q_mha = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
	K_mha = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
	V_mha = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)

	# Indexer inputs (can use same heads for now)
	Q_idx = torch.randn(batch, heads, M, K, device=DEVICE, dtype=torch.float16)
	K_idx = torch.randn(batch, heads, N, K, device=DEVICE, dtype=torch.float16)
	W_idx = torch.randn(batch, heads, M, N, device=DEVICE, dtype=torch.float16)

	# Causal mask
	causal_mask_bool = torch.triu(
		torch.ones(M, N, dtype=torch.bool, device=DEVICE),
		diagonal=1,
	)
	causal_mask_float = torch.zeros(M, N, dtype=torch.float16, device=DEVICE)
	causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float("-inf"))
	Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).repeat(batch, heads, 1, 1)

	index_mask = torch.full(
		(batch, heads, M, N),
		fill_value=-torch.inf,
		device=DEVICE,
		dtype=torch.float16,
	)
	index_topk = 2

	# Baseline output
	Out_baseline = torch.empty(batch, heads, M, K, device=DEVICE, dtype=torch.float16)

	def torch_baseline_launcher():
		out, _, _ = torch_dsa_attention_ref(
			Q_mha,
			K_mha,
			V_mha,
			Q_idx,
			K_idx,
			W_idx,
			Mask,
			index_mask,
			scale,
			index_topk,
		)
		Out_baseline.copy_(out)

	# Indexer-only forward to produce sparse mask for fused MHA
	with torch.no_grad():
		_, _, index_mask_scatter = torch_dsa_attention_ref(
			Q_mha,
			K_mha,
			V_mha,
			Q_idx,
			K_idx,
			W_idx,
			Mask,
			index_mask,
			scale,
			index_topk,
		)

	graph = build_mha_graph("dsa_mha", M, N, K, BM, BN, batch, heads, scale)

	global_inputs = {
		"Q": Q_mha,
		"K": K_mha,
		"V": V_mha,
		"Mask": index_mask_scatter,
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
		device=DEVICE,
        enable_causal_opt=False,  # DSA 稀疏 mask，禁用 causal pass
	)

	fused_args_list, _, output_tensor, _ = graph._build_fused_args_from_partition(
		best_partition, global_inputs, params_dict, DEVICE
	)

	def fused_launcher():
		for (compiled, grid), subgraph_args in zip(best_compiled, fused_args_list):
			compiled[grid](*subgraph_args)

	print("\n" + "=" * 70)
	print("Performance Benchmark")
	print("=" * 70)

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
		Out_baseline,
		output_tensor,
		rtol=1e-1,
		atol=1e-1,
	)

	print("\n" + "=" * 70)
	print("DSA v2 Demo Complete!")
	print("=" * 70)


if __name__ == "__main__":
	main()

