import os
# Must set before importing torch to enable expandable segments for better memory management
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import json
import click
import torch
import numpy as np
import random
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer

import time
import importlib.util
from pathlib import Path
import importlib
import sys

def benchmark_fn(fn, warmup=10, runs=50, label="benchmark"):
    """Measure ms per call using CUDA Events."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Use CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(runs):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))
        
    import numpy as _np
    arr = _np.array(times)
    print(f"{label}: runs={runs} mean={arr.mean():.3f} ms median={_np.median(arr):.3f} ms min={arr.min():.3f} ms p95={_np.percentile(arr,95):.3f} ms")
    return arr

# =============================================================================
# 1. Fused DSA MLA Operator (Our System)
# =============================================================================

class FusedDSAMLAOp:
    def __init__(self, heads, M, N, K, D, TopK, graph, best_compiled, best_partition, params_dict, output_tensor=None):
        self.heads = heads
        self.M = M
        self.N = N
        self.K = K
        self.D = D
        self.TopK = TopK
        self.graph = graph
        self.best_compiled = best_compiled
        self.best_partition = best_partition
        self.params_dict = params_dict
        
        # Optimization: Cache mask and pre-allocated output
        self.cached_mask = None
        self.cached_seq = -1
        self.cached_device = None
        
        if output_tensor is not None:
            self.output_tensor = output_tensor.transpose(1, 2)
        else:
            self.output_tensor = self.graph.outputs.get("attn_output")
        
        # Fast path cache for arguments
        self.fast_args = None
        self.tensor_indices = {} # name -> [(subgraph_idx, arg_idx)]

    def __call__(self, q, k, indices):
        # Input q: [B, S, H, K], k: [B, S, G, K], indices: [B, S, G, TopK]
        # Transpose to [B, H, S, K] for internal kernels
        q_internal = q.transpose(1, 2)
        k_internal = k.transpose(1, 2)
        indices_internal = indices.transpose(1, 2)

        batch = q_internal.shape[0]
        device = q_internal.device
        dtype = q_internal.dtype

        # Optimization: Only regenerate mask if seq or device changes
        if self.cached_mask is None or self.cached_seq != self.M or self.cached_device != device:
            causal_mask_bool = torch.triu(torch.ones(self.M, self.TopK, device=device, dtype=torch.bool), diagonal=1)
            # Use expand() instead of allocating full (batch, heads, M, TopK) tensor
            # This saves ~4GB for seqlen=8192: (1*128*8192*2048*2) bytes
            causal_mask_base = torch.zeros(self.M, self.TopK, device=device, dtype=dtype).masked_fill(causal_mask_bool, float('-inf'))
            self.cached_mask = causal_mask_base.unsqueeze(0).unsqueeze(0).expand(batch, self.heads, -1, -1)
            # Note: Do NOT call .contiguous() - kernel handles broadcasting via strides
            del causal_mask_bool, causal_mask_base
            self.cached_seq = self.M
            self.cached_device = device
            # Reset fast path when seq changes because strides/shapes might change
            self.fast_args = None

        if self.fast_args is None:
            # Slow path: build arguments and identify tensor positions
            global_inputs = {'Q': q_internal, 'K': k_internal, 'Indices': indices_internal, 'Mask': self.cached_mask}
            external_outputs = {"attn_output": self.output_tensor}
            self.fast_args, _, _, _ = self.graph._build_fused_args_from_partition(
                self.best_partition, global_inputs, self.params_dict, device,
                external_outputs=external_outputs
            )
            
            # Identify where Q, K, Indices, Mask, Output are in the argument list
            self.tensor_indices = {'Q': [], 'K': [], 'Indices': [], 'Mask': [], 'Output': []}
            for sg_idx, args in enumerate(self.fast_args):
                for arg_idx, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        ptr = arg.data_ptr()
                        if ptr == q_internal.data_ptr():
                            self.tensor_indices['Q'].append((sg_idx, arg_idx))
                        elif ptr == k_internal.data_ptr():
                            self.tensor_indices['K'].append((sg_idx, arg_idx))
                        elif ptr == indices_internal.data_ptr():
                            self.tensor_indices['Indices'].append((sg_idx, arg_idx))
                        elif ptr == self.cached_mask.data_ptr():
                            self.tensor_indices['Mask'].append((sg_idx, arg_idx))
                        elif ptr == self.output_tensor.data_ptr():
                            self.tensor_indices['Output'].append((sg_idx, arg_idx))
        else:
            # Fast path: just update tensor pointers in the cached argument list
            for name, tensor in [('Q', q_internal), ('K', k_internal), ('Indices', indices_internal), ('Mask', self.cached_mask), ('Output', self.output_tensor)]:
                for sg_idx, arg_idx in self.tensor_indices[name]:
                    self.fast_args[sg_idx][arg_idx] = tensor

        # Execute kernels
        for (compiled, grid), args in zip(self.best_compiled, self.fast_args):
            compiled[grid](*args)

        # Return [B, S, H, D]
        return self.output_tensor.transpose(1, 2)
    
def generate_safe_mla_indices(B, S, G, TopK, device="cuda", padding_index=None):
    """
    生成符合 Sparse MLA 要求的稀疏索引
    Shape: [B, S, G, TopK]

    TileLang uses index=S as an out-of-range padding sentinel and filters it
    with its causal mask. TileFusion kernels gather before applying their
    positional mask, so they must use an in-range sink index (0) instead.
    """
    if padding_index is None:
        padding_index = S
    indices = torch.full(
        (B, S, G, TopK), padding_index, dtype=torch.int32, device=device
    )
    
    for t in range(S):
        valid_len = t + 1
        
        if valid_len <= TopK:
            selection = torch.arange(valid_len, device=device, dtype=torch.int32)
            indices[:, t, :, :valid_len] = selection.view(1, 1, -1)
        else:
            # 使用 randperm 采样 TopK 个
            for b in range(B):
                for g in range(G):
                    perm = torch.randperm(valid_len, device=device)[:TopK]
                    indices[b, t, g, :] = perm.to(torch.int32)
    
    return indices

def compile(inputs, D, TopK, system, device, configs=None):
    import gc
    from tilefusion.core.compute_graph import ComputeGraph
    batch = 1
    G = 1      
    BM = 64
    BN = 64
    num_warps = 8
    num_stages = 3
    Q_orig = inputs[0]
    Kmat_orig = inputs[1]
    indices_orig = inputs[2]
    # Q_orig shape: [B, S, H, K] - where S=seqlen, H=heads
    M = Q_orig.shape[1]       # seqlen
    N = Kmat_orig.shape[1]    # seqlen
    K = Q_orig.shape[3]
    heads = Q_orig.shape[2]   # heads
    scale = 1.0 / math.sqrt(K)
    print("=" * 70)
    print("DSA MLA v2 - Using ComputeGraph + Kernel Registry")
    print("=" * 70)
    print(f"Problem: batch={batch}, heads={heads}, M={M}, N={N}, K={K}, D={D}, TopK={TopK}")
    print(f"Block: BM={BM}, BN={BN}, warps={num_warps}, stages={num_stages}")

    # Transpose for our system's compilation (which expects B H S D)
    Q = Q_orig.transpose(1, 2).contiguous()
    Kmat = Kmat_orig.transpose(1, 2).contiguous()
    indices = indices_orig.transpose(1, 2).contiguous()

    causal_mask_bool = torch.triu(
        torch.ones(M, TopK, dtype=torch.bool, device=device),
        diagonal=1
    )
    causal_mask_float = torch.zeros(M, TopK, dtype=torch.float16, device=device)
    causal_mask_float = causal_mask_float.masked_fill(causal_mask_bool, float('-inf'))
    Mask = causal_mask_float.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1).contiguous()

    graph = ComputeGraph("dsa_mla_graph")
    graph.add_input("Q", "K", "Indices", "Mask")

    common_params = {
        "M": M, "N": N, "TopK": TopK, "K": K, "D": D, "BLOCK_D1": D, "BLOCK_D2": K - D, 
        "BM": BM, "BN": BN, "batch": batch, "heads": heads
    }
    
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
    
    graph.add_node(
        "softmax_mgrid",
        inputs={"output": "input"},
        parents=[6],
        M=M, TopK=TopK, BM=BM, BN=BN, batch=batch, heads=heads
    )
    
    graph.add_node(
        "gather_gemm_sv_mgrid",
        inputs={"output": "Scores", "K": "V", "Indices": "Indices"},
        parents=[7, 1, 2],  
        **common_params
    )
    
    # 添加输出节点
    graph.add_output("attn_output", source=8, shape=(batch, heads, M, D), device=device)

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

    # If configs is provided, skip Phase 2 autotuning
    search_optimal = True if configs is None else False

    best_compiled, best_partition, best_time = graph.compile(
        num_warps=num_warps,
        num_stages=num_stages,
        search_optimal=search_optimal,
        global_inputs=global_inputs,
        params_dict=params_dict,
        num_warmup=5,
        num_repeat=20,
        max_splits=1,
        device=device,
        configs=configs,
    )

    # Clean up temporary tensors used for compilation/autotuning to free GPU memory
    # Note: Q_orig, Kmat_orig, indices_orig come from specs (inputs) and will be cleaned up later
    del Q, Kmat, indices, Mask
    del global_inputs
    gc.collect()
    torch.cuda.empty_cache()

    output_tensor = torch.empty((batch, M, heads, D), device=device, dtype=torch.float16)
    return FusedDSAMLAOp(heads, M, N, K, D, TopK, graph, best_compiled, best_partition, params_dict, output_tensor=output_tensor)


block_size = 128

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: str = "fp16"
    scale_fmt: Optional[str] = None
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: str = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    rms_norm_eps: float = 1e-6
    torch_dtype: torch.dtype = torch.float16

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True) -> torch.Tensor:
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)

def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    from fast_hadamard_transform import hadamard_transform
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size ** -0.5)


# =============================================================================
# 2. FP8 Fallback / Optimized Kernels
# =============================================================================

DEVICE_NAME = torch.cuda.get_device_name(0)

def act_quant(x, block_size, scale_fmt):
    # Performance-only simplification: skip actual quantization logic
    # Just return x cast to fp8 and a dummy scale to save memory/time
    scale = torch.ones(x.shape[:-1] + (x.shape[-1] // block_size,), device=x.device, dtype=torch.float32)
    return x.to(torch.float8_e4m3fn), scale

def fp8_gemm(x, x_scale, w, w_scale):
    # Performance-only simplification: direct FP16 matmul
    return F.linear(x.to(torch.float16), w.to(torch.float16))

def fp8_index(q, weights, k_cache, k_scale_cache):
    # Performance-only simplification: avoid materializing the huge 4D tensor
    # Original: [bsz, seqlen, n_heads, end_pos] -> ~17GB for 8k seqlen
    # We compute a reduced version to save memory while keeping the computation load
    bsz, seqlen, n_heads, head_dim = q.shape
    _, end_pos, _ = k_cache.shape
    
    # [bsz, seqlen, n_heads, head_dim] -> [bsz, seqlen, head_dim] (sum over heads)
    q_sum = q.to(torch.float16).sum(dim=2) 
    # [bsz, seqlen, head_dim] @ [bsz, head_dim, end_pos] -> [bsz, seqlen, end_pos]
    logits = torch.matmul(q_sum, k_cache.to(torch.float16).transpose(-1, -2))
    # Return [bsz, seqlen, end_pos]
    return torch.relu(logits) * k_scale_cache.unsqueeze(1)

# Try to use optimized kernels if not on A100
if 'A100' not in DEVICE_NAME:
    try:
        from tilefusion.ops.dsa_kernel import act_quant as aq_opt, fp8_gemm as fg_opt, fp8_index as fi_opt
        act_quant, fp8_gemm, fp8_index = aq_opt, fg_opt, fi_opt
        print("Using optimized FP8 kernels.")
    except ImportError:
        print("Optimized kernels not found, using fallback.")
else:
    print("A100 detected, using FP16-simulated FP8 kernels.")

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           scale_fmt: Optional[str] = None) -> torch.Tensor:
    assert bias is None

    if weight.dtype != torch.float8_e4m3fn:
        return F.linear(x, weight)
    else:
        x, scale = act_quant(x, block_size, scale_fmt)
        return fp8_gemm(x, scale, weight, weight.scale)

class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        # Use float16 for embedding even if model is fp8
        dtype = Linear.dtype if Linear.dtype != torch.float8_e4m3fn else torch.float16
        self.weight = nn.Parameter(torch.empty(vocab_size, self.dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)

class Linear(nn.Module):
    dtype = torch.float16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias, self.scale_fmt)

class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, reduce_output = True, dtype = None):
        super().__init__(in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias, self.scale_fmt)
        return y.type_as(x)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        dtype = x.dtype
        if residual is None:
            x = x.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype)
        else:
            x = residual = x.float() + residual.float()
            var = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(var + self.eps)
            return (self.weight * x).to(dtype), residual.to(dtype)

class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)

def weight_dequant(weight, scale):
    shape = weight.shape
    assert weight.dim() == 2
    weight = weight.view(shape[0] // block_size, block_size, shape[1] // block_size, block_size).transpose(1, 2).contiguous().view(-1, block_size * block_size)
    weight = (weight.float() * scale.view(-1, 1).float()).to(torch.get_default_dtype()).view(shape[0] // block_size, shape[1] // block_size, block_size, block_size).transpose(1, 2).contiguous().view(shape)
    return weight

class Indexer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.k_norm = LayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in fp16.
        self.weights_proj = Linear(self.dim, self.n_heads)
        self.softmax_scale = self.head_dim ** -0.5
        self.scale_fmt = args.scale_fmt

        self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float8_e4m3fn), persistent=False)
        # k_scale_cache: [batch, seq] - sum of scales for fp8_index kernel
        self.register_buffer("k_scale_cache", torch.zeros(args.max_batch_size, args.max_seq_len, dtype=torch.float32), persistent=False)


    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        q = rotate_activation(q)
        k = rotate_activation(k)
        q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
        self.k_cache[:bsz, start_pos:end_pos] = k_fp8
        # Sum k_scale over last dim: [b, n, d//block_size] -> [b, n]
        self.k_scale_cache[:bsz, start_pos:end_pos] = k_scale.sum(dim=-1)
        
        weights = self.weights_proj(x) * self.n_heads ** -0.5
        # q_scale: [b, m, h, d//block_size], sum to [b, m, h] then multiply
        q_scale_sum = q_scale.sum(dim=-1)  # [b, m, h]
        weights = weights * q_scale_sum * self.softmax_scale  # [b, m, h]
        
        index_score = fp8_index(
            q_fp8.contiguous(), 
            weights.contiguous(), 
            self.k_cache[:bsz, :end_pos].contiguous(), 
            self.k_scale_cache[:bsz, :end_pos].contiguous()
        )
        if mask is not None:
            index_score += mask
        topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1].to(torch.int32)
        return topk_indices

class MLA(nn.Module):
    def __init__(self, args: ModelArgs, attn_f=None):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.scale_fmt = args.scale_fmt
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.indexer = Indexer(args)

        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
        self.dequant_wkv_b = None
        self.attn_f = attn_f

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_norm(kv)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        # we use fp8 kv cache in actual deployment, so here we simulate the precision by casting kv to fp8 and then back to fp16.
        kv_fp8, kv_scale = act_quant(kv, block_size, self.scale_fmt)
        kv = (kv_fp8.view(-1, block_size).float() * kv_scale.view(-1, 1)).to(kv.dtype).view_as(kv)
        self.kv_cache[:bsz, start_pos:end_pos] = kv
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        if mask is not None:    # MHA prefill
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(kv)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            
            # indexer
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)

            if self.attn_f is not None:
                # Use [B, S, H, D] layout directly
                # Use only the first head for KV to match G=1 in benchmark
                k_kernel = k[:, :, :1, :].contiguous()
                # topk_indices is [B, S, TopK], we need [B, S, G, TopK]
                indices_kernel = topk_indices.unsqueeze(2).contiguous() # [B, S, 1, TopK]
                
                x = self.attn_f(q.contiguous(), k_kernel, indices_kernel)
                if isinstance(x, tuple):
                    x = x[0]
                # x is now [B, S, H, D]
                x = x.reshape(bsz, seqlen, -1)
            else:
                scores = torch.einsum("bshd,bthd->bsht", q, k).mul_(self.softmax_scale)
                index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
                index_mask += mask
                scores += index_mask.unsqueeze(2)

                scores = scores.softmax(dim=-1)
                x = torch.einsum("bsht,bthd->bshd", scores, v).flatten(2)
        else:                   # MQA decode
            if self.dequant_wkv_b is None and self.wkv_b.scale is not None:
                self.dequant_wkv_b = weight_dequant(self.wkv_b.weight, self.wkv_b.scale)
            wkv_b = self.wkv_b.weight if self.dequant_wkv_b is None else self.dequant_wkv_b
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale

            # indexer
            topk_indices = self.indexer(x, qr, start_pos, freqs_cis, mask)
            index_mask = torch.full((bsz, 1, end_pos), float("-inf"), device=x.device).scatter_(-1, topk_indices, 0)
            scores += index_mask.unsqueeze(2)

            scores = scores.softmax(dim=-1)
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
            x = x.flatten(2)
        x = self.wo(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int, reduce_output: bool = True):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim, reduce_output=reduce_output)
        self.w3 = ColumnParallelLinear(dim, inter_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))

class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32)) if self.dim == 7168 else None
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices

class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).type_as(x))

class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_activated_experts = args.n_activated_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim, reduce_output=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, attn_f=None):
        super().__init__()
        self.attn = MLA(args, attn_f)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)
    def forward(self, x, residual, start_pos, freqs_cis, mask):
        if residual is None:
            x, residual = self.attn_norm(x), x
        else:
            x, residual = self.attn_norm(x, residual)
        x = self.attn(x, start_pos, freqs_cis, mask)
        x, residual = self.ffn_norm(x, residual)
        x = self.ffn(x)
        return x, residual

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, attn_f=None):
        super().__init__()
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.float16
        Linear.scale_fmt = args.scale_fmt
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args, attn_f))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.float32)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        seqlen = tokens.size(1)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1) if seqlen > 1 else None
        h, residual = self.embed(tokens), None
        for layer in self.layers:
            h, residual = layer(h, residual, start_pos, freqs_cis, mask)
        h, _ = self.norm(h, residual)
        logits = self.head(h[:, -1].float())
        return logits

# =============================================================================
# 4. DSA MLA Kernel Class (Specification)
# =============================================================================

def ref_sparse_mla_fwd_interface(q, kv, indices, dv=128, sm_scale=None, is_casual=True):
    """
    参考实现 (来自 dsa_mla.py)
    
    Args:
        q: [B, M, H, D] Query
        kv: [B, N, G, D] Key/Value 合并 (G=1)
        indices: [B, M, G, TopK] 稀疏索引
        dv: head dimension for V
        sm_scale: softmax scale
        is_casual: 是否使用 causal mask
        
    Returns:
        o: [B, M, H, Dv] 输出
    """
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    k = kv
    v = kv[..., :dv]

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

class DSAMLA(nn.Module):
    def __init__(self, heads=128, M=4096, N=4096, K=576, D=512, TopK=2048):
        super().__init__()
        self.heads = heads
        self.M = M
        self.N = N
        self.K = K
        self.D = D
        self.TopK = TopK

    def forward(self, q, k, indices):
        res = ref_sparse_mla_fwd_interface(
            q, 
            k, 
            indices, 
            dv=self.D,
            sm_scale=self.K**-0.5,
            is_casual=True
        )
        return res

    def prepare(
        self,
        batch_size=1,
        seqlen=4096,
        dtype=torch.float16,
        device="cuda",
        padding_index=None,
    ):
        # Use [B, S, H, D] layout
        Q = torch.randn(batch_size, seqlen, self.heads, self.K, device=device, dtype=torch.float16)
        Kmat = torch.randn(batch_size, seqlen, 1, self.K, device=device, dtype=torch.float16)
        indices = generate_safe_mla_indices(
            batch_size,
            seqlen,
            1,
            self.TopK,
            device=device,
            padding_index=padding_index,
        )
        ret = {
            'input': {
                'q': Q,
                'k': Kmat,
                'indices': indices,
            },
            'output': ['out']
        }
        return ret

# =============================================================================
# 4. Setup and Main
# =============================================================================

def llm_setup(seqlen, layer_num, vocab_size=None):
    device = torch.cuda.current_device()
    args = ModelArgs()
    args.max_seq_len = seqlen
    if layer_num is not None:
        args.n_layers = layer_num
    if vocab_size is not None:
        args.vocab_size = vocab_size
    # Use random token IDs for a cleaner and more robust benchmark
    print(f"Using random token IDs (vocab_size={args.vocab_size})")
    batch_size = 1
    token_ids = torch.randint(0, args.vocab_size, (batch_size, seqlen), dtype=torch.int64, device=device).contiguous()

    return args, token_ids

@click.command()
@click.option('--model', '-m', default='dsa_mla', help='Model name')
@click.option('--system', '-s', default='our', help='System name (our, torch, tilelang, tilelang-ws)')
@click.option('--seqlen', type=int, default=4096, help='seqlen')
@click.option('--layer_num', type=int, default=None, help='layer_num')
@click.option('--platform', '-p', default='H800', help='platform(H800, A100, H100)')
@click.option('--mode', default='kernel', type=click.Choice(['kernel', 'both']), help='Benchmark mode: kernel only or both kernel and E2E')
@click.option('--check/--no-check', default=True, help='Check correctness against Torch')
def main(model, system, seqlen, layer_num, platform, mode, check):
    print(f"{model=} {system=} {seqlen=} {layer_num=} {mode=} {check=}")
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Problem scale (Aligned with dsa_mla_v2.py)
    heads = 128
    K = 576 
    D = 512
    if platform == 'A100':
        K = 144 
        D = 128
    if seqlen <= 2048:
        TopK = int(seqlen / 2)
    else:
        TopK = 2048

    kernel_cls = DSAMLA
    model_cls = Transformer

    kernel = kernel_cls(heads=heads, M=seqlen, N=seqlen, K=K, D=D, TopK=TopK).eval().cuda()
    # TileFusion gathers K/V before masking, so an out-of-range sentinel can
    # produce non-finite values on some GPUs (for example, A100). TileLang
    # deliberately uses S as a sentinel and filters it in its own causal mask.
    padding_index = 0 if system == "our" else seqlen
    specs = kernel.prepare(seqlen=seqlen, padding_index=padding_index)
    input_names = list(specs['input'].keys())
    inputs = [specs['input'][name] for name in input_names]
    output_names = specs['output']
    kernel_f = None
    attn_callable = None

    if system == 'tilelang':
        tilelang_interface = None
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        tilelang_path = os.path.join(os.path.dirname(CURRENT_DIR), f"baselines/tilelang_dsa_h{K}.py")
        if os.path.exists(tilelang_path):
            try:
                spec = importlib.util.spec_from_file_location(f'tilelang_dsa_h{K}', tilelang_path)
                tilelang_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tilelang_mod)
                if hasattr(tilelang_mod, 'sparse_mla_fwd_interface'):
                    tilelang_interface = tilelang_mod.sparse_mla_fwd_interface
                    print(f"Loaded TileLang interface from {tilelang_path}")
            except Exception as e:
                print(f"Failed to load TileLang interface from {tilelang_path}: {e}")
        if tilelang_interface is None:
            raise click.BadParameter("TileLang interface not loaded. Check if ../tilelang_dsa.py exists.")
        def tilelang_attn_wrapper(q, k, indices):
            # q: [B, S, H, D], k: [B, S, G, D], indices: [B, S, G, TopK]
            # TileLang interface expects these shapes directly
            out, lse = tilelang_interface(q, 
                                        k, 
                                        indices, 
                                        d_v=D,
                                        block_I=64, num_stages=2, threads=256
                                    )
            return out
        attn_callable = tilelang_attn_wrapper
    elif system == 'tilelang-ws':
        tilelang_ws_interface = None
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        tilelang_ws_path = os.path.join(os.path.dirname(CURRENT_DIR), f"baselines/tilelang_dsa_h{K}_ws.py")
        if os.path.exists(tilelang_ws_path):
            try:
                spec = importlib.util.spec_from_file_location(f'tilelang_dsa_h{K}_ws', tilelang_ws_path)
                tilelang_ws_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(tilelang_ws_mod)
                if hasattr(tilelang_ws_mod, 'sparse_mla_fwd_interface'):
                    tilelang_ws_interface = tilelang_ws_mod.sparse_mla_fwd_interface
                    print(f"Loaded TileLang-WS interface from {tilelang_ws_path}")
            except Exception as e:
                print(f"Failed to load TileLang-WS interface from {tilelang_ws_path}: {e}")
        if tilelang_ws_interface is None:
            raise click.BadParameter(f"TileLang-WS interface not loaded. Check if {tilelang_ws_path} exists.")
        # Use q_start_s_index=seqlen to match official test setup (avoids early-position edge cases)
        _ws_q_start = seqlen
        _device = torch.cuda.current_device()
        _B = specs['input']['q'].shape[0]
        _ws_indices = torch.full((_B, seqlen, 1, TopK), seqlen, dtype=torch.int32, device=_device)
        for _t in range(seqlen):
            for _b in range(_B):
                _valid = min(max(1, _t + _ws_q_start), seqlen)
                _perm = torch.randperm(_valid, device=_device)[:TopK]
                _ws_indices[_b, _t, 0, :len(_perm)] = _perm.to(torch.int32)
        specs['input']['indices'] = _ws_indices
        def tilelang_ws_attn_wrapper(q, k, indices):
            # q: [B, S, H, D], k: [B, S, G, D], indices: [B, S, G, TopK]
            out, lse = tilelang_ws_interface(q, k, indices, _ws_q_start, 1)
            return out
        attn_callable = tilelang_ws_attn_wrapper
    elif system == 'torch':
        attn_callable = kernel.forward
    elif system == 'dynamo':
        print("Compiling with Torch Dynamo...", flush=True)
        attn_callable = torch.compile(kernel.forward)
    elif system == 'tensorrt':
        print("Compiling with TensorRT via ONNX...", flush=True)
        onnx_path = "dsa_mla.onnx"
        engine_path = "dsa_mla.engine"
        
        # 1. Export to ONNX
        torch.onnx.export(
            kernel, 
            (specs['input']['q'], specs['input']['k'], specs['input']['indices']),
            onnx_path,
            input_names=['q', 'k', 'indices'],
            output_names=['out'],
            # opset_version=17,
            do_constant_folding=True
        )
        
        # 2. Build TRT Engine
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX file")
        
        config.set_flag(trt.BuilderFlag.FP16)
        # Set memory pool limit for TRT 10.x
        # A100 has plenty of memory, let's give it 8GB to avoid tactic skipping
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 24 << 30) 
        
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("Failed to build TRT engine")
            
        with open(engine_path, 'wb') as f:
            f.write(engine_bytes)
            
        # 3. Load and Wrap
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()
        static_out = torch.empty((1, seqlen, heads, D), device='cuda', dtype=torch.float16)
        def trt_attn_wrapper(q, k, indices):
            q = q.contiguous()
            k = k.contiguous()
            indices = indices.contiguous()
            bindings = [
                q.data_ptr(), 
                k.data_ptr(), 
                indices.data_ptr(), 
                static_out.data_ptr()
            ]
            success = context.execute_v2(bindings)
            
            if not success:
                raise RuntimeError("TensorRT execute_v2 failed!")
                
            return static_out
            
        attn_callable = trt_attn_wrapper
    elif system == 'our':
        print("Compiling fused DSA MLA kernel...", flush=True)
        # Example config to skip Phase 2 autotuning
        test_configs = [{'BM': 64, 'BN': 64, 'num_warps': 8, 'num_stages': 2}] if platform == 'A100' else None
        kernel_f = compile(inputs=inputs,
            D=D, TopK=TopK,
            system=system, device=torch.cuda.current_device(),
            # configs=test_configs,
        )
        attn_callable = kernel_f

    torch.cuda.synchronize()
    q = specs['input']['q']
    k = specs['input']['k']
    indices = specs['input']['indices']
    # 0) Check correctness
    if check:
        print("Checking correctness against Torch...")
        try:
            with torch.no_grad():
                if system == 'tilelang-ws':
                    out_ref = tilelang_ws_mod.ref_sparse_mla_fwd_interface(
                        q, k, indices, q_start_index_s=_ws_q_start, kv_stride=1,
                        sm_scale=K**-0.5,
                    )
                else:
                    out_ref = kernel(q, k, indices)
                out_test = attn_callable(q, k, indices)
                torch.testing.assert_close(out_test, out_ref, rtol=1e-3, atol=1e-2)
                print("Correctness check passed!")
        except Exception as exc:
            print(f"Correctness check failed: {exc}")

    # 1) Benchmark attention kernel only
    def run_kernel():
        _ = attn_callable(q, k, indices)
    benchmark_fn(run_kernel, warmup=20, runs=100, label=model + ' kernel')

    if mode == 'kernel':
        return

    # Clean up kernel benchmark tensors before E2E benchmark to free GPU memory
    import gc
    del q, k, indices, specs, inputs
    del kernel  # Reference kernel no longer needed
    gc.collect()
    torch.cuda.empty_cache()
    vocab_size = 102400
    if platform == 'A100':
        vocab_size = 1000
    args, token_ids = llm_setup(seqlen, layer_num, vocab_size=vocab_size)
    if platform == 'A100':
        args.n_layers = args.n_layers // 2
        args.dim = 512
        args.max_batch_size = 1
        args.inter_dim = 1024
        # args.moe_inter_dim = 512
        # args.kv_lora_rank = 256
        # args.q_lora_rank = 256
        # args.n_routed_experts = 8
        # print(f"Aggressively reducing model dimensions for 40GB GPU performance test")

    # Update args to match benchmark parameters
    args.n_heads = heads
    args.v_head_dim = D
    args.qk_nope_head_dim = K - args.qk_rope_head_dim
    args.index_topk = TopK

    model_inst = model_cls(args=args, attn_f=attn_callable)
    model_inst = model_inst.eval().cuda()

    # 2) Benchmark end-to-end model (single forward)
    def run_model():
        with torch.no_grad():
            model_inst(token_ids)
    
    if mode == 'both':
        print("Starting E2E benchmark...")
        benchmark_fn(run_model, warmup=50, runs=50, label=model + ' E2E')

    print("Inference + benchmarks finished.")


if __name__ == '__main__':
    main()

