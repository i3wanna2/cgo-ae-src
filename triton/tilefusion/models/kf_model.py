import os
import json
import math
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as _np
import random
import numpy as np
from transformers import AutoConfig, AutoTokenizer
from tilefusion.utils.llama_base import collect_weight_dict, init_cos_sin_cache, RmsNorm, MLP, rotary_embedding_online_cuda

# Import ComputeGraph from the local directory
from tilefusion.core.compute_graph import ComputeGraph
import importlib.util
from pathlib import Path
import importlib
from tilefusion.ops.gemm_mgrid_kernel import gemm_mgrid_kernel, launch_gemm_mgrid
from tilefusion.ops.softmax_blockM import softmax_kernel_stable, softmax_triton
from tilefusion.ops.sum import sum_h2o_triton
from tilefusion.ops.elementwise import div, add, log_neg
from tilefusion.ops.gemm_mgrid_loopk_kernel import gemm_mgrid_loopk_kernel, launch_gemm_mgrid_loopk


def triton_baseline(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, exp_rand: torch.Tensor, *, tau: float = 1.5):
    """Baseline implementation using existing Triton ops.

    Inputs:
      q: [B, S_q, H, D]
      k: [B, S_kv, H, D] (current code path assumes group_size == 1)
      v: [B, S_kv, H, D]
      exp_rand: [B, H, S_q, S_kv]

    Returns:
      out: [B, S_q, H, D]
      kf_score: [B, H, S_kv]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda and exp_rand.is_cuda
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    batch, q_len, q_heads, head_dim = q.shape
    batch2, kv_len, kv_heads, head_dim2 = k.shape
    assert batch == batch2 and head_dim == head_dim2
    assert q_heads == kv_heads, "triton_baseline currently assumes q_heads == kv_heads (group_size==1)"
    assert exp_rand.shape == (batch, q_heads, q_len, kv_len)

    scale = 1.0 / math.sqrt(head_dim)

    q_bhsd = q.transpose(1, 2).contiguous()  # [B,H,S,D]
    k_bhsd = k.transpose(1, 2).contiguous()  # [B,H,S,D]
    v_bhsd = v.transpose(1, 2).contiguous()  # [B,H,S,D]

    # GEMM QK^T
    k_mat = k_bhsd.transpose(-2, -1).contiguous()  # [B,H,D,S]
    scores = launch_gemm_mgrid(q_bhsd, k_mat,)  # [B,H,Q,KV]

    # scale + mask
    scores = div(scores, scale=scale, block_m=64, block_n=64)
    base = torch.full((q_len, kv_len), float("-inf"), device=q.device, dtype=q.dtype)
    base = torch.triu(base, diagonal=kv_len - q_len + 1)
    mask = base.unsqueeze(0).unsqueeze(0).expand(batch, q_heads, q_len, kv_len).contiguous()
    scores = add(scores, mask, block_m=64, block_n=64)

    # softmax + GEMM PV
    probs = softmax_triton(scores, block_m=64, block_n=128)  # [B,H,Q,KV]
    # PV GEMM has K=kv_len (e.g. 4096). The mgrid kernel would pick BLOCK_K=4096 and blow up shared memory.
    # Use loop-K variant with small BLOCK_K.
    out_bhsd = launch_gemm_mgrid_loopk(probs, v_bhsd, block_m=64, block_k=32)  # [B,H,Q,D]
    out = out_bhsd.transpose(1, 2).contiguous()  # [B,Q,H,D]

    # Keyformer score
    gumbels = log_neg(exp_rand, block_m=64, block_n=64)
    kf_logits = add(scores, gumbels, block_m=64, block_n=64)
    kf_logits = div(kf_logits, scale=(1.0 / tau), block_m=64, block_n=64)
    kf_probs = softmax_triton(kf_logits, block_m=64, block_n=128)
    kf_score = sum_h2o_triton(kf_probs, block_m=64, block_n=128)

    return out, kf_score


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

    arr = _np.array(times)
    print(f"{label}: runs={runs} mean={arr.mean():.3f} ms median={_np.median(arr):.3f} ms min={arr.min():.3f} ms p95={_np.percentile(arr, 95):.3f} ms")
    return arr


# =============================================================================
# 1. Fused Keyformer Attention Operator (Our System)
# =============================================================================


class FusedKFOp:
    def __init__(self, head_num, head_dim, graph, best_compiled, best_partition, params_dict, tau, output_tensors=None):
        self.head_num = head_num
        self.head_dim = head_dim
        self.graph = graph
        self.best_compiled = best_compiled
        self.best_partition = best_partition
        self.params_dict = params_dict
        self.tau = tau

        # Optimization: Cache mask and pre-allocated outputs
        self.cached_mask = None
        self.cached_seq = -1
        self.cached_device = None
        
        if output_tensors:
            self.output_tensor = output_tensors.get("attn_output").transpose(1, 2)
            self.kf_score_tensor = output_tensors.get("kf_score")
        else:
            self.output_tensor = self.graph.outputs.get("attn_output")
            self.kf_score_tensor = self.graph.outputs.get("kf_score")

        # Fast path cache for arguments
        self.fast_args = None
        self.tensor_indices = {}  # name -> [(subgraph_idx, arg_idx)]

    def __call__(self, q, k, v, exp_rand):
        batch, seq, _, _ = q.shape
        device = q.device
        dtype = q.dtype

        # Optimization: Only regenerate mask if seq or device changes
        if self.cached_mask is None or self.cached_seq != seq or self.cached_device != device:
            causal_mask_bool = torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1)
            self.cached_mask = torch.zeros(batch, self.head_num, seq, seq, device=device, dtype=dtype).masked_fill(causal_mask_bool, float('-inf'))
            self.cached_seq = seq
            self.cached_device = device
            # Reset fast path when seq changes because strides/shapes might change
            self.fast_args = None

        # Transpose to BHSD for ComputeGraph (Metadata only)
        q_bhsd = q.transpose(1, 2)
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        # exp_rand is already [B, H, S, S]

        if self.fast_args is None:
            # Slow path: build arguments and identify tensor positions
            global_inputs = {'Q': q_bhsd, 'K': k_bhsd, 'V': v_bhsd, 'Mask': self.cached_mask, 'ExpRand': exp_rand}
            external_outputs = {
                "attn_output": self.output_tensor,
                "kf_score": self.kf_score_tensor
            }
            self.fast_args, _, _, _ = self.graph._build_fused_args_from_partition(
                self.best_partition, global_inputs, self.params_dict, device,
                external_outputs=external_outputs
            )

            # Identify where Q, K, V, Mask, ExpRand, Outputs are in the argument list
            self.tensor_indices = {'Q': [], 'K': [], 'V': [], 'Mask': [], 'ExpRand': [], 'Output': [], 'KFScore': []}
            for sg_idx, args in enumerate(self.fast_args):
                for arg_idx, arg in enumerate(args):
                    if isinstance(arg, torch.Tensor):
                        ptr = arg.data_ptr()
                        if ptr == q_bhsd.data_ptr():
                            self.tensor_indices['Q'].append((sg_idx, arg_idx))
                        elif ptr == k_bhsd.data_ptr():
                            self.tensor_indices['K'].append((sg_idx, arg_idx))
                        elif ptr == v_bhsd.data_ptr():
                            self.tensor_indices['V'].append((sg_idx, arg_idx))
                        elif ptr == self.cached_mask.data_ptr():
                            self.tensor_indices['Mask'].append((sg_idx, arg_idx))
                        elif ptr == exp_rand.data_ptr():
                            self.tensor_indices['ExpRand'].append((sg_idx, arg_idx))
                        elif ptr == self.output_tensor.data_ptr():
                            self.tensor_indices['Output'].append((sg_idx, arg_idx))
                        elif ptr == self.kf_score_tensor.data_ptr():
                            self.tensor_indices['KFScore'].append((sg_idx, arg_idx))
        else:
            # Fast path: just update tensor pointers
            for name, tensor in [('Q', q_bhsd), ('K', k_bhsd), ('V', v_bhsd), ('Mask', self.cached_mask), ('ExpRand', exp_rand), ('Output', self.output_tensor), ('KFScore', self.kf_score_tensor)]:
                for sg_idx, arg_idx in self.tensor_indices[name]:
                    self.fast_args[sg_idx][arg_idx] = tensor

        # Execute kernels
        for (compiled, grid), args in zip(self.best_compiled, self.fast_args):
            compiled[grid](*args)

        # Return both outputs
        return self.output_tensor.transpose(1, 2), self.kf_score_tensor


def compile(model, input_names, inputs, output_names, system, configs=None, mode='both'):
    # Extract q, k, v, exp_rand from inputs
    q = inputs[0]
    k = inputs[1]
    v = inputs[2]
    exp_rand = inputs[3]
    batch, seq, head, dim = q.shape
    tau = model.tau

    print(f"Compiling 'our' system for B={batch}, S={seq}, H={head}, D={dim}")

    # Setup ComputeGraph and search
    graph = ComputeGraph("kf_attention")
    graph.add_input("Q", "K", "V", "Mask", "ExpRand")
    scale = 1.0 / math.sqrt(dim)

    common_params_mk = {"M": seq, "N": seq, "K": dim, "BM": 64, "BN": 64, "batch": batch, "heads": head}
    common_params_m = {"M": seq, "N": seq, "BM": 64, "BN": 64, "batch": batch, "heads": head}
    common_params_loopk = {"M": seq, "N": dim, "K": seq, "BM": 64, "BN": 64, "batch": batch, "heads": head}

    # Node indices: 0=Q, 1=K, 2=V, 3=Mask, 4=ExpRand
    # Pipeline 1: 5=gemm_qk, 6=scale, 7=add_mask, 8=softmax, 9=gemm_pv
    # Pipeline 2: 10=log_neg (gumbel), 11=add_mask (scores+gumbel), 12=scale_c (1/tau), 13=softmax_recompute, 14=sum_h2o

    graph.add_node("gemm_qk", inputs={"Q": "q_ptr", "K": "k_ptr"}, parents=[0, 1], **common_params_mk) \
        .add_node("scale_c", inputs={"scores": "input"}, parents=[5], **common_params_mk, scale=scale) \
        .add_node("add_mask", inputs={"output": "input", "Mask": "mask_ptr"}, parents=[6, 3], **common_params_m) \
        .add_node("softmax", inputs={"output": "input"}, parents=[7], **common_params_m) \
        .add_node("gemm_pv", inputs={"output": "probs", "V": "v_ptr"}, parents=[8, 2], **common_params_loopk) \
        .add_node("log_neg", inputs={"ExpRand": "input"}, parents=[4], **common_params_m) \
        .add_node("add_mask", inputs={"output": "input", "Mask": "mask_ptr"}, parents=[7, 10], **common_params_m) \
        .add_node("scale_c", inputs={"output": "input"}, parents=[11], **common_params_m, scale=1.0 / tau) \
        .add_node("softmax", inputs={"output": "input"}, parents=[12], **common_params_m) \
        .add_node("sum_h2o", inputs={"output": "input"}, parents=[13], **common_params_m) \
        .add_output("attn_output", source=9, shape=(batch, head, seq, dim)) \
        .add_output("kf_score", source=14, shape=(batch, head, seq))

    mask = torch.zeros(batch, head, seq, seq, device=q.device, dtype=q.dtype)
    global_inputs = {
        'Q': q.transpose(1, 2),
        'K': k.transpose(1, 2),
        'V': v.transpose(1, 2),
        'Mask': mask,
        'ExpRand': exp_rand
    }
    params_dict = {
        'M': seq, 'N': seq, 'K': dim,
        'BM': 64, 'BN': 64,
        'batch': batch, 'heads': head,
        'scale': scale,
        'tau': tau
    }
    search_optimal = True if configs is None else False
    best_compiled, best_partition, best_time = graph.compile(
        num_warps=4,
        num_stages=3,
        global_inputs=global_inputs,
        params_dict=params_dict,
        num_warmup=5,
        num_repeat=20,
        max_splits=2,
        device=q.device,
        configs=configs,
        search_optimal=search_optimal,
        aggressive_gc=(mode != 'kernel'),
    )

    # 释放 autotuning 期间的临时显存，避免端到端场景 OOM
    del global_inputs, mask
    import gc as _gc
    _gc.collect()
    torch.cuda.empty_cache()

    # Create pre-allocated output tensors
    output_tensors = {
        "attn_output": torch.empty((batch, seq, head, dim), device=q.device, dtype=q.dtype),
        "kf_score": torch.empty((batch, head, seq), device=q.device, dtype=q.dtype)
    }

    return FusedKFOp(head, dim, graph, best_compiled, best_partition, params_dict, tau, output_tensors=output_tensors)


# =============================================================================
# 2. Keyformer Model Components
# =============================================================================


class KeyFormer(nn.Module):
    def __init__(self, tau=1.5, kv_head_num=32, head_num=32, head_dim=128):
        super().__init__()
        self.kv_head_num = kv_head_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.hidden_size = head_num * head_dim
        self.kv_hidden_size = kv_head_num * head_dim
        self.group_size = self.head_num // self.kv_head_num
        self.tau = tau
        assert self.group_size == 1

    def forward(self, q, k, v, exp_rand):
        q_len = q.shape[1]
        kv_len = k.shape[1]
        batch_size = q.shape[0]
        mask = torch.full((1, 1, q_len, kv_len), -torch.inf, device=q.device, dtype=q.dtype)
        mask = torch.triu(mask, diagonal=kv_len - q_len + 1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores += mask
        probs = F.softmax(scores.float(), dim=-1)
        out = torch.matmul(probs.to(q.dtype), v).transpose(1, 2).contiguous()

        # keyformer
        gumbels = exp_rand
        gumbels = -gumbels.log()
        kf_score = (scores.float() + gumbels) / self.tau
        kf_score = F.softmax(kf_score.float(), dim=-1)
        kf_score = kf_score.sum(dim=2)

        return out.view(batch_size, q_len, self.head_num, self.head_dim), kf_score.view(batch_size, self.kv_head_num, kv_len)

    def prepare(self, batch_size=1, q_len=4096, kv_len=4096, dtype=torch.float16, device=torch.cuda.current_device(), system='our'):
        q = torch.randn(batch_size, q_len, self.head_num, self.head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, kv_len, self.head_num, self.head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, kv_len, self.head_num, self.head_dim, dtype=dtype, device=device)
        exp_rand_dtype = torch.float32 if system == 'flashtensor' else q.dtype
        exp_rand = 1 + torch.randn(batch_size, self.head_num, q_len, kv_len, dtype=exp_rand_dtype, device=device).abs()

        ret = {
            'input': {
                'q': q,
                'k': k,
                'v': v,
                'exp_rand': exp_rand,
            },
            'output': ['out', 'kf_score']
        }
        return ret


class Attention(nn.Module):
    def __init__(self, hf_config, attn_f, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hf_config.hidden_size
        self.head_num = hf_config.num_attention_heads
        self.kv_head_num = hf_config.num_key_value_heads
        self.head_dim = self.hidden_size // self.head_num
        self.kv_hidden_size = self.kv_head_num * self.head_dim
        self.group_size = self.head_num // self.kv_head_num
        self.cache_budget = hf_config.cache_budget
        self.dtype = hf_config.torch_dtype
        self.system = getattr(hf_config, 'system', 'our')

        self.attn_f = attn_f
        self.tau = hf_config.tau

        self.norm_factor = math.sqrt(self.head_dim)
        self.max_position = hf_config.max_position_embeddings
        self.rotary_base = hf_config.rotary_base
        self.rotary_dim = self.head_dim

        self.embed_positions = nn.Parameter(init_cos_sin_cache(theta=self.rotary_base, dim=self.rotary_dim, max_position=self.max_position))
        self.qkv_proj = nn.Linear(self.hidden_size, self.hidden_size + 2 * self.kv_hidden_size, bias=False, dtype=self.dtype)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False, dtype=self.dtype)

    def forward(self, x, kv_caches):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.hidden_size, self.kv_hidden_size, self.kv_hidden_size], dim=-1)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        pos = torch.arange(0, q.shape[1], dtype=torch.int32, device=x.device)
        rotary_embedding_online_cuda(pos, q, k, self.head_dim, self.rotary_base)
        torch.cuda.synchronize()
        exp_rand_dtype = torch.float32 if self.system == 'flashtensor' else q.dtype
        exp_rand = torch.empty(q.shape[0], self.head_num, q.shape[1], k.shape[1], dtype=exp_rand_dtype, device=q.device).uniform_(1e-8, 1.0)

        out, kf_score = self.attn_f(
            q.view(q.shape[0], q.shape[1], self.head_num, self.head_dim),
            k.view(k.shape[0], k.shape[1], self.kv_head_num, self.head_dim),
            v.view(v.shape[0], v.shape[1], self.kv_head_num, self.head_dim),
            exp_rand,
        )
        out = out.reshape(out.shape[0], out.shape[1], -1)

        _, selected_indices = torch.topk(kf_score, k=self.cache_budget, dim=-1, sorted=True)
        selected_indices = selected_indices.sort(dim=-1)[0]
        selected_indices = selected_indices.transpose(1, 2)[:, :, :, None].expand(q.shape[0], self.cache_budget, self.kv_head_num, self.head_dim)
        k_cache = torch.gather(k.reshape(k.shape[0], -1, self.kv_head_num, self.head_dim), dim=1, index=selected_indices).reshape(k.shape[0], 1, -1, self.kv_head_num, self.head_dim)
        v_cache = torch.gather(v.reshape(v.shape[0], -1, self.kv_head_num, self.head_dim), dim=1, index=selected_indices).reshape(v.shape[0], 1, -1, self.kv_head_num, self.head_dim)
        kv_cache = torch.cat([k_cache, v_cache], dim=1)
        kv_caches.append((kf_score, kv_cache))
        out = self.out_proj(out)
        return out, kv_caches


class LlamaLayer(nn.Module):
    def __init__(self, hf_config, attn_f, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.hf_config = hf_config
        self.input_layernorm = RmsNorm(dim=hf_config.hidden_size, eps=hf_config.rms_norm_eps, dtype=hf_config.torch_dtype)
        self.attention = Attention(hf_config, attn_f, layer_idx)
        self.mlp = MLP(
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=hf_config.hidden_act,
            dtype=hf_config.torch_dtype,
            bias=False,
        )
        self.post_layernorm = RmsNorm(dim=hf_config.hidden_size, eps=hf_config.rms_norm_eps, dtype=hf_config.torch_dtype)

    def forward(self, x, kv_caches):
        res = x
        x = self.input_layernorm(x)
        attn_out, kv_caches = self.attention(x, kv_caches)
        x = res + attn_out
        res = x
        x = self.post_layernorm(x)
        mlp_out = self.mlp(x)
        x = res + mlp_out
        return x, kv_caches


class Llama(nn.Module):
    def __init__(self, hf_config, attn_f):
        super().__init__()
        self.embed_tokens = nn.Embedding(hf_config.vocab_size, hf_config.hidden_size, dtype=hf_config.torch_dtype)
        self.layers = nn.ModuleList()
        for layer_idx in range(hf_config.num_hidden_layers):
            self.layers.append(LlamaLayer(hf_config, attn_f, layer_idx))
        self.rms_norm = RmsNorm(dim=hf_config.hidden_size, eps=hf_config.rms_norm_eps, dtype=hf_config.torch_dtype)
        self.hf_config = hf_config

        self.load_param(hf_config)

    def load_param(self, hf_config):
        weight_dict = collect_weight_dict(hf_config)
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'embed_positions' not in name:
                    param.copy_(weight_dict[name])
        del weight_dict

    def forward(self, token_ids):
        kv_caches = []
        x = self.embed_tokens(token_ids)
        for i in range(len(self.layers)):
            x, kv_caches = self.layers[i](x, kv_caches)
        x = self.rms_norm(x)
        return x, kv_caches


# =============================================================================
# 3. Setup and Main
# =============================================================================

def llm_setup(weight_path, seqlen, layer_num):
    device = torch.cuda.current_device()

    hf_config = AutoConfig.from_pretrained(weight_path)
    cache_budget = 512
    assert cache_budget < seqlen
    hf_config.cache_budget = cache_budget
    hf_config.tau = 1.5

    if layer_num is not None:
        hf_config.num_hidden_layers = layer_num
    hf_config.rotary_base = getattr(hf_config, 'rope_theta', 10000.0)
    print(f"{hf_config.num_hidden_layers=}")

    # assert hf_config.max_position_embeddings >= seqlen
    hf_config.max_position_embeddings = seqlen

    batch_size = 1
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "vcsum.jsonl")
    print(f"{data_path=}")
    with open(data_path, "r", encoding='utf-8') as f:
        data = json.loads(f.readline())
        data = data['context']
    token_ids = AutoTokenizer.from_pretrained(weight_path)(data).input_ids[:seqlen]
    assert len(token_ids) == seqlen
    token_ids = torch.tensor(token_ids, dtype=torch.int64, device=device).reshape(batch_size, seqlen).contiguous()
    print(f"{token_ids.shape=}")

    return hf_config, token_ids


@click.command()
@click.option('--model', '-m', default='kf', help='Model name')
@click.option('--system', '-s', default='our', help='System name (torch, flashtensor, triton, our)')
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

    # Use KeyFormer class for specification
    kernel_cls = KeyFormer
    model_cls = Llama

    # Prepare reference kernel first (same order as attn_model.py)
    kernel = kernel_cls().eval().cuda()
    specs = kernel.prepare(q_len=seqlen, kv_len=seqlen, system=system)
    input_names = list(specs['input'].keys())
    inputs = [specs['input'][name] for name in input_names]
    output_names = specs['output']

    print(f"{input_names=}")
    print(f"{output_names=}")

    kernel_f = None
    attn_callable = None
    if system == 'torch':
        attn_callable = kernel.forward
    elif system == 'triton':
        # Baseline composed of existing Triton ops (not ComputeGraph-fused).
        attn_callable = lambda q, k, v, exp_rand: triton_baseline(q, k, v, exp_rand, tau=kernel.tau)
    elif system == 'flashtensor':
        from tilefusion.utils.flashtensor_wrapper import get_flashtensor_ext_path
        ext_path = get_flashtensor_ext_path(None, model=model, seqlen=seqlen)
        ext_func = 'KeyFormer'
        path = Path(ext_path)
        print(f"write code to {path}", flush=True)
        if not path.exists():
            raise click.BadParameter(f"External path not found: {path}")
        spec = importlib.util.spec_from_file_location('our', str(path))
        pymod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pymod)
        if not hasattr(pymod, ext_func):
            raise click.BadParameter(f"Function {ext_func} not found in {path}")
        ext_fn = getattr(pymod, ext_func)
        if not callable(ext_fn):
            raise click.BadParameter(f"{ext_func} in {path} is not callable")
        def ext_attn_wrapper(q, k, v, exp_rand):
            return ext_fn(q, k, v, exp_rand)
        attn_callable = ext_attn_wrapper
    elif system == 'our':
        print("Compiling KeyFormer kernel...", flush=True)
        # test_configs = [{'BM': 64, 'BN': 64, 'num_warps': 4, 'num_stages': 3}]
        kernel_f = compile(
            model=kernel,
            input_names=input_names,
            inputs=inputs,
            output_names=output_names,
            system=system,
            mode=mode,
            # configs=test_configs,
        )
        attn_callable = kernel_f
    # Prepare attention inputs (from kernel prepare)
    q = specs['input']['q']
    k = specs['input']['k']
    v = specs['input']['v']
    exp_rand = specs['input']['exp_rand']
    # Ensure everything is ready on device
    torch.cuda.synchronize()

    # 0) Check correctness
    if check:
        print("Checking correctness against Torch...")
        try:
            with torch.no_grad():
                out_ref, kf_ref = kernel(q, k, v, exp_rand)
                out_test, kf_test = attn_callable(q, k, v, exp_rand)
                torch.testing.assert_close(out_test.to(torch.float16), out_ref.to(torch.float16), rtol=1e-3, atol=1e-2)
                torch.testing.assert_close(kf_test.to(torch.float16), kf_ref.to(torch.float16), rtol=1e-3, atol=1e-2)
                print("Correctness check passed!")
        except Exception as exc:
            print(f"Correctness check failed: {exc}")

    # 1) Benchmark attention kernel only (skip when external function provided)

    def run_kernel():
        # call compiled kernel via kernel_f
        _ = attn_callable(q, k, v, exp_rand)

    attn_times = benchmark_fn(run_kernel, warmup=20, runs=100, label=model + ' kernel')

    if mode == 'kernel':
        return 

    weight_zoo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "weight_zoo.json")
    print(f"{weight_zoo_path=}")

    weight_path = ""
    if os.path.exists(weight_zoo_path):
        with open(weight_zoo_path, "r") as f:
            weight_zoo = json.load(f)
        weight_path = weight_zoo.get(platform, "")

    hf_config, token_ids = llm_setup(weight_path, seqlen, layer_num)
    hf_config.system = system

    # build model: pass the attention callable (compiled kernel or external function)
    model_inst = model_cls(
        hf_config=hf_config,
        attn_f=attn_callable,
    )
    for param in model_inst.parameters():
        param.requires_grad = False
    model_inst = model_inst.eval().cuda()

    # -------------------------
    # Micro-benchmarks
    # -------------------------
    if mode == 'both':
        torch.cuda.synchronize() 
        # 2) Benchmark end-to-end model (single forward)
        def run_model():
            with torch.no_grad():
                model_inst(token_ids)

        model_times = benchmark_fn(run_model, warmup=50, runs=50, label=model + ' E2E')

    print("Inference + benchmarks finished.")


if __name__ == '__main__':
    main()
