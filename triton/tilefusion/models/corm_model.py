import json
import click
import torch
import numpy as np
import random
import os
import math

import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer
from tilefusion.utils.llama_base import collect_weight_dict, init_cos_sin_cache, RmsNorm, MLP, rotary_embedding_online_cuda

# Import ComputeGraph from the local directory
from tilefusion.core.compute_graph import ComputeGraph
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
    print(f"{label}: runs={runs} mean={arr.mean():.3f} ms median={_np.median(arr):.3f} ms min={arr.min():.3f} ms p95={_np.percentile(arr, 95):.3f} ms")
    return arr

# =============================================================================
# 1. Fused Corm Attention Operator (Our System)
# =============================================================================


class FusedCormOp:
    def __init__(self, head_num, head_dim, graph, best_compiled, best_partition, params_dict, output_tensors=None):
        self.head_num = head_num
        self.head_dim = head_dim
        self.graph = graph
        self.best_compiled = best_compiled
        self.best_partition = best_partition
        self.params_dict = params_dict

        # Optimization: Cache mask and pre-allocated outputs
        self.cached_mask = None
        self.cached_corm_mask = None
        self.cached_seq = -1
        self.cached_device = None
        
        if output_tensors:
            self.output_tensor = output_tensors.get("attn_output").transpose(1, 2)
            self.corm_score_tensor = output_tensors.get("corm_score")
        else:
            self.output_tensor = self.graph.outputs.get("attn_output")
            self.corm_score_tensor = self.graph.outputs.get("corm_score")

        # Fast path cache for arguments
        self.fast_args = None
        self.tensor_indices = {}  # name -> [(subgraph_idx, arg_idx)]

    def __call__(self, q, k, v, corm_mask):
        batch, seq, _, _ = q.shape
        device = q.device
        dtype = q.dtype

        # Optimization: Only regenerate mask if seq or device changes
        if self.cached_mask is None or self.cached_seq != seq or self.cached_device != device:
            causal_mask_bool = torch.triu(torch.ones(seq, seq, device=device, dtype=torch.bool), diagonal=1)
            self.cached_mask = torch.zeros(batch, self.head_num, seq, seq, device=device, dtype=dtype).masked_fill(causal_mask_bool, float('-inf'))
            self.cached_corm_mask = None  # Reset corm mask cache as well
            self.cached_seq = seq
            self.cached_device = device
            # Reset fast path when seq changes because strides/shapes might change
            self.fast_args = None

        # Expand corm_mask if needed
        if corm_mask.dim() == 2:
            if self.cached_corm_mask is None:
                self.cached_corm_mask = corm_mask.unsqueeze(0).unsqueeze(0).repeat(batch, self.head_num, 1, 1)
            corm_mask = self.cached_corm_mask

        # Transpose to BHSD for ComputeGraph (Metadata only)
        q_bhsd = q.transpose(1, 2)
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)

        if self.fast_args is None:
            # Slow path: build arguments and identify tensor positions
            global_inputs = {'Q': q_bhsd, 'K': k_bhsd, 'V': v_bhsd, 'Mask': self.cached_mask, 'CormMask': corm_mask}
            external_outputs = {
                "attn_output": self.output_tensor,
                "corm_score": self.corm_score_tensor
            }
            self.fast_args, _, _, _ = self.graph._build_fused_args_from_partition(
                self.best_partition, global_inputs, self.params_dict, device,
                external_outputs=external_outputs
            )

            # Identify where Q, K, V, Mask, CormMask, Outputs are in the argument list
            self.tensor_indices = {'Q': [], 'K': [], 'V': [], 'Mask': [], 'CormMask': [], 'Output': [], 'CormScore': []}
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
                        elif ptr == corm_mask.data_ptr():
                            self.tensor_indices['CormMask'].append((sg_idx, arg_idx))
                        elif ptr == self.output_tensor.data_ptr():
                            self.tensor_indices['Output'].append((sg_idx, arg_idx))
                        elif ptr == self.corm_score_tensor.data_ptr():
                            self.tensor_indices['CormScore'].append((sg_idx, arg_idx))
        else:
            # Fast path: just update tensor pointers
            for name, tensor in [('Q', q_bhsd), ('K', k_bhsd), ('V', v_bhsd), ('Mask', self.cached_mask), ('CormMask', corm_mask), ('Output', self.output_tensor), ('CormScore', self.corm_score_tensor)]:
                for sg_idx, arg_idx in self.tensor_indices[name]:
                    self.fast_args[sg_idx][arg_idx] = tensor

        # Execute kernels
        for (compiled, grid), args in zip(self.best_compiled, self.fast_args):
            compiled[grid](*args)

        # Return both outputs
        return self.output_tensor.transpose(1, 2), self.corm_score_tensor


def compile(model, input_names, inputs, output_names, system, configs=None):
    # Extract q, k, v, corm_mask from inputs
    q = inputs[0]
    k = inputs[1]
    v = inputs[2]
    corm_mask = inputs[3]
    batch, seq, head, dim = q.shape

    print(f"Compiling 'our' system for B={batch}, S={seq}, H={head}, D={dim}")

    # Setup ComputeGraph and search
    graph = ComputeGraph("corm_attention")
    graph.add_input("Q", "K", "V", "Mask", "CormMask")
    scale = 1.0 / math.sqrt(dim)

    common_params_mk = {"M": seq, "N": seq, "K": dim, "BM": 64, "BN": 64, "batch": batch, "heads": head}
    common_params_m = {"M": seq, "N": seq, "BM": 64, "BN": 64, "batch": batch, "heads": head}
    common_params_loopk = {"M": seq, "N": dim, "K": seq, "BM": 64, "BN": 64, "batch": batch, "heads": head}

    # Node indices: 0=Q, 1=K, 2=V, 3=Mask, 4=CormMask
    # 5=gemm_qk, 6=scale, 7=add_mask, 8=softmax_store, 9=gemm_pv, 10=mask_any
    graph.add_node("gemm_qk", inputs={"Q": "q_ptr", "K": "k_ptr"}, parents=[0, 1], **common_params_mk) \
        .add_node("scale_c", inputs={"scores": "input"}, parents=[5], **common_params_mk, scale=scale) \
        .add_node("add_mask", inputs={"output": "input", "Mask": "mask_ptr"}, parents=[6, 3], **common_params_m) \
        .add_node("softmax_store", inputs={"output": "input"}, parents=[7], **common_params_m) \
        .add_node("gemm_pv", inputs={"output": "probs", "V": "v_ptr"}, parents=[8, 2], **common_params_loopk) \
        .add_node("mask_any", inputs={"output": "input", "CormMask": "mask"}, parents=[8, 4], **common_params_m) \
        .add_output("attn_output", source=9, shape=(batch, head, seq, dim)) \
        .add_output("corm_score", source=10, shape=(batch, head, seq), dtype=torch.bool)

    mask = torch.zeros(batch, head, seq, seq, device=q.device, dtype=q.dtype)
    # Expand corm_mask if needed
    if corm_mask.dim() == 2:
        corm_mask_expanded = corm_mask.unsqueeze(0).unsqueeze(0).repeat(batch, head, 1, 1)
    else:
        corm_mask_expanded = corm_mask

    global_inputs = {
        'Q': q.transpose(1, 2),
        'K': k.transpose(1, 2),
        'V': v.transpose(1, 2),
        'Mask': mask,
        'CormMask': corm_mask_expanded
    }
    params_dict = {"M": seq, "N": seq, "K": dim, "batch": batch, "heads": head, "scale": scale}
    search_optimal = True if configs is None else False
    print(f"Starting compilation with search_optimal={search_optimal}...")
    best_compiled, best_partition, _ = graph.compile(
        global_inputs=global_inputs, params_dict=params_dict, device=q.device, num_warmup=4, num_stages=3, 
        search_optimal=search_optimal, 
        # configs=configs
    )

    # Create pre-allocated output tensors
    output_tensors = {
        "attn_output": torch.empty((batch, seq, head, dim), device=q.device, dtype=q.dtype),
        "corm_score": torch.empty((batch, head, seq), device=q.device, dtype=torch.bool)
    }

    return FusedCormOp(head, dim, graph, best_compiled, best_partition, params_dict, output_tensors=output_tensors)


class Corm(nn.Module):
    def __init__(self, kv_head_num=32, head_num=32, head_dim=128):
        super().__init__()
        self.kv_head_num = kv_head_num
        self.head_num = head_num
        self.head_dim = head_dim
        self.hidden_size = head_num * head_dim
        self.kv_hidden_size = kv_head_num * head_dim
        self.group_size = self.head_num // self.kv_head_num
        assert self.group_size == 1

    def forward(self, q, k, v, corm_mask):
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

        # corm
        corm_score = probs
        corm_score = probs >= corm_mask
        corm_score = corm_score.any(dim=2)

        return out.view(batch_size, q_len, self.head_num, self.head_dim), corm_score.view(batch_size, self.kv_head_num, kv_len)

    def prepare(self, batch_size=1, q_len=4096, kv_len=4096, dtype=torch.float16, device=torch.cuda.current_device()):
        q = torch.randn(batch_size, q_len, self.head_num, self.head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, kv_len, self.head_num, self.head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, kv_len, self.head_num, self.head_dim, dtype=dtype, device=device)

        corm_mask = torch.ones(q_len, kv_len, dtype=torch.float16, device=device)
        for i in range(q_len):
            corm_mask[i] /= i + 1

        ret = {
            'input': {
                'q': q,
                'k': k,
                'v': v,
                'corm_mask': corm_mask
            },
            'output': ['out', 'corm_score']
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

        self.attn_f = attn_f
        self.corm_mask = hf_config.corm_mask

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

        out, corm_score = self.attn_f(
            q.view(q.shape[0], q.shape[1], self.head_num, self.head_dim),
            k.view(k.shape[0], k.shape[1], self.kv_head_num, self.head_dim),
            v.view(v.shape[0], v.shape[1], self.kv_head_num, self.head_dim),
            self.corm_mask,
        )
        out = out.reshape(out.shape[0], out.shape[1], -1)

        selected_indices = torch.nonzero(corm_score)
        kv_cache = torch.cat([k, v], dim=1)
        kv_caches.append((corm_score, selected_indices, kv_cache))
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
# 4. Setup and Main
# =============================================================================

def llm_setup(weight_path, seqlen, layer_num):
    device = torch.cuda.current_device()

    hf_config = AutoConfig.from_pretrained(weight_path)
    cache_budget = 512
    assert cache_budget < seqlen
    hf_config.cache_budget = cache_budget
    hf_config.roco_recent = 256
    hf_config.tau = 1.5
    corm_mask = torch.ones(seqlen, seqlen, dtype=torch.float32, device=device)
    for i in range(seqlen):
        corm_mask[i] /= i + 1
    hf_config.corm_mask = corm_mask

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
    print(f"{token_ids.grad=}")

    return hf_config, token_ids


@click.command()
@click.option('--model', '-m', default='corm', help='Model name')
@click.option('--system', '-s', default='our', help='System name (our, torch, flashtensor)')
@click.option('--seqlen', type=int, default=2048, help='seqlen')
@click.option('--layer_num', type=int, default=None, help='layer_num')
@click.option('--platform', '-p', default='H800', help='platform(H800, A100, H100)')
@click.option('--mode', default='kernel', type=click.Choice(['kernel', 'both']), help='Benchmark mode: kernel only or both kernel and E2E')
@click.option('--check', is_flag=True, help='Check correctness')
def main(model, system, seqlen, layer_num, platform, mode, check):
    print(f"{model=} {system=} {seqlen=} {layer_num=} {mode=} {check=}")
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Use Corm class for specification
    kernel_cls = Corm
    model_cls = Llama

    kernel = kernel_cls().eval().cuda()
    specs = kernel.prepare(q_len=seqlen, kv_len=seqlen)
    input_names = list(specs['input'].keys())
    inputs = [specs['input'][name] for name in input_names]
    output_names = specs['output']

    print(f"{input_names=}")
    print(f"{output_names=}")

    # Optionally load an external attention function from a Python file (skip kernel compilation if provided)
    kernel_f = None
    attn_callable = None
    if system == 'torch':
        attn_callable = kernel.forward
    elif system == 'flashtensor':
        from tilefusion.utils.flashtensor_wrapper import get_flashtensor_ext_path
        ext_path = get_flashtensor_ext_path(None, model=model, seqlen=seqlen)
        ext_func = 'Corm'
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
        def ext_attn_wrapper(q, k, v, corm_mask):
            return ext_fn(q, k, v, corm_mask)
        attn_callable = ext_attn_wrapper
    elif system == 'our':
        print("Compiling fused Corm kernel...", flush=True)
        test_configs = [{'BM': 64, 'BN': 64, 'num_warps': 4, 'num_stages': 3}] if platform == 'A100' else None
        kernel_f = compile(
            model=kernel,
            input_names=input_names,
            inputs=inputs,
            output_names=output_names,
            system=system,
            configs=test_configs,
        )
        attn_callable = kernel_f
    # Prepare attention inputs (from kernel prepare)
    q = specs['input']['q']
    k = specs['input']['k']
    v = specs['input']['v']
    corm_mask = specs['input']['corm_mask']
    # Ensure everything is ready on device
    torch.cuda.synchronize()

    # 0) Check correctness
    if check:
        print("Checking correctness...")
        with torch.no_grad():
            out_ref, corm_ref = kernel(q, k, v, corm_mask)
            out_test, corm_test = attn_callable(q, k, v, corm_mask)
            torch.testing.assert_close(out_test, out_ref, rtol=1e-3, atol=1e-2)
            torch.testing.assert_close(corm_test.float(), corm_ref.float(), rtol=1e-3, atol=1e-2)
            print("Correctness check passed!")

    # 1) Benchmark attention kernel only (skip when external function provided)

    def run_kernel():
        # call compiled kernel via kernel_f
        _ = attn_callable(q, k, v, corm_mask)
    torch.cuda.synchronize()
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
