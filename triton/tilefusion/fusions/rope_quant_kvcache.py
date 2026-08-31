import json
import math
import time
import tempfile
import sys
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.compiler import compile as triton_compile

# Add CUDA extension path
_mla_rope_dir = os.path.join(os.path.dirname(__file__), "..", "ops", "csrc", "mla_rope")
sys.path.insert(0, _mla_rope_dir)
import mla_rope_quant_cuda  # mla_rope_quant_fused.cu: rope + fp8 quant + kvcache (uint8 output)

# Use local module imports (same style as other demos)
from tilefusion.utils.utils import DEVICE, build_combined_module, benchmark_performance, validate_correctness
from tilefusion.core.compiler import fuse_kernels_in_ttir, opt_kernels_in_ttir
from tilefusion.ops.mla_rope_ops import (
    _ttir_of_rope_q, _ttir_of_rope_k,
    rope_apply_q_kernel, rope_apply_k_kernel,
)


# =============================================================================
# Triton Kernel: KV-Cache Write with FP8 Static Per-Tensor Quantization
# This extends kv_cache_write_kernel from mla_rope_ops.py to support fp8 quant.
# kv_cache is stored as uint8 (fp8 e4m3); scale = 1 / max_val.
# =============================================================================

@triton.jit
def kv_cache_quant_write_kernel(
    k_pe_ptr,               # [num_tokens, rot_dim]        fp16/bf16
    kv_c_ptr,               # [num_tokens, kv_lora_rank]   fp16/bf16
    kv_cache_ptr,           # [num_blocks, block_size, total_dim]  uint8 (fp8)
    slot_mapping_ptr,       # [num_tokens]
    kv_cache_quant_scale_ptr,  # scalar, float32
    num_tokens,
    rot_dim,
    kv_lora_rank,
    block_size,
    k_pe_stride_token,
    k_pe_stride_dim,
    kv_c_stride_token,
    kv_c_stride_dim,
    block_stride,
    entry_stride,
    IS_NEOX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Write K_PE and KV_C to KV cache with FP8 static per-tensor quantization.

    kv_cache is uint8 (fp8 e4m3).  Values are quantized as:
        q = clamp(round(x * scale), -128, 127)
    where scale = kv_cache_quant_scale (1 / max_val).
    """
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    block_idx = slot_idx // block_size
    entry_idx = slot_idx % block_size

    kv_cache_entry_ptr = kv_cache_ptr + block_idx * block_stride + entry_idx * entry_stride

    scale = tl.load(kv_cache_quant_scale_ptr)
    embed_dim = rot_dim // 2

    # Write K_PE (rope part) with fp8 quant
    # Loop over embed_dim, write pairs (x, y) matching rope_apply_k_kernel's index layout.
    # IS_NEOX=True:  idx_x = pair_idx,     idx_y = embed_dim + pair_idx
    # IS_NEOX=False: idx_x = pair_idx * 2, idx_y = pair_idx * 2 + 1
    k_pe_token_ptr = k_pe_ptr + token_idx * k_pe_stride_token
    kv_cache_k_ptr = kv_cache_entry_ptr + kv_lora_rank
    for i in range(0, embed_dim, BLOCK_SIZE):
        pair_offs = i + tl.arange(0, BLOCK_SIZE)
        mask = pair_offs < embed_dim
        if IS_NEOX:
            idx_x = pair_offs
            idx_y = embed_dim + pair_offs
        else:
            idx_x = pair_offs * 2
            idx_y = pair_offs * 2 + 1
        x_vals = tl.load(k_pe_token_ptr + idx_x * k_pe_stride_dim, mask=mask, other=0.0)
        y_vals = tl.load(k_pe_token_ptr + idx_y * k_pe_stride_dim, mask=mask, other=0.0)
        x_q = tl.clamp(tl.extra.cuda.libdevice.round(x_vals.to(tl.float32) * scale), -128.0, 127.0).to(tl.int8)
        y_q = tl.clamp(tl.extra.cuda.libdevice.round(y_vals.to(tl.float32) * scale), -128.0, 127.0).to(tl.int8)
        tl.store(kv_cache_k_ptr + idx_x, x_q, mask=mask)
        tl.store(kv_cache_k_ptr + idx_y, y_q, mask=mask)

    # Write KV_C (nope part) with fp8 quant
    kv_c_token_ptr = kv_c_ptr + token_idx * kv_c_stride_token
    for i in range(0, kv_lora_rank, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < kv_lora_rank
        vals = tl.load(kv_c_token_ptr + offs * kv_c_stride_dim, mask=mask, other=0.0)
        qvals = tl.clamp(tl.extra.cuda.libdevice.round(vals.to(tl.float32) * scale), -128.0, 127.0).to(tl.int8)
        tl.store(kv_cache_entry_ptr + offs, qvals, mask=mask)


# =============================================================================
# Torch vectorized baseline (no per-token Python loop)
# =============================================================================

def torch_baseline(
    positions,              # [num_tokens]
    q_pe,                   # [num_tokens, num_q_heads, rot_dim]  (in-place modified)
    k_pe,                   # [num_tokens, rot_dim]               (in-place modified)
    kv_c,                   # [num_tokens, kv_lora_rank]
    rope_cos_sin_cache,     # [max_position, rot_dim]
    rope_is_neox,
    kv_cache_slot_mapping,  # [num_tokens]
    kv_cache,               # [num_blocks, block_size, kv_lora_rank + rot_dim] uint8
    kv_cache_quant_scale,   # scalar tensor, float32
):
    """Vectorized PyTorch baseline: RoPE + FP8 quant + KV-cache write.

    All operations are batched over num_tokens; no Python-level token loop.
    """
    rot_dim = q_pe.shape[2]
    embed_dim = rot_dim // 2
    kv_lora_rank = kv_c.shape[1]
    block_size = kv_cache.shape[1]
    scale = kv_cache_quant_scale.item()

    # Gather cos/sin for each token: [num_tokens, embed_dim]
    cos = rope_cos_sin_cache[positions, :embed_dim]          # [T, embed_dim]
    sin = rope_cos_sin_cache[positions, embed_dim:]          # [T, embed_dim]

    # --- RoPE for q_pe [T, H, rot_dim] ---
    # Broadcast cos/sin over heads: [T, 1, embed_dim]
    cos_q = cos.unsqueeze(1)
    sin_q = sin.unsqueeze(1)
    if rope_is_neox:
        q1 = q_pe[..., :embed_dim].clone()
        q2 = q_pe[..., embed_dim:].clone()
        q_pe[..., :embed_dim] = q1 * cos_q - q2 * sin_q
        q_pe[..., embed_dim:] = q2 * cos_q + q1 * sin_q
    else:
        q1 = q_pe[..., ::2].clone()
        q2 = q_pe[..., 1::2].clone()
        q_pe[..., ::2]  = q1 * cos_q - q2 * sin_q
        q_pe[..., 1::2] = q2 * cos_q + q1 * sin_q

    # --- RoPE for k_pe [T, rot_dim] ---
    if rope_is_neox:
        k1 = k_pe[..., :embed_dim].clone()
        k2 = k_pe[..., embed_dim:].clone()
        k_pe[..., :embed_dim] = k1 * cos - k2 * sin
        k_pe[..., embed_dim:] = k2 * cos + k1 * sin
    else:
        k1 = k_pe[..., ::2].clone()
        k2 = k_pe[..., 1::2].clone()
        k_pe[..., ::2]  = k1 * cos - k2 * sin
        k_pe[..., 1::2] = k2 * cos + k1 * sin

    # --- FP8 static per-tensor quantization ---
    def quantize(t):
        return t.float().mul(scale).round_().clamp_(-128, 127).to(torch.int8)

    kv_c_q = quantize(kv_c)   # [T, kv_lora_rank]
    k_pe_q = quantize(k_pe)   # [T, rot_dim]
    entry = torch.cat([kv_c_q, k_pe_q], dim=-1)  # [T, kv_lora_rank + rot_dim]

    # --- Scatter to KV-cache using slot_mapping ---
    block_idx = kv_cache_slot_mapping // block_size   # [T]
    entry_idx = kv_cache_slot_mapping % block_size    # [T]
    # kv_cache view as int8 for writing (underlying bytes identical)
    kv_cache_i8 = kv_cache.view(torch.int8)
    kv_cache_i8[block_idx, entry_idx] = entry


# =============================================================================
# Dynamo (torch.compile) baseline
# =============================================================================

# The compiled function is cached after first call (compilation happens lazily).
_torch_baseline_compiled = None


def _make_dynamo_fn():
    """Wrap torch_baseline for torch.compile.

    torch.compile works best on functions with Tensor arguments only.
    We wrap it to avoid non-Tensor scalars where possible.
    """
    @torch.compile(fullgraph=False, dynamic=True)
    def _compiled(
        positions, q_pe, k_pe, kv_c,
        rope_cos_sin_cache, kv_cache_slot_mapping,
        kv_cache, kv_cache_quant_scale, rope_is_neox_tensor,
    ):
        rot_dim = q_pe.shape[2]
        embed_dim = rot_dim // 2
        kv_lora_rank = kv_c.shape[1]
        block_size = kv_cache.shape[1]
        scale = kv_cache_quant_scale.item()

        cos = rope_cos_sin_cache[positions, :embed_dim]
        sin = rope_cos_sin_cache[positions, embed_dim:]

        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        # Neox-style (rope_is_neox=True)
        q1 = q_pe[..., :embed_dim].clone()
        q2 = q_pe[..., embed_dim:].clone()
        q_pe = torch.cat([q1 * cos_q - q2 * sin_q, q2 * cos_q + q1 * sin_q], dim=-1)

        k1 = k_pe[..., :embed_dim].clone()
        k2 = k_pe[..., embed_dim:].clone()
        k_pe = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        kv_c_q  = kv_c.float().mul(scale).round().clamp(-128, 127).to(torch.int8)
        k_pe_q  = k_pe.float().mul(scale).round().clamp(-128, 127).to(torch.int8)
        entry   = torch.cat([kv_c_q, k_pe_q], dim=-1)

        block_idx = kv_cache_slot_mapping // block_size
        entry_idx = kv_cache_slot_mapping % block_size
        kv_cache_i8 = kv_cache.view(torch.int8)
        kv_cache_i8[block_idx, entry_idx] = entry
        return q_pe, k_pe, kv_cache
    return _compiled


def dynamo_baseline(
    positions, q_pe, k_pe, kv_c,
    rope_cos_sin_cache, rope_is_neox,
    kv_cache_slot_mapping, kv_cache, kv_cache_quant_scale,
):
    """torch.compile baseline (Dynamo + inductor backend)."""
    global _torch_baseline_compiled
    if _torch_baseline_compiled is None:
        print("[Dynamo] Compiling torch_baseline with torch.compile...")
        _torch_baseline_compiled = _make_dynamo_fn()
    # rope_is_neox passed as dummy tensor (not used; function assumes neox=True)
    q_out, k_out, _ = _torch_baseline_compiled(
        positions, q_pe, k_pe, kv_c,
        rope_cos_sin_cache, kv_cache_slot_mapping,
        kv_cache, kv_cache_quant_scale,
        torch.zeros(1, device=q_pe.device),  # placeholder for rope_is_neox
    )
    # Copy RoPE results back to the input buffers (compiled fn returns new tensors)
    q_pe.copy_(q_out)
    k_pe.copy_(k_out)


# =============================================================================
# TensorRT baseline (ONNX export -> TRT engine, rope+quant only)
# =============================================================================

class _RopeQuantModule(torch.nn.Module):
    """Exportable module: RoPE (neox) + FP8 static quant.

    Inputs:  q_pe, k_pe, kv_c, cos, sin, scale
    Outputs: q_out, k_out, kv_c_entry, k_pe_entry  (all float16 for ONNX compat)

    KV-cache scatter is NOT exported to TRT (scatter with dynamic indices is
    poorly supported); it is performed in PyTorch after TRT inference.
    """
    def forward(self, q_pe, k_pe, kv_c, cos, sin, scale):
        # q_pe: [T, H, rot_dim], cos/sin: [T, embed_dim]
        embed_dim = cos.shape[-1]
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)

        q1 = q_pe[..., :embed_dim]
        q2 = q_pe[..., embed_dim:]
        q_out = torch.cat([q1 * cos_q - q2 * sin_q, q2 * cos_q + q1 * sin_q], dim=-1)

        k1 = k_pe[..., :embed_dim]
        k2 = k_pe[..., embed_dim:]
        k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        s = scale.item()
        kv_c_q = kv_c.float().mul(s).round().clamp(-128, 127).to(torch.float16)
        k_pe_q = k_out.float().mul(s).round().clamp(-128, 127).to(torch.float16)
        return q_out, k_out, kv_c_q, k_pe_q


class TensorRTBaseline:
    """TensorRT baseline: RoPE+quant via TRT engine, scatter via PyTorch."""

    def __init__(
        self,
        num_tokens, num_q_heads, rot_dim, kv_lora_rank,
        block_size, device, dtype=torch.float16,
    ):
        self.rot_dim    = rot_dim
        self.embed_dim  = rot_dim // 2
        self.kv_lora_rank = kv_lora_rank
        self.block_size = block_size
        self.device     = device
        self.context    = None
        self.engine     = None

        self._build_engine(num_tokens, num_q_heads, rot_dim, kv_lora_rank, device, dtype)

    def _build_engine(self, num_tokens, num_q_heads, rot_dim, kv_lora_rank, device, dtype):
        try:
            import tensorrt as trt
        except ImportError:
            print("[TensorRT] tensorrt not installed; TRT baseline disabled.")
            return

        import io
        embed_dim = rot_dim // 2
        mod = _RopeQuantModule().to(device).to(dtype).eval()

        dummy_q   = torch.randn(num_tokens, num_q_heads, rot_dim, device=device, dtype=dtype)
        dummy_k   = torch.randn(num_tokens, rot_dim, device=device, dtype=dtype)
        dummy_kvc = torch.randn(num_tokens, kv_lora_rank, device=device, dtype=dtype)
        dummy_cos = torch.randn(num_tokens, embed_dim, device=device, dtype=dtype)
        dummy_sin = torch.randn(num_tokens, embed_dim, device=device, dtype=dtype)
        dummy_scale = torch.tensor([1.0 / 448.0], device=device, dtype=torch.float32)

        buf = io.BytesIO()
        with torch.no_grad():
            torch.onnx.export(
                mod,
                (dummy_q, dummy_k, dummy_kvc, dummy_cos, dummy_sin, dummy_scale),
                buf,
                input_names=['q_pe', 'k_pe', 'kv_c', 'cos', 'sin', 'scale'],
                output_names=['q_out', 'k_out', 'kv_c_q', 'k_pe_q'],
                opset_version=17,
                do_constant_folding=True,
            )
        onnx_bytes = buf.getvalue()

        logger  = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        config  = builder.create_builder_config()
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser  = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_bytes):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("[TensorRT] Failed to parse ONNX model")

        config.set_flag(trt.BuilderFlag.FP16)
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("[TensorRT] Failed to build TRT engine")

        runtime = trt.Runtime(logger)
        self.engine  = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()

        # Pre-allocate output buffers
        self._out_q   = torch.empty_like(dummy_q)
        self._out_k   = torch.empty_like(dummy_k)
        self._out_kvc = torch.empty(num_tokens, kv_lora_rank, device=device, dtype=dtype)
        self._out_kpe = torch.empty(num_tokens, rot_dim, device=device, dtype=dtype)
        print("[TensorRT] Engine built successfully.")

    def __call__(
        self,
        positions, q_pe, k_pe, kv_c,
        rope_cos_sin_cache, rope_is_neox,
        kv_cache_slot_mapping, kv_cache, kv_cache_quant_scale,
    ):
        if self.context is None:
            raise RuntimeError("[TensorRT] Engine not built. Is tensorrt installed?")

        embed_dim = self.embed_dim
        cos = rope_cos_sin_cache[positions, :embed_dim].contiguous()
        sin = rope_cos_sin_cache[positions, embed_dim:].contiguous()

        q_pe_c  = q_pe.contiguous()
        k_pe_c  = k_pe.contiguous()
        kv_c_c  = kv_c.contiguous()
        scale_c = kv_cache_quant_scale.contiguous()

        # Bind by engine I/O name/order. Classic ONNX may constant-fold
        # scale.item() away, so "scale" is optional.
        name_to_ptr = {
            "q_pe": q_pe_c.data_ptr(),
            "k_pe": k_pe_c.data_ptr(),
            "kv_c": kv_c_c.data_ptr(),
            "cos": cos.data_ptr(),
            "sin": sin.data_ptr(),
            "scale": scale_c.data_ptr(),
            "q_out": self._out_q.data_ptr(),
            "k_out": self._out_k.data_ptr(),
            "kv_c_q": self._out_kvc.data_ptr(),
            "k_pe_q": self._out_kpe.data_ptr(),
        }
        bindings = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name not in name_to_ptr:
                raise RuntimeError(f"[TensorRT] unexpected I/O tensor: {name}")
            bindings.append(name_to_ptr[name])
        success = self.context.execute_v2(bindings)
        if not success:
            raise RuntimeError("[TensorRT] execute_v2 failed")

        # Scatter quantized values into KV cache (PyTorch, not TRT)
        kv_c_int8  = self._out_kvc.to(torch.int8)
        k_pe_int8  = self._out_kpe.to(torch.int8)
        entry      = torch.cat([kv_c_int8, k_pe_int8], dim=-1)
        block_idx  = kv_cache_slot_mapping // self.block_size
        entry_idx  = kv_cache_slot_mapping % self.block_size
        kv_cache.view(torch.int8)[block_idx, entry_idx] = entry

        # Copy RoPE outputs back to q_pe / k_pe
        q_pe.copy_(self._out_q)
        k_pe.copy_(self._out_k)


# =============================================================================
# Reference implementation
# =============================================================================

def rope_reference(x, cos, sin, is_neox=True):
    """Reference RoPE implementation in PyTorch"""
    rot_dim = x.shape[-1]
    embed_dim = rot_dim // 2

    if is_neox:
        x1, x2 = x[..., :embed_dim], x[..., embed_dim:]
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    if is_neox:
        return torch.cat([out1, out2], dim=-1)
    else:
        out = torch.stack([out1, out2], dim=-1)
        return out.flatten(-2)


def concat_and_cache_mla_rope_quant_reference(
    positions,              # [num_tokens]
    q_pe,                   # [num_tokens, num_q_heads, rot_dim]
    k_pe,                   # [num_tokens, rot_dim]
    kv_c,                   # [num_tokens, kv_lora_rank]
    rope_cos_sin_cache,     # [max_position, rot_dim]
    rope_is_neox,
    kv_cache_slot_mapping,  # [num_tokens]
    kv_cache,               # [num_blocks, block_size, kv_lora_rank + rot_dim] uint8
    kv_cache_quant_scale,   # scalar float
):
    """PyTorch reference: RoPE + FP8 static quant + KV-cache write"""
    num_tokens = q_pe.shape[0]
    rot_dim = q_pe.shape[2]
    embed_dim = rot_dim // 2
    kv_lora_rank = kv_c.shape[1]
    block_size = kv_cache.shape[1]
    scale = kv_cache_quant_scale

    for token_idx in range(num_tokens):
        pos = positions[token_idx].item()
        cos = rope_cos_sin_cache[pos, :embed_dim]
        sin = rope_cos_sin_cache[pos, embed_dim:]

        for head_idx in range(q_pe.shape[1]):
            q_pe[token_idx, head_idx] = rope_reference(
                q_pe[token_idx, head_idx], cos, sin, rope_is_neox
            )

        k_pe[token_idx] = rope_reference(k_pe[token_idx], cos, sin, rope_is_neox)

        slot_idx = kv_cache_slot_mapping[token_idx].item()
        block_idx = slot_idx // block_size
        entry_idx = slot_idx % block_size

        # Quantize kv_c and k_pe to int8 (fp8 proxy)
        def quantize(t):
            return t.float().mul(scale).round().clamp(-128, 127).to(torch.int8)

        kv_cache[block_idx, entry_idx, :kv_lora_rank] = quantize(kv_c[token_idx])
        kv_cache[block_idx, entry_idx, kv_lora_rank:] = quantize(k_pe[token_idx])

    return q_pe, k_pe, kv_cache


# =============================================================================
# TTIR Generator for quant cache write kernel
# =============================================================================

def _ttir_of_cache_quant_write(
    num_tokens: int,
    rot_dim: int,
    kv_lora_rank: int,
    num_blocks: int = 256,
    block_size: int = 16,
    BM: int = 64,
) -> str:
    """Generate TTIR for FP8-quantized cache write kernel"""
    k_pe = torch.randn(num_tokens, rot_dim, device=DEVICE, dtype=torch.float16)
    kv_c = torch.randn(num_tokens, kv_lora_rank, device=DEVICE, dtype=torch.float16)
    # kv_cache stored as uint8 (fp8)
    kv_cache = torch.zeros(
        num_blocks, block_size, kv_lora_rank + rot_dim,
        device=DEVICE, dtype=torch.int8
    )
    slot_mapping = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    kv_cache_quant_scale = torch.tensor([1.0 / 448.0], device=DEVICE, dtype=torch.float32)

    grid = (num_tokens,)

    compiled = kv_cache_quant_write_kernel[grid](
        k_pe, kv_c, kv_cache, slot_mapping, kv_cache_quant_scale,
        num_tokens, rot_dim, kv_lora_rank, block_size,
        k_pe.stride(0), k_pe.stride(1),
        kv_c.stride(0), kv_c.stride(1),
        kv_cache.stride(0), kv_cache.stride(1),
        IS_NEOX=True, BLOCK_SIZE=BM,
    )
    return compiled.asm['ttir']


# =============================================================================
# Triton unfused baseline (3 separate kernels)
# =============================================================================


def triton_baseline(
    q_pe,
    rope_cos_sin_cache,
    positions,
    num_tokens,
    num_q_heads,
    rot_dim,
    k_pe,
    kv_cache_slot_mapping,
    kv_c,
    kv_cache,
    kv_cache_quant_scale,
    kv_lora_rank,
    block_size,
    BM=64,
    num_warps=4,
    num_stages=2,
):
    IS_NEOX = True
    grid = (num_tokens, 1, 1)

    rope_apply_q_kernel[grid](
        q_pe, rope_cos_sin_cache, positions,
        num_tokens, num_q_heads, rot_dim,
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        rope_cos_sin_cache.stride(0), rope_cos_sin_cache.stride(1),
        IS_NEOX=IS_NEOX, BLOCK_SIZE=BM,
        num_warps=num_warps, num_stages=num_stages,
    )
    rope_apply_k_kernel[grid](
        k_pe, rope_cos_sin_cache, positions, kv_cache_slot_mapping,
        num_tokens, rot_dim,
        k_pe.stride(0), k_pe.stride(1),
        rope_cos_sin_cache.stride(0), rope_cos_sin_cache.stride(1),
        IS_NEOX=IS_NEOX, BLOCK_SIZE=BM,
        num_warps=num_warps, num_stages=num_stages,
    )
    kv_cache_quant_write_kernel[grid](
        k_pe, kv_c, kv_cache, kv_cache_slot_mapping, kv_cache_quant_scale,
        num_tokens, rot_dim, kv_lora_rank, block_size,
        k_pe.stride(0), k_pe.stride(1),
        kv_c.stride(0), kv_c.stride(1),
        kv_cache.stride(0), kv_cache.stride(1),
        IS_NEOX=IS_NEOX, BLOCK_SIZE=BM,
        num_warps=num_warps, num_stages=num_stages,
    )


# =============================================================================
# Fused kernel builder (rope_q -> rope_k -> cache_quant_write)
# =============================================================================

def build_and_compile_fused_kernel(
    num_tokens, num_q_heads, rot_dim, kv_lora_rank,
    num_blocks, block_size, BM, num_warps=4, num_stages=2,
):
    current_ttir = _ttir_of_rope_q(num_tokens, num_q_heads, rot_dim, BM)
    current_producer_name = "rope_apply_q_kernel"

    stages = [
        # Stage 1: rope_q -> rope_k
        ("rope_apply_k_kernel",
         lambda: _ttir_of_rope_k(num_tokens, rot_dim, BM),
         [1, 2, 3, 5, 8], [1, 2, 4, 5, 7]),
        # Stage 2: rope_k -> kv_cache_quant_write
        ("kv_cache_quant_write_kernel",
         lambda: _ttir_of_cache_quant_write(num_tokens, rot_dim, kv_lora_rank, num_blocks, block_size, BM),
         [9, 10, 3, 5, 11], [0, 3, 5, 6, 9]),
    ]

    fused_path = None
    for consumer_kernel_name, ttir_fn, prod_out_idx, cons_in_idx in stages:
        consumer_ttir = ttir_fn()
        combined = build_combined_module(
            current_ttir, consumer_ttir,
            current_producer_name, consumer_kernel_name,
        )
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
            f.write(combined)
            combined_path = f.name
        fused_path, fused_ttir_text = fuse_kernels_in_ttir(
            combined_path,
            producer_kernel_name=current_producer_name,
            consumer_kernel_name=consumer_kernel_name,
            producer_output_arg_idx=prod_out_idx,
            consumer_input_arg_idx=cons_in_idx,
        )
        current_ttir = fused_ttir_text
        current_producer_name = f"{current_producer_name}_{consumer_kernel_name}"

    fused_path, fused_ttir_text = opt_kernels_in_ttir(fused_path, enable_rle=True)
    options = {'num_warps': num_warps, 'num_stages': num_stages}
    compiled = triton_compile(fused_path, options=options)
    grid = (num_tokens, 1, 1)
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(compiled.asm['ttgir'])
        print(f"Fused module written to: {f.name}")
    return compiled, grid


# =============================================================================
# Autotuner: search best (BM, num_warps, num_stages) for each shape
# =============================================================================

def autotune_fused_kernel(
    num_tokens, num_q_heads, rot_dim, kv_lora_rank,
    num_blocks, block_size, rope_is_neox,
    bm_choices=None,
    num_warps_choices=None,
    num_stages_choices=None,
    warmup=5,
    rep=20,
):
    """Grid search over (BM, num_warps, num_stages) and return the best config.

    Returns
    -------
    best_cfg : dict  {BM, num_warps, num_stages, latency_ms}
    all_results : list of dicts sorted by latency
    """
    if bm_choices is None:
        bm_choices = [32, 64, 128]
    if num_warps_choices is None:
        num_warps_choices = [4, 8, 16]
    if num_stages_choices is None:
        num_stages_choices = [1, 2, 3,]

    inputs = create_test_inputs(
        num_tokens=num_tokens, num_q_heads=num_q_heads,
        rot_dim=rot_dim, kv_lora_rank=kv_lora_rank,
        num_blocks=num_blocks, block_size=block_size,
    )
    IS_NEOX = rope_is_neox

    results = []
    total = len(bm_choices) * len(num_warps_choices) * len(num_stages_choices)
    done = 0

    for BM in bm_choices:
        for nw in num_warps_choices:
            for ns in num_stages_choices:
                done += 1
                tag = f"BM={BM:3d} num_warps={nw:2d} num_stages={ns}"
                try:
                    compiled, grid = build_and_compile_fused_kernel(
                        num_tokens, num_q_heads, rot_dim, kv_lora_rank,
                        num_blocks, block_size, BM,
                        num_warps=nw, num_stages=ns,
                    )
                except Exception as e:
                    print(f"  [{done:3d}/{total}] {tag}  COMPILE FAIL: {e}")
                    continue

                # Build launcher
                def make_launcher(compiled, grid, inputs, kv_lora_rank, block_size):
                    def launcher():
                        q = inputs['q_pe'].clone()
                        k = inputs['k_pe'].clone()
                        c = inputs['kv_cache'].clone()
                        args = [
                            q, inputs['rope_cos_sin_cache'], inputs['positions'],
                            num_tokens, num_q_heads, rot_dim,
                            q.stride(0), q.stride(1),
                            inputs['rope_cos_sin_cache'].stride(0),
                            k, inputs['kv_cache_slot_mapping'], k.stride(0),
                            inputs['kv_c'], c, inputs['kv_cache_quant_scale'],
                            kv_lora_rank, block_size,
                            inputs['kv_c'].stride(0),
                            c.stride(0), c.stride(1),
                        ]
                        compiled[grid](*args)
                    return launcher

                launcher = make_launcher(compiled, grid, inputs, kv_lora_rank, block_size)

                # Warmup
                try:
                    for _ in range(warmup):
                        launcher()
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"  [{done:3d}/{total}] {tag}  RUN FAIL: {e}")
                    continue

                # Measure
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(rep):
                    launcher()
                end.record()
                torch.cuda.synchronize()
                ms = start.elapsed_time(end) / rep

                print(f"  [{done:3d}/{total}] {tag}  {ms:.4f} ms")
                results.append({
                    "BM": BM, "num_warps": nw, "num_stages": ns,
                    "latency_ms": round(ms, 4),
                })

    if not results:
        print("  All configs failed!")
        return None, []

    results.sort(key=lambda x: x["latency_ms"])
    best = results[0]
    print(f"\n  Best config: BM={best['BM']} num_warps={best['num_warps']} "
          f"num_stages={best['num_stages']} -> {best['latency_ms']:.4f} ms")
    return best, results


# =============================================================================
# Test input factory
# =============================================================================

def create_test_inputs(
    num_tokens=128,
    num_q_heads=32,
    rot_dim=64,
    kv_lora_rank=512,
    max_position=4096,
    num_blocks=256,
    block_size=16,
    kv_scale=1.0 / 448.0,   # FP8 E4M3 max = 448
    device=DEVICE,
    dtype=torch.float16,
):
    """Create test inputs for rope + quant + kvcache fusion"""
    positions = torch.randint(0, max_position, (num_tokens,), device=device, dtype=torch.int64)

    q_pe = 0.5 * torch.randn(num_tokens, num_q_heads, rot_dim, device=device, dtype=dtype)
    k_pe = 0.5 * torch.randn(num_tokens, rot_dim, device=device, dtype=dtype)
    kv_c = 0.5 * torch.randn(num_tokens, kv_lora_rank, device=device, dtype=dtype)

    t = torch.arange(max_position, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (
        torch.arange(0, rot_dim, 2, device=device, dtype=torch.float32) / rot_dim
    ))
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    rope_cos_sin_cache = torch.cat([cos, sin], dim=-1)  # [max_position, rot_dim]

    import random
    total_slots = num_blocks * block_size
    assert total_slots >= num_tokens, "Not enough kv slots!"
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    kv_cache_slot_mapping = torch.tensor(slot_mapping_lst, device=device, dtype=torch.int64)

    # kv_cache stored as uint8 (fp8 e4m3 proxy)
    kv_cache = torch.zeros(
        num_blocks, block_size, kv_lora_rank + rot_dim,
        device=device, dtype=torch.uint8,
    )

    kv_cache_quant_scale = torch.tensor([kv_scale], device=device, dtype=torch.float32)

    return {
        'positions': positions,
        'q_pe': q_pe,
        'k_pe': k_pe,
        'kv_c': kv_c,
        'rope_cos_sin_cache': rope_cos_sin_cache,
        'kv_cache_slot_mapping': kv_cache_slot_mapping,
        'kv_cache': kv_cache,
        'kv_cache_quant_scale': kv_cache_quant_scale,
    }


def get_hw_name():
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0).replace(" ", "_")


# =============================================================================
# Plot results
# =============================================================================

def plot_results(all_records, hw_name, token_sizes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    methods = ["CUDA", "Triton_fused", "Triton_baseline", "Torch", "Dynamo", "TensorRT"]
    colors  = {"CUDA": "#1f77b4", "Triton_fused": "#ff7f0e", "Triton_baseline": "#2ca02c",
               "Torch": "#d62728", "Dynamo": "#9467bd", "TensorRT": "#8c564b"}
    markers = {"CUDA": "o", "Triton_fused": "s", "Triton_baseline": "^",
               "Torch": "D", "Dynamo": "v", "TensorRT": "P"}

    # Collect timings: {method: {num_tokens: ms}}
    timings = {m: {} for m in methods}
    for rec in all_records:
        m = rec["method"]
        t = rec["seqlen"]
        if m in timings:
            timings[m][t] = rec["avg"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"MLA RoPE + FP8 Quant + KV-Cache Fusion\n{hw_name}", fontsize=13)

    # --- Left: latency (ms) ---
    ax = axes[0]
    for m in methods:
        xs = sorted(timings[m].keys())
        ys = [timings[m][x] for x in xs]
        ax.plot(xs, ys, label=m, color=colors[m], marker=markers[m], linewidth=2, markersize=6)
    ax.set_xlabel("num_tokens")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs num_tokens")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xscale("log", base=2)

    # --- Right: speedup over CUDA ---
    ax = axes[1]
    for m in methods:
        if m == "CUDA":
            continue
        xs = sorted(timings[m].keys())
        ys = []
        for x in xs:
            cuda_ms = timings["CUDA"].get(x)
            this_ms = timings[m].get(x)
            if cuda_ms and this_ms:
                ys.append(cuda_ms / this_ms)
            else:
                ys.append(None)
        ax.plot(xs, ys, label=f"{m} / CUDA", color=colors[m], marker=markers[m], linewidth=2, markersize=6)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1, label="CUDA baseline (1.0×)")
    ax.set_xlabel("num_tokens")
    ax.set_ylabel("Speedup over CUDA")
    ax.set_title("Speedup vs num_tokens")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xscale("log", base=2)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "../result/rope_quant_kvcache_results.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")


# =============================================================================
# Single config benchmark + validation
# =============================================================================

def run_single_config(
    num_tokens, num_q_heads, rot_dim, kv_lora_rank,
    num_blocks, block_size, rope_is_neox, hw_name,
):
    print(f"\n{'=' * 70}")
    print(f"num_tokens={num_tokens}, num_q_heads={num_q_heads}, "
          f"rot_dim={rot_dim}, kv_lora_rank={kv_lora_rank}")
    print(f"{'=' * 70}")

    inputs = create_test_inputs(
        num_tokens=num_tokens, num_q_heads=num_q_heads,
        rot_dim=rot_dim, kv_lora_rank=kv_lora_rank,
        num_blocks=num_blocks, block_size=block_size,
    )

    # Step 1: Autotune fused kernel to find best (BM, num_warps, num_stages)
    print(f"\n[Autotune] Searching best config for num_tokens={num_tokens}...")
    best_cfg, _ = autotune_fused_kernel(
        num_tokens, num_q_heads, rot_dim, kv_lora_rank,
        num_blocks, block_size, rope_is_neox,
    )
    if best_cfg is None:
        print("  Autotune failed, falling back to default config")
        best_cfg = {"BM": 64, "num_warps": 4, "num_stages": 2}

    BM         = best_cfg["BM"]
    num_warps  = best_cfg["num_warps"]
    num_stages = best_cfg["num_stages"]
    print(f"[Autotune] Best config: BM={BM}, num_warps={num_warps}, num_stages={num_stages}")

    # Step 2: Compile fused kernel with the best config
    compiled, grid = build_and_compile_fused_kernel(
        num_tokens, num_q_heads, rot_dim, kv_lora_rank,
        num_blocks, block_size, BM=BM,
        num_warps=num_warps, num_stages=num_stages,
    )

    # --- CUDA reference launcher (mla_rope_quant_cuda: rope + fp8 quant, uint8 kv_cache) ---
    q_pe_cuda = inputs['q_pe'].clone()
    k_pe_cuda = inputs['k_pe'].clone()
    kv_cache_cuda = torch.zeros(
        inputs['kv_cache'].shape[0], inputs['kv_cache'].shape[1],
        inputs['kv_cache'].shape[2],
        device=DEVICE, dtype=torch.uint8,
    )

    def cuda_launcher():
        mla_rope_quant_cuda.concat_and_cache_mla_rope_quant_fused(
            inputs['positions'],
            q_pe_cuda,
            k_pe_cuda,
            inputs['kv_c'],
            inputs['rope_cos_sin_cache'],
            rope_is_neox,
            inputs['kv_cache_slot_mapping'],
            kv_cache_cuda,
            inputs['kv_cache_quant_scale'],
        )

    def fresh_cuda_launcher():
        q = inputs['q_pe'].clone()
        k = inputs['k_pe'].clone()
        c = torch.zeros(
            inputs['kv_cache'].shape[0], inputs['kv_cache'].shape[1],
            inputs['kv_cache'].shape[2],
            device=DEVICE, dtype=torch.uint8,
        )
        mla_rope_quant_cuda.concat_and_cache_mla_rope_quant_fused(
            inputs['positions'], q, k, inputs['kv_c'],
            inputs['rope_cos_sin_cache'], rope_is_neox,
            inputs['kv_cache_slot_mapping'], c,
            inputs['kv_cache_quant_scale'],
        )

    # --- Triton baseline launcher ---
    def triton_baseline_launcher():
        q = inputs['q_pe'].clone()
        k = inputs['k_pe'].clone()
        c = inputs['kv_cache'].clone()
        triton_baseline(
            q, inputs['rope_cos_sin_cache'], inputs['positions'],
            num_tokens, num_q_heads, rot_dim,
            k, inputs['kv_cache_slot_mapping'],
            inputs['kv_c'], c, inputs['kv_cache_quant_scale'],
            kv_lora_rank, block_size,
            BM=BM, num_warps=num_warps, num_stages=num_stages,
        )

    # --- Triton fused launcher ---

    def fresh_triton_fused_launcher():
        q = inputs['q_pe'].clone()
        k = inputs['k_pe'].clone()
        c = inputs['kv_cache'].clone()
        args = [
            q, inputs['rope_cos_sin_cache'], inputs['positions'],
            num_tokens, num_q_heads, rot_dim,
            q.stride(0), q.stride(1), inputs['rope_cos_sin_cache'].stride(0),
            k, inputs['kv_cache_slot_mapping'], k.stride(0),
            inputs['kv_c'], c, inputs['kv_cache_quant_scale'],
            kv_lora_rank, block_size,
            inputs['kv_c'].stride(0),
            c.stride(0), c.stride(1),
        ]
        compiled[grid](*args)

    # --- Torch baseline launcher ---
    def fresh_torch_launcher():
        q = inputs['q_pe'].clone()
        k = inputs['k_pe'].clone()
        c = inputs['kv_cache'].clone()
        torch_baseline(
            inputs['positions'], q, k, inputs['kv_c'],
            inputs['rope_cos_sin_cache'], rope_is_neox,
            inputs['kv_cache_slot_mapping'], c, inputs['kv_cache_quant_scale'],
        )

    # --- Dynamo (torch.compile) launcher ---
    # Trigger compilation before benchmarking
    print("\n[Dynamo] Triggering torch.compile warm-up...")
    _dynamo_warmup_q = inputs['q_pe'].clone()
    _dynamo_warmup_k = inputs['k_pe'].clone()
    _dynamo_warmup_c = inputs['kv_cache'].clone()
    dynamo_baseline(
        inputs['positions'], _dynamo_warmup_q, _dynamo_warmup_k, inputs['kv_c'],
        inputs['rope_cos_sin_cache'], rope_is_neox,
        inputs['kv_cache_slot_mapping'], _dynamo_warmup_c, inputs['kv_cache_quant_scale'],
    )
    del _dynamo_warmup_q, _dynamo_warmup_k, _dynamo_warmup_c

    def fresh_dynamo_launcher():
        q = inputs['q_pe'].clone()
        k = inputs['k_pe'].clone()
        c = inputs['kv_cache'].clone()
        dynamo_baseline(
            inputs['positions'], q, k, inputs['kv_c'],
            inputs['rope_cos_sin_cache'], rope_is_neox,
            inputs['kv_cache_slot_mapping'], c, inputs['kv_cache_quant_scale'],
        )

    # --- TensorRT baseline ---
    trt_runner = None
    trt_available = False
    try:
        import tensorrt  # noqa: F401
        print("\n[TensorRT] Building TRT engine...")
        trt_runner = TensorRTBaseline(
            num_tokens, num_q_heads, rot_dim, kv_lora_rank,
            block_size, device=DEVICE,
        )
        trt_available = True
    except (ImportError, RuntimeError) as e:
        print(f"\n[TensorRT] Skipped: {e}")

    def fresh_trt_launcher():
        q = inputs['q_pe'].clone()
        k = inputs['k_pe'].clone()
        c = inputs['kv_cache'].clone()
        trt_runner(
            inputs['positions'], q, k, inputs['kv_c'],
            inputs['rope_cos_sin_cache'], rope_is_neox,
            inputs['kv_cache_slot_mapping'], c, inputs['kv_cache_quant_scale'],
        )

    # Assemble benchmark list
    bench_fns    = [fresh_cuda_launcher, fresh_triton_fused_launcher,
                    triton_baseline_launcher, fresh_torch_launcher, fresh_dynamo_launcher]
    bench_labels = ["CUDA", "Triton_fused", "Triton_baseline", "Torch", "Dynamo"]
    if trt_available:
        bench_fns.append(fresh_trt_launcher)
        bench_labels.append("TensorRT")

    print("\nBenchmarking CUDA vs Triton_fused vs Triton_baseline vs Torch vs Dynamo"
          + (" vs TensorRT" if trt_available else "") + "...")
    bench_result = benchmark_performance(*bench_fns, labels=bench_labels)

    # Build JSON records
    shape_meaning = "num_q_heads,num_tokens,rot_dim,kv_lora_rank,block_size"
    shape_str = f"[{num_q_heads},{num_tokens},{rot_dim},{kv_lora_rank},{block_size}]"
    records = []
    for label, ms in bench_result["timings"]:
        records.append({
            "hw": hw_name,
            "type": "op",
            "method": label,
            "op": "rope_quant_kvcache",  # distinguish from rope_kvcache
            "seqlen": num_tokens,
            "shape_meaning": shape_meaning,
            "shape": shape_str,
            "time": "ms",
            "avg": round(ms, 3),
            "best_cfg": {"BM": BM, "num_warps": num_warps, "num_stages": num_stages},
        })

    # Validation helper: dequantize uint8/int8 kv_cache back to fp16 for comparison
    # Following PR's approach: compare dequantized fp16, not raw bytes
    def dequantize_kv_cache(kv_cache_u8, scale):
        """Reinterpret uint8 bytes as signed int8, then dequantize to fp16."""
        return kv_cache_u8.view(torch.int8).float().div(scale).to(torch.float16)

    scale_val = inputs['kv_cache_quant_scale'].item()

    # Validation 3: Torch_baseline vs CUDA
    q_torch_val = inputs['q_pe'].clone()
    k_torch_val = inputs['k_pe'].clone()
    c_torch_val = inputs['kv_cache'].clone()
    torch_baseline(
        inputs['positions'], q_torch_val, k_torch_val, inputs['kv_c'],
        inputs['rope_cos_sin_cache'], rope_is_neox,
        inputs['kv_cache_slot_mapping'], c_torch_val, inputs['kv_cache_quant_scale'],
    )
    # Re-run CUDA for fresh reference
    q_pe_cuda_v3 = inputs['q_pe'].clone()
    k_pe_cuda_v3 = inputs['k_pe'].clone()
    kv_cache_cuda_v3 = torch.zeros_like(kv_cache_cuda)
    mla_rope_quant_cuda.concat_and_cache_mla_rope_quant_fused(
        inputs['positions'], q_pe_cuda_v3, k_pe_cuda_v3, inputs['kv_c'],
        inputs['rope_cos_sin_cache'], rope_is_neox,
        inputs['kv_cache_slot_mapping'], kv_cache_cuda_v3,
        inputs['kv_cache_quant_scale'],
    )
    cuda_kv_dq_v3  = dequantize_kv_cache(kv_cache_cuda_v3, scale_val)
    torch_kv_dq    = dequantize_kv_cache(c_torch_val, scale_val)
    print("\nValidating Torch_baseline vs CUDA (q_pe, k_pe, kv_cache dequantized)...")
    validate_correctness(
        lambda: None,
        lambda: None,
        [q_pe_cuda_v3, k_pe_cuda_v3, cuda_kv_dq_v3],
        [q_torch_val, k_torch_val, torch_kv_dq],
        rtol=1e-1,
        atol=1e-3,
    )

    # Validation 1: Triton_baseline vs CUDA
    # CUDA outputs uint8 kv_cache (fp8 quant); Triton also writes int8 quant
    # Dequantize both to fp16 before comparing (following PR's test approach)
    q_base_val = inputs['q_pe'].clone()
    k_base_val = inputs['k_pe'].clone()
    c_base_val = inputs['kv_cache'].clone()

    def baseline_for_cuda_val():
        triton_baseline(
            q_base_val, inputs['rope_cos_sin_cache'], inputs['positions'],
            num_tokens, num_q_heads, rot_dim,
            k_base_val, inputs['kv_cache_slot_mapping'],
            inputs['kv_c'], c_base_val, inputs['kv_cache_quant_scale'],
            kv_lora_rank, block_size,
        )

    cuda_launcher()  # populate q_pe_cuda, k_pe_cuda, kv_cache_cuda
    baseline_for_cuda_val()  # populate q_base_val, k_base_val, c_base_val

    cuda_kv_dq   = dequantize_kv_cache(kv_cache_cuda, scale_val)
    base_kv_dq   = dequantize_kv_cache(c_base_val, scale_val)

    print("\nValidating Triton_baseline vs CUDA (q_pe, k_pe, kv_cache dequantized)...")
    validate_correctness(
        lambda: None,
        lambda: None,
        [q_pe_cuda, k_pe_cuda, cuda_kv_dq],
        [q_base_val, k_base_val, base_kv_dq],
        rtol=1e-1,
        atol=1e-3,   # dequantized fp16, matching PR's atol=0.001
    )

    # Validation 2: Triton_fused vs CUDA
    q_pe_cuda2 = inputs['q_pe'].clone()
    k_pe_cuda2 = inputs['k_pe'].clone()
    kv_cache_cuda2 = torch.zeros_like(kv_cache_cuda)

    def cuda_launcher2():
        mla_rope_quant_cuda.concat_and_cache_mla_rope_quant_fused(
            inputs['positions'], q_pe_cuda2, k_pe_cuda2, inputs['kv_c'],
            inputs['rope_cos_sin_cache'], rope_is_neox,
            inputs['kv_cache_slot_mapping'], kv_cache_cuda2,
            inputs['kv_cache_quant_scale'],
        )

    q_fused_val = inputs['q_pe'].clone()
    k_fused_val = inputs['k_pe'].clone()
    c_fused_val = inputs['kv_cache'].clone()

    def fused_for_cuda_val():
        args = [
            q_fused_val, inputs['rope_cos_sin_cache'], inputs['positions'],
            num_tokens, num_q_heads, rot_dim,
            q_fused_val.stride(0), q_fused_val.stride(1),
            inputs['rope_cos_sin_cache'].stride(0),
            k_fused_val, inputs['kv_cache_slot_mapping'], k_fused_val.stride(0),
            inputs['kv_c'], c_fused_val, inputs['kv_cache_quant_scale'],
            kv_lora_rank, block_size,
            inputs['kv_c'].stride(0),
            c_fused_val.stride(0), c_fused_val.stride(1),
        ]
        compiled[grid](*args)

    cuda_launcher2()
    fused_for_cuda_val()

    cuda_kv_dq2  = dequantize_kv_cache(kv_cache_cuda2, scale_val)
    fused_kv_dq  = dequantize_kv_cache(c_fused_val, scale_val)

    print("\nValidating Triton_fused vs CUDA (q_pe, k_pe, kv_cache dequantized)...")
    validate_correctness(
        lambda: None,
        lambda: None,
        [q_pe_cuda2, k_pe_cuda2, cuda_kv_dq2],
        [q_fused_val, k_fused_val, fused_kv_dq],
        rtol=1e-1,
        atol=1e-3,   # dequantized fp16, matching PR's atol=0.001
    )

    # Validation 4: Dynamo vs CUDA
    q_dynamo_val = inputs['q_pe'].clone()
    k_dynamo_val = inputs['k_pe'].clone()
    c_dynamo_val = inputs['kv_cache'].clone()
    dynamo_baseline(
        inputs['positions'], q_dynamo_val, k_dynamo_val, inputs['kv_c'],
        inputs['rope_cos_sin_cache'], rope_is_neox,
        inputs['kv_cache_slot_mapping'], c_dynamo_val, inputs['kv_cache_quant_scale'],
    )
    q_pe_cuda_v4 = inputs['q_pe'].clone()
    k_pe_cuda_v4 = inputs['k_pe'].clone()
    kv_cache_cuda_v4 = torch.zeros_like(kv_cache_cuda)
    mla_rope_quant_cuda.concat_and_cache_mla_rope_quant_fused(
        inputs['positions'], q_pe_cuda_v4, k_pe_cuda_v4, inputs['kv_c'],
        inputs['rope_cos_sin_cache'], rope_is_neox,
        inputs['kv_cache_slot_mapping'], kv_cache_cuda_v4,
        inputs['kv_cache_quant_scale'],
    )
    cuda_kv_dq_v4 = dequantize_kv_cache(kv_cache_cuda_v4, scale_val)
    dynamo_kv_dq  = dequantize_kv_cache(c_dynamo_val, scale_val)
    print("\nValidating Dynamo vs CUDA (q_pe, k_pe, kv_cache dequantized)...")
    validate_correctness(
        lambda: None,
        lambda: None,
        [q_pe_cuda_v4, k_pe_cuda_v4, cuda_kv_dq_v4],
        [q_dynamo_val, k_dynamo_val, dynamo_kv_dq],
        rtol=1e-1,
        atol=1e-3,
    )

    # Validation 5: TensorRT vs CUDA (only if TRT engine was built)
    if trt_available:
        q_trt_val = inputs['q_pe'].clone()
        k_trt_val = inputs['k_pe'].clone()
        c_trt_val = inputs['kv_cache'].clone()
        trt_runner(
            inputs['positions'], q_trt_val, k_trt_val, inputs['kv_c'],
            inputs['rope_cos_sin_cache'], rope_is_neox,
            inputs['kv_cache_slot_mapping'], c_trt_val, inputs['kv_cache_quant_scale'],
        )
        q_pe_cuda_v5 = inputs['q_pe'].clone()
        k_pe_cuda_v5 = inputs['k_pe'].clone()
        kv_cache_cuda_v5 = torch.zeros_like(kv_cache_cuda)
        mla_rope_quant_cuda.concat_and_cache_mla_rope_quant_fused(
            inputs['positions'], q_pe_cuda_v5, k_pe_cuda_v5, inputs['kv_c'],
            inputs['rope_cos_sin_cache'], rope_is_neox,
            inputs['kv_cache_slot_mapping'], kv_cache_cuda_v5,
            inputs['kv_cache_quant_scale'],
        )
        cuda_kv_dq_v5 = dequantize_kv_cache(kv_cache_cuda_v5, scale_val)
        trt_kv_dq     = dequantize_kv_cache(c_trt_val, scale_val)
        print("\nValidating TensorRT vs CUDA (q_pe, k_pe, kv_cache dequantized)...")
        validate_correctness(
            lambda: None,
            lambda: None,
            [q_pe_cuda_v5, k_pe_cuda_v5, cuda_kv_dq_v5],
            [q_trt_val, k_trt_val, trt_kv_dq],
            rtol=1e-1,
            atol=1e-3,
        )

    return records


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("MLA RoPE + FP8 Quant + KV-Cache Fused Kernel - Benchmark & Validation")
    print("=" * 70)

    num_q_heads = 128
    rot_dim = 64
    kv_lora_rank = 512
    num_blocks = 512
    block_size = 16
    rope_is_neox = True

    token_sizes = [128, 256, 512, 1024, 2048, 4096]

    hw_name = get_hw_name()
    print(f"\nDetected hardware: {hw_name}")

    all_records = []
    for num_tokens in token_sizes:
        records = run_single_config(
            num_tokens, num_q_heads, rot_dim, kv_lora_rank,
            num_blocks, block_size, rope_is_neox, hw_name,
        )
        all_records.extend(records)

    output_path = os.path.join(os.path.dirname(__file__), f"../result/rope_quant_kvcache_results_{get_hw_name()}.json")
    with open(output_path, "w") as f:
        json.dump(all_records, f, indent=4)
    print(f"Results saved to: {output_path}")

    plot_results(all_records, hw_name, token_sizes)


if __name__ == "__main__":
    main()
