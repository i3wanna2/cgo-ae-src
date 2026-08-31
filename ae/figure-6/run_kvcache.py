#!/usr/bin/env python3
"""KV-cache (rope_quant_kvcache) AE runner for torch 2.9.1 conda.

Paper bars (tilefusion-paper/script/kernel_rope_quant_kvcache_4096.py):
  torch, dynamo, tensorrt, cuda, ours
JSON methods:
  Torch, Dynamo, TensorRT, CUDA, Triton_fused
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

TF = Path("/workspace/triton/tilefusion")
sys.path.insert(0, str(TF.parent))
sys.path.insert(0, str(TF / "ops" / "csrc" / "mla_rope"))


def load_fusion():
    import torch

    _orig = torch.onnx.export

    def _classic(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return _orig(*args, **kwargs)

    torch.onnx.export = _classic  # type: ignore[assignment]

    fusion = TF / "fusions" / "rope_quant_kvcache.py"
    spec = importlib.util.spec_from_file_location("rope_quant_kvcache", fusion)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["rope_quant_kvcache"] = mod
    spec.loader.exec_module(mod)
    return mod, torch


def main() -> None:
    mod, torch = load_fusion()
    import tensorrt as trt

    assert torch.__version__.startswith("2.9.1"), torch.__version__
    print(
        f"[kvcache-291] torch={torch.__version__} trt={trt.__version__} "
        f"cuda={torch.cuda.is_available()} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}",
        flush=True,
    )
    if trt.__version__ != "10.13.2.6":
        raise RuntimeError(f"Expected TensorRT 10.13.2.6, got {trt.__version__}")

    import mla_rope_quant_cuda  # noqa: F401

    print("[kvcache-291] mla_rope_quant_cuda OK", flush=True)

    seqlen = int(os.environ.get("KVCACHE_SEQLEN", "4096"))
    num_q_heads, rot_dim, kv_lora_rank = 128, 64, 512
    num_blocks, block_size, rope_is_neox = 512, 16, True
    hw_name = mod.get_hw_name()

    print(f"[kvcache-291] run_single_config seqlen={seqlen} hw={hw_name}", flush=True)
    records = mod.run_single_config(
        seqlen,
        num_q_heads,
        rot_dim,
        kv_lora_rank,
        num_blocks,
        block_size,
        rope_is_neox,
        hw_name,
    )

    paper = {"Torch", "Dynamo", "TensorRT", "CUDA", "Triton_fused"}
    kept = [r for r in records if r.get("method") in paper or r.get("method") == "Triton_baseline"]
    # Always print summary lines so collect_results.py can parse the tee'd log.
    for r in kept:
        if r.get("seqlen") == seqlen:
            print(f"  {r.get('method')}: {r.get('avg')} ms", flush=True)

    write_json = os.environ.get("KVCACHE_WRITE_JSON", "0") == "1"
    if write_json:
        out_dir = Path(os.environ.get("KVCACHE_OUT_DIR", str(TF / "result")))
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"rope_quant_kvcache_results_{hw_name}_ae291_seq{seqlen}.json"
        out.write_text(json.dumps(kept, indent=2) + "\n")
        print(f"[kvcache-291] wrote {out}", flush=True)
    else:
        print("[kvcache-291] skip side JSON (use figure-6/collect.py)", flush=True)

    need = {"Torch", "Dynamo", "TensorRT", "CUDA", "Triton_fused"}
    got = {r.get("method") for r in kept if r.get("seqlen") == seqlen}
    missing = need - got
    if missing:
        raise RuntimeError(f"Missing paper methods: {sorted(missing)}")


if __name__ == "__main__":
    main()
