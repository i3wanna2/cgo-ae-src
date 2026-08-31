#!/usr/bin/env python3
"""Collect Figure-6 (rope_quant_kvcache) log → JSON under figure-6/results/."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
KV_TIMING_LINE = re.compile(
    r"^\s*(CUDA|Triton_fused|Triton_baseline|Torch|Dynamo|TensorRT)\s*:\s*"
    r"([\d.]+)\s*ms",
    re.MULTILINE,
)
KV_BEST = re.compile(
    r"Best config:\s*BM=(\d+)\s+num_warps=(\d+)\s+num_stages=(\d+)\s*->\s*([\d.]+)\s*ms"
)


def platform_from_hw(hw: str) -> str:
    u = hw.upper()
    if "A100" in u:
        return "A100"
    if "H100" in u:
        return "H100"
    if "H800" in u:
        return "H800"
    return "H800"


def kv_shape(
    seqlen: int,
    num_q_heads: int = 128,
    rot_dim: int = 64,
    kv_lora_rank: int = 512,
    block_size: int = 16,
) -> str:
    return f"[{num_q_heads},{seqlen},{rot_dim},{kv_lora_rank},{block_size}]"


def detect_hw_from_log(text: str, fallback: str = "UNKNOWN") -> str:
    m = re.search(r"gpu=([^\s,]+)", text)
    if m:
        return m.group(1).replace(" ", "_")
    m = re.search(r"GPU[=:]?\s*([^\n]+)", text, re.I)
    if m:
        return m.group(1).strip().replace(" ", "_")
    return fallback


def default_hw_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception:
        pass
    return "UNKNOWN"


def collect_kvcache(log_path: Path, seqlen: int, hw: str | None) -> dict:
    if not log_path.is_file():
        raise FileNotFoundError(f"KV log not found: {log_path}")
    text = log_path.read_text(errors="replace")
    hw_final = hw or detect_hw_from_log(text)
    plat = platform_from_hw(hw_final)

    matches = list(KV_TIMING_LINE.finditer(text))
    by_method: dict[str, float] = {}
    for m in matches:
        by_method[m.group(1)] = float(m.group(2))

    best = None
    bm = KV_BEST.search(text)
    if bm:
        best = {
            "BM": int(bm.group(1)),
            "num_warps": int(bm.group(2)),
            "num_stages": int(bm.group(3)),
            "tune_ms": float(bm.group(4)),
        }

    shape = kv_shape(seqlen)
    records = []
    for method, avg in by_method.items():
        rec = {
            "type": "op",
            "op": "rope_quant_kvcache",
            "method": method,
            "seqlen": seqlen,
            "shape": shape,
            "shape_meaning": "num_q_heads,num_tokens,rot_dim,kv_lora_rank,block_size",
            "avg": avg,
            "hw": hw_final,
            "log": log_path.name,
            "status": "ok",
        }
        if best and method in ("CUDA", "Triton_fused"):
            rec["best_cfg"] = {
                "BM": best["BM"],
                "num_warps": best["num_warps"],
                "num_stages": best["num_stages"],
            }
        records.append(rec)

    order = ["Torch", "Dynamo", "TensorRT", "CUDA", "Triton_fused", "Triton_baseline"]
    rank = {m: i for i, m in enumerate(order)}
    records.sort(key=lambda r: rank.get(r["method"], 99))

    out = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "rope_quant_kvcache",
        "hw": hw_final,
        "platform": plat,
        "seqlen": seqlen,
        "unit": "ms",
        "source": "logs",
        "records": records,
    }
    if best:
        out["best_cfg"] = best
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Collect Figure-6 KV logs into JSON")
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--hw", default=None)
    p.add_argument(
        "--kv-log",
        type=Path,
        default=None,
        help="Default: figure-6/logs/rope_quant_kvcache_<seqlen>.log",
    )
    p.add_argument("--out-dir", type=Path, default=FIG_DIR / "results")
    args = p.parse_args()

    log_path = args.kv_log or (FIG_DIR / "logs" / f"rope_quant_kvcache_{args.seqlen}.log")
    obj = collect_kvcache(log_path, args.seqlen, args.hw or default_hw_name())
    out = args.out_dir / f"rope_quant_kvcache_{obj['hw']}_seq{args.seqlen}.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_ok = sum(1 for r in obj["records"] if r.get("avg") is not None)
    print(f"Wrote {n_ok}/{len(obj['records'])} records -> {out}")


if __name__ == "__main__":
    main()
