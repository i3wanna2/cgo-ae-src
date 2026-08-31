#!/usr/bin/env python3
"""Collect Figure-7 attention-kernel logs → JSON under figure-7/results/.

Success is content-based (not filename-only):
  TileFusion:  ``<op> kernel: ... mean=<ms> ms``
  FlashTensor: ``[<label>] avg <ms> ms``

Paper plot ops (kernel_atten_4096.py): attn,corm,h2o,roco,keyformer,snapkv,gemma2
On disk / runners: keyformer ↔ kf.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1

MODELS = ("attn", "corm", "h2o", "roco", "kf", "snapkv", "gemma2")
SYSTEMS = ("torch", "dynamo", "tensorrt", "tvm", "flashtensor", "our")
# Plot renames kf → keyformer (same as paper script).
OP_PLOT_NAME = {"kf": "keyformer"}

TF_MEAN = re.compile(r"(\w+)\s+kernel:.*mean=([\d.]+)\s*ms")
FT_AVG = re.compile(r"\[([^\]]+)\]\s+avg\s+([\d.]+)\s+ms")
LOG_NAME = re.compile(r"(.+)_kernel_(.+)_(\d+)\.log$")

SHAPE = "[1,32,seqlen,128]"


def platform_from_hw(hw: str) -> str:
    u = hw.upper()
    if "A100" in u:
        return "A100"
    if "H100" in u:
        return "H100"
    if "H800" in u:
        return "H800"
    return "H800"


def default_hw_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception:
        pass
    return "UNKNOWN"


def detect_hw_from_log(text: str) -> str | None:
    m = re.search(r"gpu=([^\s,]+)", text)
    if m:
        return m.group(1).replace(" ", "_")
    m = re.search(r"GPU[=:]?\s*([^\n]+)", text, re.I)
    if m:
        return m.group(1).strip().replace(" ", "_")
    return None


def parse_avg_ms(text: str, method: str) -> float | None:
    """Return last successful kernel latency, or None if incomplete/failed."""
    if "torch.OutOfMemoryError: CUDA out of memory" in text:
        return None
    # Prefer TileFusion-style mean= (our / torch via tilefusion).
    last_tf = None
    for m in TF_MEAN.finditer(text):
        last_tf = float(m.group(2))
    if last_tf is not None:
        return last_tf
    # FlashTensor-AE: ``[our] avg`` when method=flashtensor, else ``[torch] avg`` etc.
    last_ft = None
    for m in FT_AVG.finditer(text):
        last_ft = float(m.group(2))
    return last_ft


def log_path(log_dir: Path, model: str, system: str, seqlen: int) -> Path:
    return log_dir / f"{model}_kernel_{system}_{seqlen}.log"


def collect(log_dir: Path, seqlen: int, hw: str | None) -> dict:
    records: list[dict] = []
    detected_hw = hw
    for model in MODELS:
        for system in SYSTEMS:
            path = log_path(log_dir, model, system, seqlen)
            op = OP_PLOT_NAME.get(model, model)
            if not path.is_file():
                records.append(
                    {
                        "type": "op",
                        "op": op,
                        "method": system,
                        "seqlen": seqlen,
                        "shape": SHAPE.replace("seqlen", str(seqlen)),
                        "avg": None,
                        "hw": detected_hw or "UNKNOWN",
                        "log": path.name,
                        "status": "missing",
                    }
                )
                continue
            text = path.read_text(errors="replace")
            if detected_hw is None:
                detected_hw = detect_hw_from_log(text)
            avg = parse_avg_ms(text, system)
            records.append(
                {
                    "type": "op",
                    "op": op,
                    "method": system,
                    "seqlen": seqlen,
                    "shape": SHAPE.replace("seqlen", str(seqlen)),
                    "avg": avg,
                    "hw": detected_hw or "UNKNOWN",
                    "log": path.name,
                    "status": "ok" if avg is not None else "incomplete",
                }
            )

    hw_final = hw or detected_hw or "UNKNOWN"
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "attention_variants",
        "hw": hw_final,
        "platform": platform_from_hw(hw_final),
        "seqlen": seqlen,
        "unit": "ms",
        "source": "logs",
        "records": records,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Collect Figure-7 attention logs")
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--hw", default=None)
    p.add_argument("--log-dir", type=Path, default=FIG_DIR / "logs")
    p.add_argument("--out-dir", type=Path, default=FIG_DIR / "results")
    args = p.parse_args()

    obj = collect(args.log_dir, args.seqlen, args.hw or default_hw_name())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"atten_kernel_{obj['hw']}_seq{args.seqlen}.json"
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_ok = sum(1 for r in obj["records"] if r.get("avg") is not None)
    print(f"Wrote {n_ok}/{len(obj['records'])} ok records -> {out}")


if __name__ == "__main__":
    main()
