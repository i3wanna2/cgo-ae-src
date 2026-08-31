#!/usr/bin/env python3
"""Collect Figure-5 (DSA) logs → JSON under figure-5/results/."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
DSA_MEAN = re.compile(r"(\w+)\s+kernel:.*mean=([\d.]+)\s*ms")
DSA_LOG_NAME = re.compile(r"dsa_mla_model_(.+)_(\d+)\.log$")


def platform_from_hw(hw: str) -> str:
    u = hw.upper()
    if "A100" in u:
        return "A100"
    if "H100" in u:
        return "H100"
    if "H800" in u:
        return "H800"
    return "H800"


def dsa_shape(platform: str, seqlen: int) -> str:
    k = 144 if platform == "A100" else 576
    return f"[1,128,{seqlen},{k}]"


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


def collect_dsa(log_dir: Path, seqlen: int, hw: str | None, platform: str | None) -> dict:
    if not log_dir.is_dir():
        raise FileNotFoundError(f"DSA log dir not found: {log_dir}")

    records: list[dict] = []
    detected_hw = hw
    for path in sorted(log_dir.glob("dsa_mla_model_*_*.log")):
        m = DSA_LOG_NAME.match(path.name)
        if not m:
            continue
        method, file_seqlen = m.group(1), int(m.group(2))
        if file_seqlen != seqlen:
            continue
        text = path.read_text(errors="replace")
        if detected_hw is None:
            detected_hw = detect_hw_from_log(text)
        km = None
        for km in DSA_MEAN.finditer(text):
            pass
        plat = platform or platform_from_hw(detected_hw or "H800")
        shape = dsa_shape(plat, seqlen)
        if km is None:
            records.append(
                {
                    "type": "op",
                    "op": "dsa_mla",
                    "method": method,
                    "seqlen": seqlen,
                    "shape": shape,
                    "avg": None,
                    "hw": detected_hw or "UNKNOWN",
                    "log": path.name,
                    "status": "missing",
                }
            )
            continue
        records.append(
            {
                "type": "op",
                "op": km.group(1),
                "method": method,
                "seqlen": seqlen,
                "shape": shape,
                "avg": float(km.group(2)),
                "hw": detected_hw or "UNKNOWN",
                "log": path.name,
                "status": "ok",
            }
        )

    hw_final = hw or detected_hw or "UNKNOWN"
    plat = platform or platform_from_hw(hw_final)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "dsa_mla",
        "hw": hw_final,
        "platform": plat,
        "seqlen": seqlen,
        "unit": "ms",
        "source": "logs",
        "records": records,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Collect Figure-5 DSA logs into JSON")
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--hw", default=None)
    p.add_argument("--platform", default=None)
    p.add_argument("--log-dir", type=Path, default=FIG_DIR / "logs")
    p.add_argument("--out-dir", type=Path, default=FIG_DIR / "results")
    args = p.parse_args()

    obj = collect_dsa(args.log_dir, args.seqlen, args.hw or default_hw_name(), args.platform)
    out = args.out_dir / f"dsa_mla_{obj['hw']}_seq{args.seqlen}.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_ok = sum(1 for r in obj["records"] if r.get("avg") is not None)
    print(f"Wrote {n_ok}/{len(obj['records'])} records -> {out}")


if __name__ == "__main__":
    main()
