#!/usr/bin/env python3
"""List missing Figure-7 attention kernel cells (content-based).

A cell is done only if its log contains a successful latency line
(see collect.parse_avg_ms). Partial/in-progress logs count as missing.

Prints bash-friendly lines after a header:
  MODEL SYSTEM
"""
from __future__ import annotations

import argparse
from pathlib import Path

from collect import MODELS, SYSTEMS, collect, default_hw_name, log_path

FIG_DIR = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="Check missing Figure-7 attention runs")
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--hw", default=None)
    p.add_argument("--log-dir", type=Path, default=FIG_DIR / "logs")
    p.add_argument(
        "--quiet-header",
        action="store_true",
        help="Only print MODEL SYSTEM lines (for bash loops)",
    )
    args = p.parse_args()

    obj = collect(args.log_dir, args.seqlen, args.hw or default_hw_name())
    missing = []
    for r in obj["records"]:
        if r.get("avg") is None:
            # reverse keyformer → kf for runner
            model = "kf" if r["op"] == "keyformer" else r["op"]
            missing.append((model, r["method"]))

    if not args.quiet_header:
        total = len(MODELS) * len(SYSTEMS)
        print(f"{len(missing)}/{total} missing (seqlen={args.seqlen}):")
        for model, system in missing:
            path = log_path(args.log_dir, model, system, args.seqlen)
            status = "no_log" if not path.is_file() else "incomplete"
            print(f"  {model:8s} {system:12s}  ({status}: {path.name})")
        return

    for model, system in missing:
        print(f"{model} {system}")


if __name__ == "__main__":
    main()
