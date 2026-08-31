#!/usr/bin/env python3
"""Figure-6 (rope_quant_kvcache) plot @ seqlen=4096.

Reads JSON from figure-6/results/rope_quant_kvcache_<hw>_seq4096.json
(produced by figure-6/collect.py). Writes PDF into this directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = CURRENT_DIR / "results"
DEFAULT_FIGDIR = CURRENT_DIR

DEVICE_LABEL_MAP = {
    "NVIDIA_A100-SXM4-40GB": "A100",
    "NVIDIA_H100_80GB_HBM3": "H100",
    "NVIDIA_H800": "H800",
}

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 24
BAR_EDGECOLOR = "#2F2F2F"
BAR_LINEWIDTH = 0.7

ROPE_BAR_ORDER = ("torch", "dynamo", "tensorrt", "cuda", "ours")
METHOD_CANDIDATES = {
    "torch": ("Torch", "torch", "PyTorch"),
    "tensorrt": ("TensorRT", "tensorrt"),
    "dynamo": ("Dynamo", "dynamo", "TorchInductor"),
    "cuda": ("CUDA",),
    "ours": ("Triton_fused",),
}
METHOD_LABEL = {
    "torch": "PyTorch",
    "tensorrt": "TensorRT",
    "dynamo": "TorchInductor",
    "cuda": "CUDA (manual fusion)",
    "ours": "TileFusion (Ours)",
}
METHOD_COLOR = {
    "torch": "#8ECFC9",
    "tensorrt": "#FFBE7A",
    "dynamo": "#BEB8DC",
    "cuda": "#C4A35A",
    "ours": "#FA7F6F",
}
ROPE_BAR_WIDTH = 0.32
OUTPUT_BASENAME = "eva_rope_quant_kvcache_kernel_4096"


def _configure_font():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "pdf.fonttype": 42,
        }
    )


def discover_devices(results_dir: Path, seqlen: int) -> list[str]:
    paths = sorted(results_dir.glob(f"rope_quant_kvcache_*_seq{seqlen}.json"))
    devices = []
    for p in paths:
        prefix, mid = "rope_quant_kvcache_", f"_seq{seqlen}.json"
        name = p.name
        if name.startswith(prefix) and name.endswith(mid):
            devices.append(name[len(prefix) : -len(mid)])
    preferred = [
        "NVIDIA_A100-SXM4-40GB",
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_H800",
    ]
    ordered = [d for d in preferred if d in devices]
    ordered += [d for d in devices if d not in ordered]
    return ordered


def load_op_df(results_dir: Path, device: str, seqlen: int, op_name: str = "rope_quant_kvcache"):
    path = results_dir / f"rope_quant_kvcache_{device}_seq{seqlen}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", payload if isinstance(payload, list) else [])
    df = pd.DataFrame(records)
    df["avg"] = pd.to_numeric(df["avg"], errors="coerce")
    df = df.dropna(subset=["avg"])
    return df[(df["type"] == "op") & (df["op"] == op_name)]


def _get_method_avg(sub: pd.DataFrame, candidates: tuple):
    for name in candidates:
        rows = sub[sub["method"] == name]
        if not rows.empty:
            return float(rows["avg"].iloc[0])
    methods = sub["method"].astype(str)
    for name in candidates:
        m = methods.str.lower() == str(name).lower()
        if m.any():
            return float(sub.loc[m, "avg"].iloc[0])
    return None


def _draw_rope_bars(ax, sub):
    vals = {k: _get_method_avg(sub, METHOD_CANDIDATES[k]) for k in ROPE_BAR_ORDER}
    present = [v for v in vals.values() if v is not None and v > 0]
    ymax = max(present) * 1.12 if present else 1e-6
    for j, key in enumerate(ROPE_BAR_ORDER):
        ms = vals[key]
        if ms is None:
            continue
        ax.bar(
            j,
            ms,
            width=ROPE_BAR_WIDTH,
            color=METHOD_COLOR[key],
            edgecolor=BAR_EDGECOLOR,
            linewidth=BAR_LINEWIDTH,
        )
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(len(ROPE_BAR_ORDER)))
    ax.tick_params(axis="x", labelbottom=False)


def plot_rope(
    results_dir: Path,
    out_dir: Path,
    seqlen: int = 4096,
    devices: list[str] | None = None,
    op_name: str = "rope_quant_kvcache",
):
    device_order = devices or discover_devices(results_dir, seqlen)
    if not device_order:
        raise SystemExit(f"No KV JSON under {results_dir} for seqlen={seqlen}")

    _configure_font()
    fig_w, fig_h = max(5.0, 5.0 * len(device_order)), 4.0
    fig, axs = plt.subplots(1, len(device_order), figsize=(fig_w, fig_h), sharey=False)
    if len(device_order) == 1:
        axs = [axs]

    for i, device in enumerate(device_order):
        ax = axs[i]
        df = load_op_df(results_dir, device, seqlen, op_name)
        df = df[df["seqlen"] == seqlen]
        ax.set_title(DEVICE_LABEL_MAP.get(device, device), fontsize=TITLE_FONTSIZE)
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        else:
            _draw_rope_bars(ax, df)
            ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
            ax.grid(True, linestyle="--", alpha=0.7, axis="y")
        if i == 0:
            ax.set_ylabel("Time (ms)", fontsize=LABEL_FONTSIZE)

    handles = [
        plt.Rectangle(
            (0, 0),
            0.5,
            0.8,
            facecolor=METHOD_COLOR[k],
            edgecolor=BAR_EDGECOLOR,
            linewidth=BAR_LINEWIDTH,
        )
        for k in ROPE_BAR_ORDER
    ]
    fig.legend(
        handles,
        [METHOD_LABEL[k] for k in ROPE_BAR_ORDER],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=1.0,
        labelspacing=0.55,
    )
    fig.subplots_adjust(left=0.05, right=0.995, top=0.62, bottom=0.14, wspace=0.14)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{OUTPUT_BASENAME}.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def print_tilefusion_speedups(
    results_dir: Path,
    seqlen: int = 4096,
    devices: list[str] | None = None,
    op_name: str = "rope_quant_kvcache",
):
    device_order = devices or discover_devices(results_dir, seqlen)
    print(f"\n[RoPE KVCache speedups] op={op_name}, seqlen={seqlen}")
    for device in device_order:
        df = load_op_df(results_dir, device, seqlen, op_name)
        subset = df[df["seqlen"] == seqlen]
        if subset.empty:
            print(f"- {DEVICE_LABEL_MAP.get(device, device)}: no data")
            continue
        our_ms = _get_method_avg(subset, METHOD_CANDIDATES["ours"])
        if our_ms is None or our_ms <= 0:
            print(f"- {device}: missing TileFusion")
            continue
        print(f"\nDevice: {DEVICE_LABEL_MAP.get(device, device)} ({device})")
        print(f"  TileFusion (Ours): {our_ms:.4f} ms")
        for key in ROPE_BAR_ORDER:
            if key == "ours":
                continue
            base_ms = _get_method_avg(subset, METHOD_CANDIDATES[key])
            if base_ms is None or base_ms <= 0:
                continue
            print(f"  vs {METHOD_LABEL[key]:19s}: {base_ms / our_ms:8.3f}x   ({base_ms:.4f} ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_FIGDIR)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--devices", nargs="*", default=None)
    args = ap.parse_args()
    print_tilefusion_speedups(args.results_dir, args.seqlen, args.devices)
    plot_rope(args.results_dir, args.out_dir, args.seqlen, args.devices)


if __name__ == "__main__":
    main()
