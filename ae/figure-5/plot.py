#!/usr/bin/env python3
"""Figure-5 (DSA) plot @ seqlen=4096.

Reads JSON from figure-5/results/dsa_mla_<hw>_seq4096.json
(produced by figure-5/collect.py). Writes PDF into this directory.
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

DSA_BAR_ORDER = ("torch", "dynamo", "tensorrt", "tilelang", "our")
# Plot bar "tilelang": A100 → plain tilelang; H100/H800 → tilelang-ws.
TILELANG_METHOD_BY_DEVICE = {
    "NVIDIA_A100-SXM4-40GB": "tilelang",
    "NVIDIA_H100_80GB_HBM3": "tilelang-ws",
    "NVIDIA_H800": "tilelang-ws",
}


def tilelang_method_for_device(device: str) -> str:
    """Pick TileLang variant for the TileLang legend bar (A100 cannot run -ws)."""
    if device in TILELANG_METHOD_BY_DEVICE:
        return TILELANG_METHOD_BY_DEVICE[device]
    u = device.upper()
    if "A100" in u:
        return "tilelang"
    if "H100" in u or "H800" in u:
        return "tilelang-ws"
    return "tilelang-ws"


SYS_LABEL_MAP = {
    "torch": "PyTorch",
    "tensorrt": "TensorRT",
    "dynamo": "TorchInductor",
    "tilelang": "TileLang (manual fusion)",
    "our": "TileFusion (Ours)",
}
SYS_COLOR_MAP = {
    "torch": "#8ECFC9",
    "tensorrt": "#FFBE7A",
    "dynamo": "#BEB8DC",
    "tilelang": "#B07AA1",
    "our": "#FA7F6F",
}
DSA_BAR_WIDTH = 0.36
BAR_EDGECOLOR = "#2F2F2F"
BAR_LINEWIDTH = 0.7
DEVICE_LABEL_MAP = {
    "NVIDIA_A100-SXM4-40GB": "A100",
    "NVIDIA_H100_80GB_HBM3": "H100",
    "NVIDIA_H800": "H800",
}
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 24


def _configure_font():
    # Prefer Times; fall back silently if the face is missing in the AE image.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "pdf.fonttype": 42,
        }
    )


def discover_devices(results_dir: Path, seqlen: int) -> list[str]:
    paths = sorted(results_dir.glob(f"dsa_mla_*_seq{seqlen}.json"))
    devices = []
    for p in paths:
        # dsa_mla_<hw>_seq4096.json  — hw may contain underscores
        name = p.name
        prefix, mid = f"dsa_mla_", f"_seq{seqlen}.json"
        if name.startswith(prefix) and name.endswith(mid):
            devices.append(name[len(prefix) : -len(mid)])
    # Prefer paper order when present
    preferred = [
        "NVIDIA_A100-SXM4-40GB",
        "NVIDIA_H100_80GB_HBM3",
        "NVIDIA_H800",
    ]
    ordered = [d for d in preferred if d in devices]
    ordered += [d for d in devices if d not in ordered]
    return ordered


def load_records(results_dir: Path, device: str, seqlen: int) -> pd.DataFrame | None:
    path = results_dir / f"dsa_mla_{device}_seq{seqlen}.json"
    if not path.exists():
        print(f"Warning: {path} not found.")
        return None
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", payload if isinstance(payload, list) else [])
    if not records:
        print(f"Error: No data found for {device}.")
        return None
    df = pd.DataFrame(records)
    df["avg"] = pd.to_numeric(df["avg"], errors="coerce")
    df = df.dropna(subset=["avg"])
    return df


def plot_dsa_kernel_combined(
    results_dir: Path,
    out_dir: Path,
    seqlen: int = 4096,
    devices: list[str] | None = None,
):
    device_order = devices or discover_devices(results_dir, seqlen)
    if not device_order:
        raise SystemExit(f"No DSA JSON under {results_dir} for seqlen={seqlen}")

    _configure_font()
    fig_w, fig_h = max(5.0, 5.0 * len(device_order)), 4.0
    fig, axs = plt.subplots(1, len(device_order), figsize=(fig_w, fig_h), sharey=False)
    if len(device_order) == 1:
        axs = [axs]

    for i, device in enumerate(device_order):
        df = load_records(results_dir, device, seqlen)
        if df is None:
            continue
        subset = df[(df["type"] == "op") & (df["op"] == "dsa_mla") & (df["seqlen"] == seqlen)]
        if subset.empty:
            print(f"No DSA data for {device}")
            continue

        ax = axs[i]
        xs = list(range(len(DSA_BAR_ORDER)))
        for j, legend_key in enumerate(DSA_BAR_ORDER):
            if legend_key == "tilelang":
                src = tilelang_method_for_device(device)
                method_data = subset[subset["method"] == src]
                color = SYS_COLOR_MAP["tilelang"]
            else:
                method_data = subset[subset["method"] == legend_key]
                color = SYS_COLOR_MAP.get(legend_key, "#888888")
            if not method_data.empty:
                ax.bar(
                    j,
                    float(method_data["avg"].iloc[0]),
                    width=DSA_BAR_WIDTH,
                    color=color,
                    edgecolor=BAR_EDGECOLOR,
                    linewidth=BAR_LINEWIDTH,
                )

        ax.set_title(DEVICE_LABEL_MAP.get(device, device), fontsize=TITLE_FONTSIZE)
        ax.set_xticks(xs)
        ax.tick_params(axis="x", labelbottom=False)
        if i == 0:
            ax.set_ylabel("Time (ms)", fontsize=LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    handles = [
        plt.Rectangle(
            (0, 0),
            0.5,
            0.8,
            facecolor=SYS_COLOR_MAP[m],
            edgecolor=BAR_EDGECOLOR,
            linewidth=BAR_LINEWIDTH,
        )
        for m in DSA_BAR_ORDER
    ]
    fig.legend(
        handles,
        [SYS_LABEL_MAP[m] for m in DSA_BAR_ORDER],
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
    fig.subplots_adjust(left=0.05, right=0.995, top=0.66, bottom=0.14, wspace=0.14)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "eva_dsa_kernel_4096.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def print_tilefusion_speedups(results_dir: Path, seqlen: int = 4096, devices: list[str] | None = None):
    device_order = devices or discover_devices(results_dir, seqlen)
    print(f"\n[DSA speedups] op=dsa_mla, seqlen={seqlen}")
    for device in device_order:
        df = load_records(results_dir, device, seqlen)
        if df is None:
            continue
        subset = df[(df["type"] == "op") & (df["op"] == "dsa_mla") & (df["seqlen"] == seqlen)]
        lat = {}
        for _, row in subset.iterrows():
            try:
                lat[row["method"]] = float(row["avg"])
            except Exception:
                continue
        if "our" not in lat:
            print(f"- {device}: missing TileFusion ('our')")
            continue
        our_ms = lat["our"]
        print(f"\nDevice: {DEVICE_LABEL_MAP.get(device, device)} ({device})")
        print(f"  TileFusion (our): {our_ms:.4f} ms")
        for lk in DSA_BAR_ORDER:
            if lk == "our":
                continue
            if lk == "tilelang":
                src = tilelang_method_for_device(device)
                if src not in lat:
                    continue
                base_ms = lat[src]
            elif lk not in lat:
                continue
            else:
                base_ms = lat[lk]
            print(f"  vs {SYS_LABEL_MAP[lk]:16s}: {base_ms / our_ms:8.3f}x   ({base_ms:.4f} ms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_FIGDIR)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--devices", nargs="*", default=None)
    args = ap.parse_args()
    print_tilefusion_speedups(args.results_dir, args.seqlen, args.devices)
    plot_dsa_kernel_combined(args.results_dir, args.out_dir, args.seqlen, args.devices)


if __name__ == "__main__":
    main()
