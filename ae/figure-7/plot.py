#!/usr/bin/env python3
"""Figure-7 plot: attention variants @ seqlen=4096 (paper kernel_atten_4096.py).

Reads ae/figure-7/results/atten_kernel_<hw>_seq4096.json
Writes PDF to this directory: eva_atten_kernel_4096.pdf
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

# Paper DEVICE_NAMES is A100+H100; AE also accepts H800 when present.
DEVICE_NAMES = [
    "NVIDIA_A100-SXM4-40GB",
    "NVIDIA_H100_80GB_HBM3",
    "NVIDIA_H800",
]

model_names = ["attn", "corm", "h2o", "roco", "keyformer", "snapkv", "gemma2"]
sys_names = ["torch", "dynamo", "tensorrt", "tvm", "flashtensor", "our"]
SYS_LABEL_MAP = {
    "torch": "PyTorch",
    "dynamo": "TorchInductor",
    "tensorrt": "TensorRT",
    "tvm": "TVM",
    "flashtensor": "FlashTensor",
    "our": "TileFusion (Ours)",
}
SYS_COLOR_MAP = {
    "torch": "#8ECFC9",
    "dynamo": "#BEB8DC",
    "tensorrt": "#FFBE7A",
    "tvm": "#B5DCAA",
    "flashtensor": "#87CEEB",
    "our": "#FA7F6F",
}
BAR_EDGECOLOR = "#2F2F2F"
BAR_LINEWIDTH = 0.7
MODEL_LABEL_MAP = {
    "attn": "V.A.",
    "corm": "CoRM",
    "h2o": "H2O",
    "roco": "RoCo",
    "keyformer": "Keyformer",
    "snapkv": "SnapKV",
    "gemma2": "Gemma-2",
}
FONT_SCALE = 1.5
TITLE_FONTSIZE = 20
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 10 * FONT_SCALE
LEGEND_FONTSIZE = 25
DEVICE_DISPLAY_NAMES = {
    "NVIDIA_A100-SXM4-40GB": "A100",
    "NVIDIA_H100_80GB_HBM3": "H100",
    "NVIDIA_H800": "H800",
}


def _configure_font():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "pdf.fonttype": 42,
        }
    )


def discover_devices(results_dir: Path, seqlen: int) -> list[str]:
    paths = sorted(results_dir.glob(f"atten_kernel_*_seq{seqlen}.json"))
    devices = []
    for p in paths:
        prefix, mid = "atten_kernel_", f"_seq{seqlen}.json"
        name = p.name
        if name.startswith(prefix) and name.endswith(mid):
            devices.append(name[len(prefix) : -len(mid)])
    ordered = [d for d in DEVICE_NAMES if d in devices]
    ordered += [d for d in devices if d not in ordered]
    return ordered


def load_device_df(results_dir: Path, device: str, seqlen: int) -> pd.DataFrame | None:
    path = results_dir / f"atten_kernel_{device}_seq{seqlen}.json"
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
    df["op"] = df["op"].replace({"kf": "keyformer"})
    return df


def _safe_float(x):
    try:
        v = float(x)
    except Exception:
        return None
    if not (v > 0):
        return None
    return v


def print_tilefusion_vs_flashtensor_speedups_4096(device_data_dict, seqlen: int = 4096):
    baseline_method = "flashtensor"
    ours_method = "our"
    devices = [d for d in DEVICE_NAMES if device_data_dict.get(d) is not None]
    devices += [d for d in device_data_dict if d not in devices and device_data_dict.get(d) is not None]
    if not devices:
        print("No device data available for speedup printing.")
        return

    for device in devices:
        df = device_data_dict[device]
        subset = df[(df["type"] == "op") & (df["seqlen"] == seqlen)]
        print("")
        print(f"=== {DEVICE_DISPLAY_NAMES.get(device, device)} (seqlen={seqlen}) ===")
        if subset.empty:
            print("No kernel (op) data found.")
            continue

        ops = sorted(subset["op"].unique())
        ordered_ops = [m for m in model_names if m in ops]
        ordered_ops += [m for m in ops if m not in model_names]
        ordered_ops = [m for m in ordered_ops if m != "dsa_mla"]

        speedups = []
        for op in ordered_ops:
            op_df = subset[subset["op"] == op]
            if op_df.empty:
                continue
            ft_row = op_df[op_df["method"] == baseline_method]
            our_row = op_df[op_df["method"] == ours_method]
            ft = _safe_float(ft_row["avg"].iloc[0]) if not ft_row.empty else None
            our = _safe_float(our_row["avg"].iloc[0]) if not our_row.empty else None
            if ft is None or our is None:
                missing = []
                if ft is None:
                    missing.append("FlashTensor")
                if our is None:
                    missing.append("TileFusion")
                print(f"- {MODEL_LABEL_MAP.get(op, op)}: N/A (missing {', '.join(missing)})")
                continue
            sp = ft / our
            speedups.append(sp)
            print(f"- {MODEL_LABEL_MAP.get(op, op)}: {sp:.3f}x (FT {ft:.4f} ms / Ours {our:.4f} ms)")

        if speedups:
            print(f"Average speedup across {len(speedups)} workloads: {sum(speedups)/len(speedups):.3f}x")
        else:
            print("Average speedup: N/A (no comparable workloads with both methods)")


def plot_combined_kernel_charts(device_data_dict, out_dir: Path, seqlen: int = 4096):
    all_models_set = set()
    for df in device_data_dict.values():
        if df is None:
            continue
        subset = df[(df["type"] == "op") & (df["seqlen"] == seqlen)]
        all_models_set.update(subset["op"].unique())

    valid_models = [m for m in model_names if m in all_models_set]
    valid_models += [m for m in all_models_set if m not in model_names]
    valid_models = [m for m in valid_models if m != "dsa_mla"]
    if not valid_models:
        print("No kernel data found")
        return

    devices = [d for d in DEVICE_NAMES if device_data_dict.get(d) is not None]
    devices += [d for d in device_data_dict if d not in devices and device_data_dict.get(d) is not None]
    num_devices = len(devices)
    num_models = len(valid_models)

    _configure_font()
    fig, axs = plt.subplots(
        num_devices,
        num_models,
        figsize=(3.5 * num_models, 3 * num_devices),
        sharey=False,
    )
    if num_devices == 1:
        axs = [axs]
    if num_models == 1:
        axs = [[ax] for ax in axs]

    for row, device in enumerate(devices):
        df = device_data_dict[device]
        subset = df[(df["type"] == "op") & (df["seqlen"] == seqlen)]
        for col, model in enumerate(valid_models):
            ax = axs[row][col]
            model_data = subset[subset["op"] == model]
            if model_data.empty:
                ax.set_visible(False)
                continue
            methods = sorted(model_data["method"].unique())
            ordered_methods = [m for m in sys_names if m in methods]
            ordered_methods += [m for m in methods if m not in sys_names]
            plot_methods = [m for m in ordered_methods if m not in ("tilelang", "flashtensor-3.5")]
            if not plot_methods:
                ax.set_visible(False)
                continue
            xs = list(range(len(plot_methods)))
            for i, method in enumerate(plot_methods):
                m_df = model_data[model_data["method"] == method]
                height = float(m_df["avg"].iloc[0]) if not m_df.empty else 0.0
                ax.bar(
                    xs[i],
                    height,
                    width=0.65,
                    color=SYS_COLOR_MAP.get(method, "#888888"),
                    edgecolor=BAR_EDGECOLOR,
                    linewidth=BAR_LINEWIDTH,
                    label=method,
                )
            if row == 0:
                ax.set_title(MODEL_LABEL_MAP.get(model, model.upper()), fontsize=TITLE_FONTSIZE)
            if col == 0:
                ax.set_ylabel(
                    f"{DEVICE_DISPLAY_NAMES.get(device, device)}\nTime (ms)",
                    fontsize=LABEL_FONTSIZE,
                )
            ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
            ax.set_xticks(xs)
            ax.set_xticklabels([])
            ax.tick_params(axis="x", labelbottom=False)
            ax.grid(True, linestyle="--", alpha=0.7, axis="y")

    all_handles, all_labels, seen = [], [], set()
    for row in axs:
        for ax in row:
            if not ax.get_visible():
                continue
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in seen:
                    all_handles.append(handle)
                    all_labels.append(label)
                    seen.add(label)

    sorted_indices = []
    for s in sys_names:
        if s in all_labels:
            sorted_indices.append(all_labels.index(s))
    for i, _ in enumerate(all_labels):
        if i not in sorted_indices:
            sorted_indices.append(i)
    final_handles = [all_handles[i] for i in sorted_indices]
    final_labels = [SYS_LABEL_MAP.get(all_labels[i], all_labels[i]) for i in sorted_indices]

    fig.legend(
        final_handles,
        final_labels,
        loc="upper center",
        ncol=max(1, len(final_labels)),
        bbox_to_anchor=(0.5, 1.22),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    # Leave headroom so the top legend does not overlap subplot titles.
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "eva_atten_kernel_4096.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_FIGDIR)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--devices", nargs="*", default=None)
    args = ap.parse_args()

    devices = args.devices or discover_devices(args.results_dir, args.seqlen)
    device_data_dict = {d: load_device_df(args.results_dir, d, args.seqlen) for d in devices}
    print_tilefusion_vs_flashtensor_speedups_4096(device_data_dict, args.seqlen)
    plot_combined_kernel_charts(device_data_dict, args.out_dir, args.seqlen)


if __name__ == "__main__":
    main()
