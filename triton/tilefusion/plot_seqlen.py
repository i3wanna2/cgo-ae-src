import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

# 1. Load data
DEVICE_NAME = torch.cuda.get_device_name(0).replace(" ", "_")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
file_paths = [
    os.path.join(CURRENT_DIR, f"result/collected_seqlen_{DEVICE_NAME}.json"),
    os.path.join(CURRENT_DIR, f"result/collected_seqlen_ft_{DEVICE_NAME}.json")
]

records = []
for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        continue
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                records.append(json.loads(line))
            except:
                continue

if not records:
    print("Error: No data found in any of the specified files.")
    exit()

df = pd.DataFrame(records)

# 2. Data Cleaning
df['avg'] = pd.to_numeric(df['avg'], errors='coerce')
df = df.dropna(subset=['avg']) # Remove OOM or invalid data for line plots
df['op'] = df['op'].replace({'kf': 'keyformer'})

# Definitions from plot.py for consistency
model_names = ['attn', 'corm', 'h2o', 'roco', 'keyformer', 'snapkv', 'gemma2', 'dsa_mla']
sys_names = ['torch', 'sdpa', 'flash_attn', 'flashtensor', 'flashtensor-3.5', 'tilelang', 'our', 'dynamo', 'tvm']
COLOR_DEF = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2', '#008080', '#E6B333', '#3366E6']
MODEL_LABEL_MAP = {'attn': 'Attention', 'corm': 'CoRM', 'h2o': 'H2O', 'roco': 'RoCo', 'keyformer': 'Keyformer', 'snapkv': 'SnapKV', 'gemma2': 'Gemma-2', 'dsa_mla': 'DSA'}

# 3. Plotting function
def plot_line_charts(df_type, suffix):
    subset = df[df['type'] == df_type]
    all_models = sorted(subset['op'].unique())
    
    if not all_models:
        print(f"No data for {df_type}")
        return

    # Filter and order models
    valid_models = [m for m in model_names if m in all_models]
    valid_models += [m for m in all_models if m not in model_names]
    
    num_models = len(valid_models)
    plt.rcParams.update({"figure.figsize": (4 * num_models, 4), 'font.sans-serif': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, num_models, sharey=False)
    
    if num_models == 1:
        axs = [axs]

    # Map methods to colors
    method_to_color = {s: COLOR_DEF[i % len(COLOR_DEF)] for i, s in enumerate(sys_names)}

    for i, (model, ax) in enumerate(zip(valid_models, axs)):
        model_data = subset[subset['op'] == model]
        methods = sorted(model_data['method'].unique())
        
        # Ensure 'our' is plotted last or has a specific style if needed
        ordered_methods = [m for m in sys_names if m in methods]
        ordered_methods += [m for m in methods if m not in sys_names]

        for method in ordered_methods:
            method_data = model_data[model_data['method'] == method].sort_values('seqlen')
            color = method_to_color.get(method, None)
            ax.plot(method_data['seqlen'], method_data['avg'], marker='o', label=method, color=color, linewidth=2)
        
        ax.set_title(MODEL_LABEL_MAP.get(model, model.upper()), fontsize=12)
        ax.set_xlabel("Sequence Length", fontsize=10)
        if i == 0:
            ax.set_ylabel("Time (ms)", fontsize=10)
        
        # Set x-ticks to be the sequence lengths present in the data
        seqlens = sorted(model_data['seqlen'].unique())
        ax.set_xticks(seqlens)
        ax.grid(True, linestyle='--', alpha=0.7)

    # Global legend
    all_handles = []
    all_labels = []
    seen_labels = set()
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen_labels:
                all_handles.append(handle)
                all_labels.append(label)
                seen_labels.add(label)
    
    # Sort legend by sys_names order
    sorted_indices = []
    for s in sys_names:
        if s in all_labels:
            sorted_indices.append(all_labels.index(s))
    for i, l in enumerate(all_labels):
        if i not in sorted_indices:
            sorted_indices.append(i)
    
    final_handles = [all_handles[i] for i in sorted_indices]
    final_labels = [all_labels[i] for i in sorted_indices]

    fig.legend(final_handles, final_labels, loc='upper center', ncol=len(final_labels), 
               bbox_to_anchor=(0.5, 1.1), frameon=False, fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(CURRENT_DIR, f"result/fig_seqlen_combined_{suffix}_{DEVICE_NAME}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    os.makedirs(os.path.join(CURRENT_DIR, "result"), exist_ok=True)
    plot_line_charts('op', 'kernel')
    plot_line_charts('model', 'e2e')
