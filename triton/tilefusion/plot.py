import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import re
import torch
# 1. 更加鲁棒的数据读取逻辑
records = []
DEVICE_NAME = torch.cuda.get_device_name(0).replace(" ", "_")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(CURRENT_DIR, f"result/collected_res_{DEVICE_NAME}.json")

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
    exit()

with open(file_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line or line in ['[', ']']:
            continue
        
        # 移除可能存在的干扰字符（如行尾逗号、Python 风格注释、单引号包裹）
        line = re.sub(r'#.*$', '', line).strip()
        if line.endswith(','): line = line[:-1].strip()
        line = line.lstrip("'").rstrip("'")
        line = line.replace(':oom', ':"oom"')
            
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

df = pd.DataFrame(records)

# 2. 数据清洗
def clean_avg(val):
    if val == "oom":
        return -2.0
    if val == "?" or val is None or pd.isna(val): 
        return -1.0
    try:
        return float(val)
    except:
        return -1.0

df['avg'] = df['avg'].apply(clean_avg)
df['op'] = df['op'].replace({'kf': 'keyformer'})

# 定义顺序
model_names = ['attn', 'corm', 'h2o', 'roco', 'keyformer', 'snapkv', 'gemma2', 'dsa_mla']
sys_names = ['torch', 'sdpa', 'flash_attn', 'flashtensor', 'flashtensor-3.5', 'tilelang', 'our']

# 使用 pivot_table 解决重复条目问题
def get_pivot_df(df_type):
    subset = df[df['type'] == df_type]
    if subset.empty:
        return pd.DataFrame(index=model_names, columns=sys_names).fillna(-1.0)
    
    # aggfunc='last' 表示如果数据重复，取最后一条
    pivot = subset.pivot_table(index='op', columns='method', values='avg', aggfunc='last')
    return pivot.reindex(index=model_names, columns=sys_names).fillna(-1.0)

op_df = get_pivot_df('op')
model_df = get_pivot_df('model')

# 3. 绘图配置
COLOR_DEF = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#82B0D2', '#BEB8DC', '#E7DAD2', '#999999']
HATCH_DEF = [None, None, None, None, None, None, '//']
MODEL_LABEL_MAP = {'attn': 'Attention', 'corm': 'CoRM', 'h2o': 'H2O', 'roco': 'RoCo', 'keyformer': 'Keyformer', 'snapkv': 'SnapKV', 'gemma2': 'Gemma-2', 'dsa_mla': 'DSA'}

def plot_data(dataset, title_prefix, filename, is_e2e):
    # 过滤掉在当前 dataset 中完全没有数据的系统
    active_sys_names = [s for s in sys_names if s in dataset.columns and not (dataset[s] == -1).all()]
    
    valid_models = [m for m in model_names if m in dataset.index and not (dataset.loc[m] == -1).all()]
    if not valid_models:
        print(f"No valid data to plot for {title_prefix}")
        return

    plt.rcParams.update({"figure.figsize": (14, 2.5), 'font.sans-serif': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig, axs = plt.subplots(1, len(valid_models), sharey=False)
    ylim = 1000 if is_e2e else 15
    
    if len(valid_models) == 1: axs = [axs]

    # 为当前图表中的所有系统分配统一的字母标识
    abc_full = [chr(ord('A') + i) for i in range(len(active_sys_names))]

    for i, (ax, model) in enumerate(zip(axs, valid_models)):
        perf = dataset.loc[model][active_sys_names]
        
        # 针对每个模型，只显示有数据的系统（不显示 NS）
        model_sys = [s for s in active_sys_names if perf[s] != -1]
        model_perf = perf[model_sys]
        model_abc = [abc_full[active_sys_names.index(s)] for s in model_sys]
        model_colors = [COLOR_DEF[sys_names.index(s)] for s in model_sys]
        model_hatches = [HATCH_DEF[sys_names.index(s)] for s in model_sys]

        norm_perf = model_perf.copy()
        norm_perf[norm_perf < 0] = 0 # 将 OOM (-2) 设为 0 以便绘图
        
        ax.bar(np.arange(len(model_sys)), norm_perf, color=model_colors, width=0.8, edgecolor='k', hatch=model_hatches)
        
        for j, (s, val) in enumerate(model_perf.items()):
            if val == -2:
                ax.text(j, ylim*0.05, 'OOM', ha='center', va='bottom', rotation=90, fontsize=8, color='red')
            elif val > ylim:
                ax.text(j, ylim*0.9, f'{val:.1f}', ha='center', va='top', rotation=90, fontsize=8)

        # Speedup (最优 baseline / our)
        if 'our' in model_sys:
            our_val = model_perf['our']
            baselines = model_perf.drop('our')
            valid_baselines = baselines[baselines > 0]
            if not valid_baselines.empty and our_val > 0:
                speedup = valid_baselines.min() / our_val
                our_idx = model_sys.index('our')
                ax.text(our_idx, norm_perf['our'], f'{speedup:.1f}x', fontweight='bold', ha='center', va='bottom', fontsize=8, clip_on=False)

        ax.set_xticks(range(len(model_abc)))
        ax.set_xticklabels(model_abc)
        ax.set_title(MODEL_LABEL_MAP.get(model, model), fontsize=10)
        ax.set_ylim(0, ylim)
        if i == 0: ax.set_ylabel('Time (ms)', fontsize=10)

    legend_handles = [mpatches.Patch(facecolor=COLOR_DEF[sys_names.index(s)], edgecolor='k', hatch=HATCH_DEF[sys_names.index(s)], 
                                     label=f'({abc_full[k]}) {s}') for k, s in enumerate(active_sys_names)]
    fig.legend(handles=legend_handles, loc='upper center', ncol=len(active_sys_names), bbox_to_anchor=(0.5, 1.15), frameon=False)
    
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')
    print(f"Saved: {filename}")

import torch
DEVICE_NAME = torch.cuda.get_device_name(0).replace(" ", "_")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
plot_data(op_df, "Kernel", f"{CURRENT_DIR}/result/fig_kernel_{DEVICE_NAME}.png", is_e2e=False)
plot_data(model_df, "E2E", f"{CURRENT_DIR}/result/fig_model_{DEVICE_NAME}.png", is_e2e=True)