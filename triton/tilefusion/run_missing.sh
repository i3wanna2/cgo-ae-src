#!/bin/bash

# Get device name
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "NVIDIA A100"; then
    DEVICE="A100"
elif nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "NVIDIA H800"; then
    DEVICE="H800"
elif nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "NVIDIA H100"; then
    DEVICE="H100"
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
PARENT_DIR=$(dirname "$SCRIPT_DIR")
LOG_DIR_FT="$SCRIPT_DIR/logs_seqlen_ft"
LOG_DIR_STD="$SCRIPT_DIR/logs_seqlen"

mkdir -p "$LOG_DIR_FT"
mkdir -p "$LOG_DIR_STD"

# echo "Re-collecting results before checking..."
conda run --no-capture-output -n base python3 "$SCRIPT_DIR/collect_seqlen_results.py"
conda run --no-capture-output -n base python3 "$SCRIPT_DIR/collect_seqlen_results_ft.py"

echo "Checking for missing results..."
MISSING_TASKS=$(conda run --no-capture-output -n base python3 "$SCRIPT_DIR/check_missing.py")

if [[ -z "$MISSING_TASKS" ]]; then
    echo "No missing tasks found!"
    exit 0
fi

echo "Found missing tasks. Starting execution..."

# Use a while loop to read the output of check_missing.py
echo "$MISSING_TASKS" | while read -r category model sys seqlen type; do
    echo "------------------------------------------------"
    echo "Task: $category | Model: $model | Sys: $sys | Seqlen: $seqlen | Type: $type"
    
    if [[ "$category" == "ft" ]]; then
        # FlashTensor tasks
        env_name="baseline"
        cmd_sys="$sys"
        if [[ "$sys" == "flashtensor" ]]; then
            env_name="flashtensor"
            cmd_sys="our"
        fi

        if [[ "$type" == "op" ]]; then
            log_file="$LOG_DIR_FT/${model}_kernel_${sys}_${seqlen}.log"
            echo "Running Kernel: $model ($sys) S=$seqlen [$env_name env]"
            conda run --no-capture-output -n "$env_name" python3 /home/meiziyuan/FlashTensor-AE/run_kernel.py --model "$model" --system "$cmd_sys" --seqlen "$seqlen" 2>&1 | tee "$log_file"
        else
            log_file="$LOG_DIR_FT/${model}_model_${sys}_${seqlen}.log"
            echo "Running E2E: $model ($sys) S=$seqlen [$env_name env]"
            conda run --no-capture-output -n "$env_name" python3 /home/meiziyuan/FlashTensor-AE/run_e2e.py --model "$model" --system "$cmd_sys" --platform "$DEVICE" --seqlen "$seqlen" 2>&1 | tee "$log_file"
        fi
    else
        # Standard/Our tasks (use base environment)
        # Map flashtensor-3.5 back to flashtensor for command line and filename
        cmd_sys="$sys"
        if [[ "$sys" == "flashtensor-3.5" ]]; then
            cmd_sys="flashtensor"
        fi

        log_file="$LOG_DIR_STD/${model}_model_${cmd_sys}_${seqlen}.log"
        model_file="$SCRIPT_DIR/models/${model}_model.py"
        if [[ ! -f "$model_file" ]]; then
            # Handle cases where model name doesn't match file name exactly if needed
            model_file="$SCRIPT_DIR/models/${model}.py"
        fi
        
        echo "Running Std: $model ($cmd_sys) S=$seqlen [base env]"
        # Standard scripts usually run both kernel and model when --mode both is used
        # We run it once and it should fill both missing entries in the next check
        conda run --no-capture-output -n base python3 "$model_file" --system "$cmd_sys" --platform "$DEVICE" --seqlen "$seqlen" --mode both 2>&1 | tee "$log_file"
    fi
done

echo "------------------------------------------------"
echo "Execution finished. Re-collecting results..."
conda run --no-capture-output -n base python3 "$SCRIPT_DIR/collect_seqlen_results.py"
conda run --no-capture-output -n base python3 "$SCRIPT_DIR/collect_seqlen_results_ft.py"
conda run --no-capture-output -n base python3 "$SCRIPT_DIR/plot_seqlen.py"
echo "Done."
