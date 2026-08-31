#!/bin/bash
# set -x
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "NVIDIA A100"; then
    DEVICE="A100"
elif nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "NVIDIA H800"; then
    DEVICE="H800"
elif nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "NVIDIA H100"; then
    DEVICE="H100"
fi
# Enable pipefail to catch errors in piped commands
set -o pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
parent_dir=$(dirname "$SCRIPT_DIR")
echo "$parent_dir"

if [[ "$DEVICE" == "A100" ]]; then
    export PYTHONPATH="$parent_dir:$PYTHONPATH"
fi

LOG_DIR="$parent_dir/tilefusion/logs_seqlen_ft"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"
# Iterate over all .py files in the models directory

SYSTEMS=("torch" "dynamo" "tvm" )

if [[ "$DEVICE" == "A100" ]]; then
    SEQLENS=(1024 2048 3072 4096)
else
    SEQLENS=(1024 2048 4096 8192)
fi
MODELS=("attn" "corm" "gemma2" "h2o" "kf" "roco" "snapkv")
for model in "${MODELS[@]}"; do
    for sys in "${SYSTEMS[@]}"; do
        for seqlen in "${SEQLENS[@]}"; do

            echo "========================================"
            echo "Running flashtensor $model with system: $sys, sequence length: $seqlen on device: $DEVICE"
            echo "========================================"
            
            log_file="$LOG_DIR/${model%.py}_kernel_${sys}_${seqlen}.log"
            echo "Logging to $log_file"
            if [[ "$DEVICE" == "A100" ]]; then
                source ~/flashtensor/bin/activate
            fi

            # Run the model script and use tee to show output and save to log file
            python3 /home/meiziyuan/FlashTensor-AE/run_kernel.py --model "$model" --system "$sys" --seqlen "$seqlen" 2>&1 | tee "$log_file"
            
            if [ $? -eq 0 ]; then
                echo "Successfully finished flashtensor $model ($sys) with sequence length $seqlen"
            else
                echo "Error running flashtensor $model ($sys) with sequence length $seqlen. Check $log_file for details."
            fi
            echo ""

            log_file="$LOG_DIR/${model%.py}_model_${sys}_${seqlen}.log"
            echo "Logging to $log_file"
            if [[ "$DEVICE" == "A100" ]]; then
                source ~/flashtensor/bin/activate
            fi

            # Run the model script and use tee to show output and save to log file
            python3 /home/meiziyuan/FlashTensor-AE/run_e2e.py --model "$model" --system "$sys" --platform "$DEVICE" --seqlen "$seqlen" 2>&1 | tee "$log_file"
            
            if [ $? -eq 0 ]; then
                echo "Successfully finished flashtensor $model ($sys) with sequence length $seqlen"
            else
                echo "Error running flashtensor $model ($sys) with sequence length $seqlen. Check $log_file for details."
            fi
            echo ""

        done
    done
done

# echo "All models have been processed."
# python3 "$parent_dir/tilefusion/collect_seqlen_results.py"
# python3 "$parent_dir/tilefusion/plot_seqlen.py"
# echo "Results collected and plots generated."
