#!/bin/bash

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
# Directory containing the models
MODELS_DIR="$parent_dir/tilefusion/models"
LOG_DIR="$parent_dir/tilefusion/logs"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

if [[ "$DEVICE" == "A100" ]]; then
    export PYTHONPATH="$parent_dir:$PYTHONPATH"
fi

# List of models to exclude
EXCLUDE=("dsa_model.py" "__init__.py")

# Function to check if a file should be excluded
is_excluded() {
    local file=$1
    for ex in "${EXCLUDE[@]}"; do
        if [[ "$file" == "$ex" ]]; then
            return 0
        fi
    done
    return 1
}

# Iterate over all .py files in the models directory
for model_file in "$MODELS_DIR"/*.py; do
    filename=$(basename "$model_file")
    
    if is_excluded "$filename"; then
        echo "Skipping excluded model: $filename"
        continue
    fi

    # Determine systems to test
    if [[ "$filename" == "dsa_mla_model.py" ]]; then
        SYSTEMS=("tilelang" "torch" "our")
    else
        SYSTEMS=("flashtensor" "torch" "our")
    fi

    for sys in "${SYSTEMS[@]}"; do
        echo "========================================"
        echo "Running model: $filename with system: $sys, on device: $DEVICE"
        echo "========================================"
        
        log_file="$LOG_DIR/${filename%.py}_${sys}.log"
        echo "Logging to $log_file"
        if [[ "$DEVICE" == "A100" ]]; then
            source ~/tilefusion/bin/activate
        fi

        # Run the model script and use tee to show output and save to log file
        python3 "$model_file" --system "$sys" --platform "$DEVICE" 2>&1 | tee "$log_file"
        
        if [ $? -eq 0 ]; then
            echo "Successfully finished $filename ($sys)"
        else
            echo "Error running $filename ($sys). Check $log_file for details."
        fi
        echo ""
    done
done

echo "All models have been processed."
python3 "$parent_dir/tilefusion/collect_results.py"
python3 "$parent_dir/tilefusion/plot.py"
echo "Results collected and plots generated."
