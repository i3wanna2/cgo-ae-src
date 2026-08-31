import os
import json
import re
import torch
DEVICE_NAME = torch.cuda.get_device_name(0).replace(" ", "_")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, "logs")
OUTPUT_FILE = os.path.join(CURRENT_DIR, f"result/collected_res_{DEVICE_NAME}.json")

# Mapping of op names to their shapes as seen in res.json
SHAPE_MAP = {
    "attn": "[1,32,4096,128]",
    "corm": "[1,32,4096,128]",
    "gemma2": "[1,32,4096,128]",
    "h2o": "[1,32,4096,128]",
    "kf": "[1,32,4096,128]",
    "roco": "[1,32,4096,128]",
    "snapkv": "[1,32,4096,128]",
    "dsa_mla": "[1,128,4096,576]",
}

def parse_logs():
    results = []
    if not os.path.exists(LOG_DIR):
        print(f"Log directory {LOG_DIR} does not exist.")
        return

    for filename in os.listdir(LOG_DIR):
        if not filename.endswith(".log"):
            continue
        
        # Filename format: {op}_model_{method}.log
        match = re.match(r"(.+)_model_(.+)\.log", filename)
        if not match:
            continue
        
        op_name_from_file = match.group(1)
        method = match.group(2)
        
        filepath = os.path.join(LOG_DIR, filename)
        with open(filepath, "r") as f:
            for line in f:
                # Look for kernel benchmark
                # Example: corm kernel: runs=100 mean=10.479 ms ...
                kernel_match = re.search(r"(\w+) kernel:.*mean=([\d\.]+) ms", line)
                if kernel_match:
                    op = kernel_match.group(1)
                    avg = float(kernel_match.group(2))
                    results.append({
                        "hw": DEVICE_NAME,
                        "method": method,
                        "type": "op",
                        "op": op,
                        "shape": SHAPE_MAP.get(op, "unknown"),
                        "avg": avg
                    })
                
                # Look for E2E benchmark
                # Example: corm E2E: runs=50 mean=406.939 ms ...
                e2e_match = re.search(r"(\w+) E2E:.*mean=([\d\.]+) ms", line)
                if e2e_match:
                    op = e2e_match.group(1)
                    avg = float(e2e_match.group(2))
                    results.append({
                        "hw": DEVICE_NAME,
                        "method": method,
                        "type": "model",
                        "op": op,
                        "shape": SHAPE_MAP.get(op, "unknown"),
                        "avg": avg
                    })

                if "torch.OutOfMemoryError: CUDA out of memory" in line:
                    results.append({
                        "hw": DEVICE_NAME,
                        "method": method,
                        "type": "model",
                        "op": op_name_from_file,
                        "shape": SHAPE_MAP.get(op_name_from_file, "unknown"),
                        "avg": "oom"
                    })
                    break

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    
    print(f"Collected {len(results)} results into {OUTPUT_FILE}")

if __name__ == "__main__":
    parse_logs()
