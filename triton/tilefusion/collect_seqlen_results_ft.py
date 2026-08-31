import os
import json
import re
import torch

DEVICE_NAME = torch.cuda.get_device_name(0).replace(" ", "_")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, "logs_seqlen_ft")
OUTPUT_FILE = os.path.join(CURRENT_DIR, f"result/collected_seqlen_ft_{DEVICE_NAME}.json")

# Mapping of op names to their shapes as seen in res.json
SHAPE_MAP = {
    "attn": "[1,32,seqlen,128]",
    "corm": "[1,32,seqlen,128]",
    "gemma2": "[1,32,seqlen,128]",
    "h2o": "[1,32,seqlen,128]",
    "kf": "[1,32,seqlen,128]",
    "roco": "[1,32,seqlen,128]",
    "snapkv": "[1,32,seqlen,128]",
    "dsa_mla": "[1,128,seqlen,576]",
}

def parse_logs():
    results = []
    if not os.path.exists(LOG_DIR):
        print(f"Log directory {LOG_DIR} does not exist.")
        return

    for filename in sorted(os.listdir(LOG_DIR)):
        if not filename.endswith(".log"):
            continue
        
        # Filename format: {op}_{type}_{method}_{seqlen}.log
        # type is either 'kernel' or 'model'
        match = re.match(r"(.+)_(kernel|model)_(.+)_(\d+)\.log", filename)
        if not match:
            continue
        
        op_name_from_file = match.group(1)
        log_type = match.group(2) # 'kernel' or 'model'
        method = match.group(3)
        seqlen = int(match.group(4))
        
        # Map 'kernel' to 'op' and 'model' to 'model' to match original script's output format
        result_type = "op" if log_type == "kernel" else "model"
        
        filepath = os.path.join(LOG_DIR, filename)
        with open(filepath, "r") as f:
            for line in f:
                # Look for benchmark result
                # Example: [dynamo] avg 36.6927 ms, ...
                match_res = re.search(r"\[(.+)\] avg ([\d\.]+) ms", line)
                if match_res:
                    avg = float(match_res.group(2))
                    results.append({
                        "hw": DEVICE_NAME,
                        "method": method,
                        "type": result_type,
                        "op": op_name_from_file,
                        "seqlen": seqlen,
                        "shape": SHAPE_MAP.get(op_name_from_file, "unknown").replace("seqlen", str(seqlen)),
                        "avg": avg
                    })
                    break # Found the result for this file

                if "torch.OutOfMemoryError: CUDA out of memory" in line:
                    results.append({
                        "hw": DEVICE_NAME,
                        "method": method,
                        "type": result_type,
                        "op": op_name_from_file,
                        "seqlen": seqlen,
                        "shape": SHAPE_MAP.get(op_name_from_file, "unknown").replace("seqlen", str(seqlen)),
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
