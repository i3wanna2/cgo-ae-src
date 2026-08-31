import os
import json
import torch

DEVICE_NAME = torch.cuda.get_device_name(0).replace(" ", "_")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Files to check
FILE_STD = os.path.join(CURRENT_DIR, f"result/collected_seqlen_{DEVICE_NAME}.json")
FILE_FT = os.path.join(CURRENT_DIR, f"result/collected_seqlen_ft_{DEVICE_NAME}.json")

# Expected configurations
if "A100" in DEVICE_NAME:
    SEQLENS = [1024, 2048, 3072, 4096]
else:
    SEQLENS = [1024, 2048, 4096, 8192]

MODELS_FT = ["attn", "corm", "gemma2", "h2o", "kf", "roco", "snapkv"]
SYSTEMS_FT = ["torch", "dynamo", "flashtensor","tensorrt", "tvm"]

MODELS_STD = ["attn", "corm", "gemma2", "h2o", "kf", "roco", "snapkv", "dsa_mla"]
# Note: dsa_mla uses tilelang instead of flashtensor
TYPES = ["op", "model"]

def load_results(file_path):
    results = set()
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("avg") == "oom":
                        continue
                    # Use a tuple as a unique key
                    # (model, method, seqlen, type)
                    # Note: in JSON, 'op' field is the model name, 'type' is 'op' or 'model'
                    results.add((data["op"], data["method"], int(data["seqlen"]), data["type"]))
                except:
                    continue
    return results

def check():
    results_std = load_results(FILE_STD)
    results_ft = load_results(FILE_FT)
    
    missing = []

    # Check FT
    # for model in MODELS_FT:
    #     for sys in SYSTEMS_FT:
    #         for seqlen in SEQLENS:
    #             for t in TYPES:
    #                 if (model, sys, seqlen, t) not in results_ft:
    #                     missing.append(("ft", model, sys, seqlen, t))

    # # Check STD
    # for model in MODELS_STD:
    #     systems = [ "tensorrt", "our", "tilelang", "torch", "dynamo",] if model == "dsa_mla" else ["our", "flashtensor-3.5"]
    #     for sys in systems:
    #         for seqlen in SEQLENS:
    #             # for t in TYPES:
    #             if (model, sys, seqlen, t) not in results_std:
    #                 missing.append(("std", model, sys, seqlen, "op"))
                    
    MODELS_STD_RETEST = ["dsa_mla"] 
    systems = ["tilelang","tilelang-ws" ]
    for model in MODELS_STD_RETEST:
        for sys in systems:
            for seqlen in SEQLENS:
                missing.append(("std", model, sys, seqlen, "op"))
    # Output for bash consumption: CATEGORY MODEL SYSTEM SEQLEN TYPE
    print(len(missing), "missing entries:")
    for m in missing:
        print(f"{m[0]} {m[1]} {m[2]} {m[3]} {m[4]}")

if __name__ == "__main__":
    check()
