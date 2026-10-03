"""KeyFormer via FlashTensor-AE, with FP16 noise.

Same path as FlashTensor-AE/run_kernel.py. The only change is that
``exp_rand`` is cast to FP16 after ``prepare()``. FlashTensor itself is not
run through this file: it keeps the stock FP32 ``prepare()``.
"""
import sys

sys.path.insert(0, "/workspace/FlashTensor-AE")

import click
import numpy as np
import random
import time
import torch

from asuka_exp.cases.kernels import KERNEL_ZOO
from asuka_exp.utils import compare, display, perf
from compile import compile


def gflops_and_mib(seqlen, f, *args):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_peak_mem_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"{start_peak_mem_mib=}", flush=True)
    f(*args)
    end_peak_mem_mib = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"{end_peak_mem_mib=}", flush=True)

    batch_size = 1
    head_num = 32
    head_dim = 128
    gflops = 4 * batch_size * head_num * seqlen * seqlen * head_dim * 1e-9 / 2
    return gflops, end_peak_mem_mib


@click.command()
@click.option("--model", "-m", default="kf", help="Model name")
@click.option("--system", "-s", required=True, help="dynamo, tensorrt, or tvm")
@click.option("--seqlen", default=4096, help="seqlen")
@click.option("--show_result", is_flag=True, help="show result")
@click.option("--check/--no-check", default=True, help="Check correctness against Torch")
def main(model, system, seqlen, show_result, check):
    print(f"{model=} {system=} {seqlen=}")
    assert model == "kf", model
    assert system in ("dynamo", "tensorrt", "tvm"), system
    assert model in KERNEL_ZOO, f"model {model} not found in KERNEL_ZOO {KERNEL_ZOO.keys()}"

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cls = KERNEL_ZOO[model]
    model = cls()
    model = model.eval().cuda()
    specs = model.prepare(q_len=seqlen, kv_len=seqlen)
    noise = specs["input"]["exp_rand"]
    if noise.dtype != torch.float16:
        specs["input"]["exp_rand"] = noise.to(dtype=torch.float16)
        noise = specs["input"]["exp_rand"]
    print(
        f"exp_rand dtype={noise.dtype} bytes={noise.numel() * noise.element_size()}",
        flush=True,
    )
    input_names = list(specs["input"].keys())
    inputs = [specs["input"][name] for name in input_names]
    output_names = specs["output"]

    print(f"{input_names=}")
    print(f"{output_names=}")
    print(f"{check=}")

    run = 10
    warmup = 100
    print(f"Starting compilation and tuning for {system}...", flush=True)
    compile_start_time = time.perf_counter()
    f = compile(
        model=model,
        input_names=input_names,
        inputs=inputs,
        output_names=output_names,
        system=system,
    )
    compile_end_time = time.perf_counter()
    compile_time = compile_end_time - compile_start_time
    print(f"Compilation and tuning time for {system}: {compile_time:.4f} seconds", flush=True)

    gflops, mib = gflops_and_mib(seqlen, f, *inputs)
    print(f"{gflops=}", flush=True)
    print(f"{mib=}", flush=True)

    if check:
        print("Checking correctness against Torch...")
        try:
            outs_ref = model(*inputs)
            outs = f(*inputs)
            torch.cuda.synchronize()
            compare(outs, outs_ref, output_names)
            got = outs if isinstance(outs, (list, tuple)) else [outs]
            ref = outs_ref if isinstance(outs_ref, (list, tuple)) else [outs_ref]
            names = output_names if isinstance(outs, (list, tuple)) else output_names[:1]
            for out, baseline, name in zip(got, ref, names):
                if (not torch.isfinite(out).all()) or (not torch.isfinite(baseline).all()):
                    raise AssertionError(f"{name} contains NaN or Inf")
                torch.testing.assert_close(out, baseline, rtol=1e-3, atol=1e-2)
            print("Correctness check passed!")
        except Exception as exc:
            print(f"Correctness check failed: {exc}")
        if show_result:
            display(outs, outs_ref, output_names)

    perf(
        label=system,
        f=f,
        args=inputs,
        run=run,
        warmup=warmup,
        profile=True,
        gflops=gflops,
    )


if __name__ == "__main__":
    main()
