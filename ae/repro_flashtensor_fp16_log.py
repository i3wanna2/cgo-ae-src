#!/usr/bin/env python3
"""Reproduce FlashTensor's FP16 log lowering failure.

FlashTensor emits `name = tl.math.log(operand)` with no cast to fp32
(see FlashTensor-AE/3rd/asuka/lib/Translate/to_triton.cpp, LogOp).
Triton only accepts fp32/fp64 for tl.math.log.

Stock KeyFormer never hits this: prepare() builds exp_rand as fp32
(asuka_exp/cases/kernels/kf.py). This script lowers the same `-x.log()`
op on an fp16 tensor through that compiler and launches the generated
kernel. It does not modify any existing source.

Expected result: Triton raises
    ValueError: Expected dtype ['fp32', 'fp64'] but got fp16
on the generated `tl.math.log(...)` line. Exit code 0 means that error
was reproduced.

Run inside the container, on any GPU:

    source /workspace/ae/env_flashtensor.sh
    "$AE_PY" /workspace/ae/repro_flashtensor_fp16_log.py
"""
import sys
import tempfile
import traceback

sys.path.insert(0, "/workspace/FlashTensor-AE")

import torch
import torch.nn as nn

from asuka.translate import asuka_from_onnx
from asuka.transform.common import optimize, kernel_to_py
from asuka.partition.kernel import Kernel
from asuka_exp.utils import torch_module_to_onnx


class LogNeg(nn.Module):
    def forward(self, x):
        return -x.log()


def _is_fp16_log_error(exc: BaseException) -> bool:
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        text = str(cur)
        if "Expected dtype" in text and "fp16" in text:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def main() -> int:
    print("FlashTensor FP16 tl.math.log repro")
    print("gpu", torch.cuda.get_device_name(0))
    model = LogNeg().eval().cuda()
    x = 1 + torch.randn(256, 256, device="cuda", dtype=torch.float16).abs()
    print(f"input dtype={x.dtype} shape={tuple(x.shape)}")

    onnx_model = torch_module_to_onnx(
        module=model,
        input_names=["x"],
        inputs=[x],
        output_names=["y"],
        simplify=False,
    )
    module = asuka_from_onnx(onnx_model, "LogNeg")
    kernel = Kernel.from_all_ops(module, "LogNeg", "logneg_k")
    optimize(kernel.kernel, context=module.context)
    py_str = kernel_to_py(kernel.kernel, add_import=True, add_benchmark=False)
    print("--- generated python ---")
    print(py_str)
    print("--- end generated python ---", flush=True)
    if "tl.math.log(" not in py_str:
        print("generated source has no tl.math.log")
        return 1

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(py_str)
        path = f.name
    ns = {}
    exec(compile(py_str, path, "exec"), ns)
    fn = ns.get("logneg_k")
    if fn is None:
        print("generated module has no logneg_k")
        return 1
    try:
        fn(x)
        torch.cuda.synchronize()
    except Exception as exc:
        if _is_fp16_log_error(exc):
            print("REPRODUCED")
            print("ValueError: Expected dtype ['fp32', 'fp64'] but got fp16")
            traceback.print_exc()
            return 0
        print("unexpected failure")
        traceback.print_exc()
        return 1
    print("unexpected success: fp16 log kernel launched")
    return 1


if __name__ == "__main__":
    sys.exit(main())
