import torch
import numpy as np
import onnx
from asuka_exp.utils import torch_module_to_onnx

def compile(model, input_names, inputs, output_names, system):
  if system == 'torch':
    f = model
  elif system == 'dynamo':
    torch._dynamo.reset()
    f = torch.compile(model) 
  elif system == 'tensorrt':
    from asuka_exp.trtllm_utils import trt_build_engine_from_onnx, trt_build_independent_runtime
    onnx_model = torch_module_to_onnx(
      module=model,
      input_names=input_names,
      inputs=inputs,
      output_names=output_names,
    )
    engine = trt_build_engine_from_onnx(onnx_model)
    f = trt_build_independent_runtime(engine)
  elif system == 'xla':
    import torch_xla.core.xla_model as xm
    def _f(*args):
      o = model(*args)
      xm.mark_step()
      xm.wait_device_ops()
      return o
    f = _f
  elif system == 'tvm':
    from asuka_exp.tvm_utils import meta_scheduler_tune, tvm_build_independent_runtime
    lib = meta_scheduler_tune(
      module=model,
      input_names=input_names,
      inputs=inputs,
      output_names=output_names,
      # num_trials_per_iter=64,
      # max_trials_per_task=1000,
      num_trials_per_iter=4,
      max_trials_per_task=128,
      exported_lib_path=None,
    )
    f = tvm_build_independent_runtime(lib, input_names, output_names)
  elif system == 'our':
    from asuka.translate import asuka_from_onnx
    from asuka.transform import fission
    from asuka.transform.common import simplify
    from asuka.partition.connected import Connected
    onnx_model = torch_module_to_onnx(
      module=model,
      input_names=input_names,
      inputs=inputs,
      output_names=output_names,
      simplify=False,
    )
    print(onnx.helper.printable_graph(onnx_model.graph), flush=True)
    func_name = model.__class__.__name__

    import time
    tik = time.time()
    module = asuka_from_onnx(onnx_model, func_name)
    module.dump()
    fission(module)
    simplify(module)
    partition = Connected(module, func_name)
    partition.module.dump()
    partition.optimize()
    tok = time.time()
    tuning_s = tok - tik
    print(f"tuning time: {tuning_s} sec", flush=True)
    perf = partition.profile()
    py_str = partition.codegen(perf)

    our = {}
    import tempfile
    import importlib
    import sys
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
      f.write(py_str)
      path = f.name
    print(f"write code to {path}", flush=True)
    spec = importlib.util.spec_from_file_location('our', path)
    pymod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pymod)

    f = getattr(pymod, func_name)
  elif system == 'flashinfer':
    import flashinfer
    model_name = model.__class__.__name__
    if model_name == 'Attn':
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        q_len = q.shape[1]
        head_num = q.shape[2]
        head_dim = q.shape[3]
        kv_len = k.shape[1]
        kv_head_num = k.shape[2]
        batch_size = q.shape[0]
        assert batch_size == 1
 
        q = q.view(q_len, head_num, head_dim)
        k = k.view(kv_len, kv_head_num, head_dim)
        v = v.view(kv_len, kv_head_num, head_dim)
        out = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
        return out.view(batch_size, q_len, head_num, head_dim)
    else:
      assert model_name == 'Gemma2'
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        q_len = q.shape[1]
        head_num = q.shape[2]
        head_dim = q.shape[3]
        kv_len = k.shape[1]
        kv_head_num = k.shape[2]
        batch_size = q.shape[0]
        assert batch_size == 1
 
        q = q.view(q_len, head_num, head_dim)
        k = k.view(kv_len, kv_head_num, head_dim)
        v = v.view(kv_len, kv_head_num, head_dim)
        out = flashinfer.single_prefill_with_kv_cache(q, k, v, logits_soft_cap=50.0, causal=True)
        return out.view(batch_size, q_len, head_num, head_dim)
    f = _f
  elif system == 'flashattn':
    from flash_attn.flash_attn_interface import flash_attn_func
    model_name = model.__class__.__name__
    if model_name == 'Attn':
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        out = flash_attn_func(q, k, v, causal=True)
        return out
    else:
      assert model_name == 'Gemma2'
      def _f(*args):
        q, k, v = args[0], args[1], args[2]
        out = flash_attn_func(q, k, v, softcap=50.0, causal=True)
        return out
    f = _f
  else:
    raise NotImplementedError(f"system {system} not implemented")
  
  return f
