import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

def bench_Gemma2_p2():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  avg_ms = triton.testing.do_bench(lambda: Gemma2_p2(rand_arg_0, rand_arg_1, rand_arg_2))
  print('[Gemma2_p2] avg_ms:', avg_ms)

def Gemma2_p2(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> torch.Tensor:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  empty_ptr_3 = torch.empty(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  grid = (1, 32, 32)
  Gemma2_p2_kernel[grid](tensor_0, tensor_1, tensor_2, empty_ptr_3, autotune_key)
  tensor_4 = empty_ptr_3
  return tensor_4

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def Gemma2_p2_kernel(
  arg_0,
  arg_1,
  arg_2,
  arg_3,
  autotune_key,
):
  pid_4 = tl.program_id(0)
  pid_5 = tl.program_id(1)
  pid_6 = tl.program_id(2)
  const_7 = 1.131250e+01
  const_8 = 5.000000e+01
  const_9 = 1.442695e+00
  const_10 = float('-inf')
  const_11 = 0.000000e+00
  const_12 = 0
  const_13 = 1
  const_14 = 4096
  const_15 = 128
  mul_16 = pid_5 * const_15
  mul_17 = mul_16 * const_14
  mul_18 = pid_6 * const_15
  add_19 = mul_17 + mul_18
  block_ptr_20 = tl.make_block_ptr(
    base=arg_0 + add_19,
    shape=(128, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  block_load_21 = tl.load(block_ptr_20)
  block_ptr_22 = tl.make_block_ptr(
    base=arg_3 + add_19,
    shape=(128, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  converted_23 = const_7
  div_24 = block_load_21 / converted_23
  div_24 = div_24.to(tl.float16)
  converted_25 = const_8
  div_26 = div_24 / converted_25
  div_26 = div_26.to(tl.float16)
  zero_27 = tl.zeros([128, 128], dtype=tl.float32)
  zero_28 = tl.zeros([128, 1], dtype=tl.float32)
  add_29 = mul_16 + const_15
  block_ptr_30 = tl.make_block_ptr(
    base=arg_2 + mul_18,
    shape=(128, 4096,),
    strides=(1, 4096,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(0, 1,),
  )
  block_ptr_31 = tl.make_block_ptr(
    base=arg_1 + mul_18,
    shape=(4096, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  for i_32 in range(const_12, add_29, const_15):
    block_load_33 = tl.load(block_ptr_30)
    block_load_34 = tl.load(block_ptr_31)
    dot_35 = tl.dot(div_26, block_load_33)
    where_36 = tl.zeros([128, 128], dtype=tl.float32)
    where_36 = tl.where(mul_16 + tl.arange(0, 128)[:, None] >= i_32 + tl.arange(0, 128)[None, :], where_36, float('-inf'))
    tanh_37 = tl.inline_asm_elementwise(
      asm='tanh.approx.f32 $0, $1;',
      constraints=('=r,r'),
      args=[dot_35],
      dtype=(tl.float32,),
      is_pure=True,
      pack=1,
    )
    mul_38 = tanh_37 * const_8
    mul_39 = mul_38 * const_9
    add_40 = mul_39 + where_36
    exp2_41 = tl.math.exp2(add_40)
    reduce_sum_42 = tl.sum(exp2_41, axis=1, keep_dims=True).to(tl.float32)
    reduce_sum_42 += zero_28
    converted_43 = exp2_41.to(tl.float16)
    dot_44 = tl.dot(converted_43, block_load_34)
    add_45 = zero_27 + dot_44
    block_advance_46 = tl.advance(block_ptr_30, (0, 128,))
    block_advance_47 = tl.advance(block_ptr_31, (128, 0,))
    block_ptr_30 = block_advance_46
    block_ptr_31 = block_advance_47
    zero_27 = add_45
    zero_28 = reduce_sum_42
  div_48 = zero_27 / zero_28
  converted_49 = div_48.to(tl.float16)
  block_store_50 = tl.store(block_ptr_22, converted_49)

def Gemma2(arg_0, arg_1, arg_2):
  k0_out_0 = Gemma2_p2(arg_0, arg_2, arg_1)
  return k0_out_0
