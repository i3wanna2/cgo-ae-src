import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

def bench_SnapKV_p1():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  avg_ms = triton.testing.do_bench(lambda: SnapKV_p1(rand_arg_0, rand_arg_1, rand_arg_2))
  print('[SnapKV_p1] avg_ms:', avg_ms)

def SnapKV_p1(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  empty_ptr_3 = torch.empty(1, 32, 4096, 1, dtype=torch.float32, device=dev)
  empty_ptr_4 = torch.empty(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  grid = (1, 32, 32)
  SnapKV_p1_kernel[grid](tensor_0, tensor_1, tensor_2, empty_ptr_3, empty_ptr_4, autotune_key)
  tensor_5 = empty_ptr_3
  tensor_6 = empty_ptr_4
  return tensor_5, tensor_6

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def SnapKV_p1_kernel(
  arg_0,
  arg_1,
  arg_2,
  arg_3,
  arg_4,
  autotune_key,
):
  pid_5 = tl.program_id(0)
  pid_6 = tl.program_id(1)
  pid_7 = tl.program_id(2)
  const_8 = 1.275311e-01
  const_9 = float('-inf')
  const_10 = 0.000000e+00
  const_11 = 0
  const_12 = 1
  const_13 = 4096
  const_14 = 128
  mul_15 = pid_6 * const_14
  mul_16 = mul_15 * const_13
  mul_17 = pid_7 * const_14
  add_18 = mul_16 + mul_17
  block_ptr_19 = tl.make_block_ptr(
    base=arg_0 + add_18,
    shape=(128, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  block_load_20 = tl.load(block_ptr_19)
  mul_21 = pid_7 * const_13
  add_22 = mul_15 + mul_21
  block_ptr_23 = tl.make_block_ptr(
    base=arg_3 + add_22,
    shape=(128, 1,),
    strides=(1, 1,),
    offsets=(0, 0,),
    block_shape=(128, 1,),
    order=(1, 0,),
  )
  block_ptr_24 = tl.make_block_ptr(
    base=arg_4 + add_18,
    shape=(128, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  converted_25 = const_8
  mul_26 = block_load_20 * converted_25
  mul_26 = mul_26.to(tl.float16)
  zero_27 = tl.zeros([128, 128], dtype=tl.float32)
  zero_28 = tl.zeros([128, 1], dtype=tl.float32)
  add_29 = mul_15 + const_14
  block_ptr_30 = tl.make_block_ptr(
    base=arg_2 + mul_17,
    shape=(128, 4096,),
    strides=(1, 4096,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(0, 1,),
  )
  block_ptr_31 = tl.make_block_ptr(
    base=arg_1 + mul_17,
    shape=(4096, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  for i_32 in range(const_11, add_29, const_14):
    block_load_33 = tl.load(block_ptr_30)
    block_load_34 = tl.load(block_ptr_31)
    dot_35 = tl.dot(mul_26, block_load_33)
    where_36 = tl.zeros([128, 128], dtype=tl.float32)
    where_36 = tl.where(mul_15 + tl.arange(0, 128)[:, None] >= i_32 + tl.arange(0, 128)[None, :], where_36, float('-inf'))
    add_37 = dot_35 + where_36
    exp2_38 = tl.math.exp2(add_37)
    reduce_sum_39 = tl.sum(exp2_38, axis=1, keep_dims=True).to(tl.float32)
    reduce_sum_39 += zero_28
    converted_40 = exp2_38.to(tl.float16)
    dot_41 = tl.dot(converted_40, block_load_34)
    add_42 = zero_27 + dot_41
    block_advance_43 = tl.advance(block_ptr_30, (0, 128,))
    block_advance_44 = tl.advance(block_ptr_31, (128, 0,))
    block_ptr_30 = block_advance_43
    block_ptr_31 = block_advance_44
    zero_27 = add_42
    zero_28 = reduce_sum_39
  div_45 = zero_27 / zero_28
  converted_46 = div_45.to(tl.float16)
  block_store_47 = tl.store(block_ptr_23, zero_28)
  block_store_48 = tl.store(block_ptr_24, converted_46)

def bench_SnapKV_p4():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 32, 4096, 1, dtype=torch.float32, device=dev)
  avg_ms = triton.testing.do_bench(lambda: SnapKV_p4(rand_arg_0, rand_arg_1, rand_arg_2))
  print('[SnapKV_p4] avg_ms:', avg_ms)

def SnapKV_p4(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> torch.Tensor:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  empty_ptr_3 = torch.empty(1, 32, 4096, dtype=torch.float32, device=dev)
  grid = (1, 32, 32)
  SnapKV_p4_kernel[grid](tensor_0, tensor_1, tensor_2, empty_ptr_3, autotune_key)
  tensor_4 = empty_ptr_3
  return tensor_4

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def SnapKV_p4_kernel(
  arg_0,
  arg_1,
  arg_2,
  arg_3,
  autotune_key,
):
  pid_4 = tl.program_id(0)
  pid_5 = tl.program_id(1)
  pid_6 = tl.program_id(2)
  const_7 = 1.275311e-01
  const_8 = float('-inf')
  const_9 = 0.000000e+00
  const_10 = 4096
  const_11 = 128
  const_12 = 1
  mul_13 = pid_6 * const_11
  mul_14 = pid_5 * const_11
  mul_15 = mul_14 * const_10
  add_16 = mul_15 + mul_13
  block_ptr_17 = tl.make_block_ptr(
    base=arg_1 + add_16,
    shape=(128, 128,),
    strides=(1, 4096,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(0, 1,),
  )
  block_load_18 = tl.load(block_ptr_17)
  mul_19 = pid_6 * const_10
  add_20 = mul_14 + mul_19
  block_ptr_21 = tl.make_block_ptr(
    base=arg_3 + add_20,
    shape=(128,),
    strides=(1,),
    offsets=(0,),
    block_shape=(128,),
    order=(0,),
  )
  converted_22 = const_7
  mul_23 = block_load_18 * converted_22
  mul_23 = mul_23.to(tl.float16)
  zero_24 = tl.zeros([128], dtype=tl.float32)
  block_ptr_25 = tl.make_block_ptr(
    base=arg_0 + add_16,
    shape=(4096, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  block_ptr_26 = tl.make_block_ptr(
    base=arg_2 + add_20,
    shape=(4096,),
    strides=(1,),
    offsets=(0,),
    block_shape=(128,),
    order=(0,),
  )
  for i_27 in range(mul_14, const_10, const_11):
    block_load_28 = tl.load(block_ptr_25)
    block_load_29 = tl.load(block_ptr_26)
    where_30 = tl.zeros([128, 128], dtype=tl.float32)
    where_30 = tl.where(i_27 + tl.arange(0, 128)[:, None] >= mul_14 + tl.arange(0, 128)[None, :], where_30, float('-inf'))
    dot_31 = tl.dot(block_load_28, mul_23)
    add_32 = dot_31 + where_30
    exp2_33 = tl.math.exp2(add_32)
    unsqueeze_34 = block_load_29[:, None]
    div_35 = exp2_33 / unsqueeze_34
    reduce_sum_36 = tl.sum(div_35, axis=0, keep_dims=False).to(tl.float32)
    reduce_sum_36 += zero_24
    block_advance_37 = tl.advance(block_ptr_25, (128, 0,))
    block_advance_38 = tl.advance(block_ptr_26, (128,))
    block_ptr_25 = block_advance_37
    block_ptr_26 = block_advance_38
    zero_24 = reduce_sum_36
  block_store_39 = tl.store(block_ptr_21, zero_24)

def bench_SnapKV_uncovered_0():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 32, 4096, dtype=torch.float32, device=dev)
  avg_ms = triton.testing.do_bench(lambda: SnapKV_uncovered_0(rand_arg_0))
  print('[SnapKV_uncovered_0] avg_ms:', avg_ms)

def SnapKV_uncovered_0(arg0: torch.Tensor) -> torch.Tensor:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  avg_pool_0 = F.avg_pool1d(arg0, kernel_size=5, stride=1, padding=2, ceil_mode=False, count_include_pad=True)
  return avg_pool_0

def SnapKV(arg_0, arg_1, arg_2):
  k0_out_0, k0_out_1 = SnapKV_p1(arg_0, arg_2, arg_1)
  k1_out_0 = SnapKV_p4(arg_0, arg_1, k0_out_0)
  k2_out_0 = SnapKV_uncovered_0(k1_out_0)
  return k0_out_1, k2_out_0
