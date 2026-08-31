import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

def bench_RoCo_p5():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  avg_ms = triton.testing.do_bench(lambda: RoCo_p5(rand_arg_0, rand_arg_1, rand_arg_2))
  print('[RoCo_p5] avg_ms:', avg_ms)

def RoCo_p5(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  empty_ptr_3 = torch.empty(1, 32, 4096, 1, dtype=torch.float32, device=dev)
  empty_ptr_4 = torch.empty(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  grid = (1, 32, 32)
  RoCo_p5_kernel[grid](tensor_0, tensor_1, tensor_2, empty_ptr_3, empty_ptr_4, autotune_key)
  tensor_5 = empty_ptr_3
  tensor_6 = empty_ptr_4
  return tensor_5, tensor_6

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def RoCo_p5_kernel(
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

def bench_RoCo_p4():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 32, 4096, 1, dtype=torch.float32, device=dev)
  avg_ms = triton.testing.do_bench(lambda: RoCo_p4(rand_arg_0, rand_arg_1, rand_arg_2))
  print('[RoCo_p4] avg_ms:', avg_ms)

def RoCo_p4(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  empty_ptr_3 = torch.empty(1, 32, 4096, dtype=torch.float32, device=dev)
  empty_ptr_4 = torch.empty(1, 32, 4096, dtype=torch.float32, device=dev)
  grid = (1, 32, 32)
  RoCo_p4_kernel[grid](tensor_0, tensor_1, tensor_2, empty_ptr_3, empty_ptr_4, autotune_key)
  tensor_5 = empty_ptr_3
  tensor_6 = empty_ptr_4
  return tensor_5, tensor_6

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def RoCo_p4_kernel(
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
  const_8 = 2.000000e+00
  const_9 = 1.275311e-01
  const_10 = float('-inf')
  const_11 = 0.000000e+00
  const_12 = 4096
  const_13 = 128
  const_14 = 1
  mul_15 = pid_7 * const_13
  mul_16 = pid_6 * const_13
  mul_17 = mul_16 * const_12
  add_18 = mul_17 + mul_15
  block_ptr_19 = tl.make_block_ptr(
    base=arg_1 + add_18,
    shape=(128, 128,),
    strides=(1, 4096,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(0, 1,),
  )
  block_load_20 = tl.load(block_ptr_19)
  mul_21 = pid_7 * const_12
  add_22 = mul_16 + mul_21
  block_ptr_23 = tl.make_block_ptr(
    base=arg_3 + add_22,
    shape=(128,),
    strides=(1,),
    offsets=(0,),
    block_shape=(128,),
    order=(0,),
  )
  block_ptr_24 = tl.make_block_ptr(
    base=arg_4 + add_22,
    shape=(128,),
    strides=(1,),
    offsets=(0,),
    block_shape=(128,),
    order=(0,),
  )
  converted_25 = const_9
  mul_26 = block_load_20 * converted_25
  mul_26 = mul_26.to(tl.float16)
  zero_27 = tl.zeros([128], dtype=tl.float32)
  zero_28 = tl.zeros([128], dtype=tl.float32)
  block_ptr_29 = tl.make_block_ptr(
    base=arg_0 + add_18,
    shape=(4096, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  block_ptr_30 = tl.make_block_ptr(
    base=arg_2 + add_22,
    shape=(4096,),
    strides=(1,),
    offsets=(0,),
    block_shape=(128,),
    order=(0,),
  )
  for i_31 in range(mul_16, const_12, const_13):
    block_load_32 = tl.load(block_ptr_29)
    block_load_33 = tl.load(block_ptr_30)
    where_34 = tl.zeros([128, 128], dtype=tl.float32)
    where_34 = tl.where(i_31 + tl.arange(0, 128)[:, None] >= mul_16 + tl.arange(0, 128)[None, :], where_34, float('-inf'))
    dot_35 = tl.dot(block_load_32, mul_26)
    add_36 = dot_35 + where_34
    exp2_37 = tl.math.exp2(add_36)
    unsqueeze_38 = block_load_33[:, None]
    div_39 = exp2_37 / unsqueeze_38
    square_41 = div_39 * div_39
    reduce_sum_42 = tl.sum(div_39, axis=0, keep_dims=False).to(tl.float32)
    reduce_sum_42 += zero_27
    reduce_sum_43 = tl.sum(square_41, axis=0, keep_dims=False).to(tl.float32)
    reduce_sum_43 += zero_28
    block_advance_44 = tl.advance(block_ptr_29, (128, 0,))
    block_advance_45 = tl.advance(block_ptr_30, (128,))
    block_ptr_29 = block_advance_44
    block_ptr_30 = block_advance_45
    zero_27 = reduce_sum_42
    zero_28 = reduce_sum_43
  block_store_46 = tl.store(block_ptr_23, zero_27)
  block_store_47 = tl.store(block_ptr_24, zero_28)

def RoCo(arg_0, arg_1, arg_2):
  k0_out_0, k0_out_1 = RoCo_p5(arg_0, arg_2, arg_1)
  k1_out_0, k1_out_1 = RoCo_p4(arg_0, arg_1, k0_out_0)
  return k0_out_1, k1_out_0, k1_out_1


def bench_RoCo():
    dev = torch.cuda.current_device()
    rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
    rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
    rand_arg_2 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
    avg_ms = triton.testing.do_bench(lambda: RoCo(rand_arg_0, rand_arg_1, rand_arg_2))
    print('[RoCo] avg_ms:', avg_ms)
    
bench_RoCo()