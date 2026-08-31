import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

def bench_KeyFormer_p8():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  avg_ms = triton.testing.do_bench(lambda: KeyFormer_p8(rand_arg_0, rand_arg_1, rand_arg_2))
  print('[KeyFormer_p8] avg_ms:', avg_ms)

def KeyFormer_p8(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  empty_ptr_3 = torch.empty(1, 32, 4096, 1, dtype=torch.float32, device=dev)
  empty_ptr_4 = torch.empty(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  grid = (1, 32, 32)
  KeyFormer_p8_kernel[grid](tensor_0, tensor_1, tensor_2, empty_ptr_3, empty_ptr_4, autotune_key)
  tensor_5 = empty_ptr_3
  tensor_6 = empty_ptr_4
  return tensor_5, tensor_6

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def KeyFormer_p8_kernel(
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

def bench_KeyFormer_p9():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 32, 4096, 4096, dtype=torch.float32, device=dev)
  avg_ms = triton.testing.do_bench(lambda: KeyFormer_p9(rand_arg_0, rand_arg_1, rand_arg_2))
  print('[KeyFormer_p9] avg_ms:', avg_ms)

def KeyFormer_p9(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> torch.Tensor:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  empty_ptr_3 = torch.empty(1, 32, 4096, 1, dtype=torch.float32, device=dev)
  grid = (1, 32, 32)
  KeyFormer_p9_kernel[grid](tensor_0, tensor_1, tensor_2, empty_ptr_3, autotune_key)
  tensor_4 = empty_ptr_3
  return tensor_4

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def KeyFormer_p9_kernel(
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
  const_8 = 9.617967e-01
  const_9 = float('-inf')
  const_10 = 0.000000e+00
  const_11 = 0
  const_12 = 16777216
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
  mul_22 = pid_6 * const_12
  add_23 = mul_17 + mul_22
  mul_24 = pid_6 * const_14
  add_25 = mul_16 + mul_24
  block_ptr_26 = tl.make_block_ptr(
    base=arg_3 + add_25,
    shape=(128, 1,),
    strides=(1, 1,),
    offsets=(0, 0,),
    block_shape=(128, 1,),
    order=(1, 0,),
  )
  converted_27 = const_7
  div_28 = block_load_21 / converted_27
  div_28 = div_28.to(tl.float16)
  zero_29 = tl.zeros([128, 1], dtype=tl.float32)
  add_30 = mul_16 + const_15
  block_ptr_31 = tl.make_block_ptr(
    base=arg_1 + mul_18,
    shape=(128, 4096,),
    strides=(1, 4096,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(0, 1,),
  )
  block_ptr_32 = tl.make_block_ptr(
    base=arg_2 + add_23,
    shape=(128, 4096,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  for i_33 in range(const_11, add_30, const_15):
    block_load_34 = tl.load(block_ptr_31)
    block_load_35 = tl.load(block_ptr_32)
    log_36 = tl.math.log(block_load_35)
    neg_37 = -(log_36)
    mul_38 = neg_37 * const_8
    where_39 = tl.zeros([128, 128], dtype=tl.float32)
    where_39 = tl.where(mul_16 + tl.arange(0, 128)[:, None] >= i_33 + tl.arange(0, 128)[None, :], where_39, float('-inf'))
    converted_40 = const_8
    mul_41 = block_load_34 * converted_40
    mul_41 = mul_41.to(tl.float16)
    dot_42 = tl.dot(div_28, mul_41)
    add_43 = dot_42 + where_39
    add_44 = add_43 + mul_38
    exp2_45 = tl.math.exp2(add_44)
    reduce_sum_46 = tl.sum(exp2_45, axis=1, keep_dims=True).to(tl.float32)
    reduce_sum_46 += zero_29
    block_advance_47 = tl.advance(block_ptr_31, (0, 128,))
    block_advance_48 = tl.advance(block_ptr_32, (0, 128,))
    block_ptr_31 = block_advance_47
    block_ptr_32 = block_advance_48
    zero_29 = reduce_sum_46
  block_store_49 = tl.store(block_ptr_26, zero_29)

def bench_KeyFormer_p7():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 32, 4096, 4096, dtype=torch.float32, device=dev)
  rand_arg_3 = torch.randn(1, 32, 4096, 1, dtype=torch.float32, device=dev)
  avg_ms = triton.testing.do_bench(lambda: KeyFormer_p7(rand_arg_0, rand_arg_1, rand_arg_2, rand_arg_3))
  print('[KeyFormer_p7] avg_ms:', avg_ms)

def KeyFormer_p7(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor) -> torch.Tensor:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  tensor_3 = arg3
  empty_ptr_4 = torch.empty(1, 32, 4096, dtype=torch.float32, device=dev)
  grid = (1, 32, 32)
  KeyFormer_p7_kernel[grid](tensor_0, tensor_1, tensor_2, tensor_3, empty_ptr_4, autotune_key)
  tensor_5 = empty_ptr_4
  return tensor_5

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def KeyFormer_p7_kernel(
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
  const_8 = 1.131250e+01
  const_9 = 9.617967e-01
  const_10 = float('-inf')
  const_11 = 0.000000e+00
  const_12 = 16777216
  const_13 = 4096
  const_14 = 128
  const_15 = 1
  mul_16 = pid_6 * const_14
  mul_17 = pid_7 * const_14
  mul_18 = mul_17 * const_13
  add_19 = mul_16 + mul_18
  block_ptr_20 = tl.make_block_ptr(
    base=arg_1 + add_19,
    shape=(128, 128,),
    strides=(1, 4096,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(0, 1,),
  )
  block_load_21 = tl.load(block_ptr_20)
  mul_22 = pid_6 * const_12
  add_23 = mul_22 + mul_17
  mul_24 = pid_6 * const_13
  add_25 = mul_24 + mul_17
  block_ptr_26 = tl.make_block_ptr(
    base=arg_4 + add_25,
    shape=(128,),
    strides=(1,),
    offsets=(0,),
    block_shape=(128,),
    order=(0,),
  )
  converted_27 = const_9
  mul_28 = block_load_21 * converted_27
  mul_28 = mul_28.to(tl.float16)
  zero_29 = tl.zeros([128], dtype=tl.float32)
  block_ptr_30 = tl.make_block_ptr(
    base=arg_0 + add_19,
    shape=(4096, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  add_31 = mul_18 + add_23
  block_ptr_32 = tl.make_block_ptr(
    base=arg_2 + add_31,
    shape=(4096, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  block_ptr_33 = tl.make_block_ptr(
    base=arg_3 + add_25,
    shape=(4096,),
    strides=(1,),
    offsets=(0,),
    block_shape=(128,),
    order=(0,),
  )
  for i_34 in range(mul_17, const_13, const_14):
    block_load_35 = tl.load(block_ptr_30)
    block_load_36 = tl.load(block_ptr_32)
    block_load_37 = tl.load(block_ptr_33)
    log_38 = tl.math.log(block_load_36)
    neg_39 = -(log_38)
    mul_40 = neg_39 * const_9
    where_41 = tl.zeros([128, 128], dtype=tl.float32)
    where_41 = tl.where(i_34 + tl.arange(0, 128)[:, None] >= mul_17 + tl.arange(0, 128)[None, :], where_41, float('-inf'))
    converted_42 = const_8
    div_43 = block_load_35 / converted_42
    div_43 = div_43.to(tl.float16)
    dot_44 = tl.dot(div_43, mul_28)
    add_45 = dot_44 + where_41
    add_46 = add_45 + mul_40
    exp2_47 = tl.math.exp2(add_46)
    unsqueeze_48 = block_load_37[:, None]
    div_49 = exp2_47 / unsqueeze_48
    reduce_sum_50 = tl.sum(div_49, axis=0, keep_dims=False).to(tl.float32)
    reduce_sum_50 += zero_29
    block_advance_51 = tl.advance(block_ptr_30, (128, 0,))
    block_advance_52 = tl.advance(block_ptr_32, (128, 0,))
    block_advance_53 = tl.advance(block_ptr_33, (128,))
    block_ptr_30 = block_advance_51
    block_ptr_32 = block_advance_52
    block_ptr_33 = block_advance_53
    zero_29 = reduce_sum_50
  block_store_54 = tl.store(block_ptr_26, zero_29)

def KeyFormer(arg_0, arg_1, arg_2, arg_3):
  k0_out_0, k0_out_1 = KeyFormer_p8(arg_0, arg_2, arg_1)
  k1_out_0 = KeyFormer_p9(arg_0, arg_1, arg_3)
  k2_out_0 = KeyFormer_p7(arg_0, arg_1, arg_3, k1_out_0)
  return k0_out_1, k2_out_0

def bench_KeyFormer():
    dev = torch.cuda.current_device()
    rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
    rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
    rand_arg_2 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
    rand_arg_3 = torch.randn(1, 32, 4096, 4096, dtype=torch.float32, device=dev)
    avg_ms = triton.testing.do_bench(lambda: KeyFormer(rand_arg_0, rand_arg_1, rand_arg_2, rand_arg_3))
    print('[KeyFormer] avg_ms:', avg_ms)
    
bench_KeyFormer()