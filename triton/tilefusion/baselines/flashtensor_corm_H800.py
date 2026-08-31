import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

def bench_Corm_p1():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  avg_ms = triton.testing.do_bench(lambda: Corm_p1(rand_arg_0, rand_arg_1, rand_arg_2))
  print('[Corm_p1] avg_ms:', avg_ms)

def Corm_p1(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  empty_ptr_3 = torch.empty(1, 32, 4096, 1, dtype=torch.float32, device=dev)
  empty_ptr_4 = torch.empty(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  grid = (1, 32, 32)
  Corm_p1_kernel[grid](tensor_0, tensor_1, tensor_2, empty_ptr_3, empty_ptr_4, autotune_key)
  tensor_5 = empty_ptr_3
  tensor_6 = empty_ptr_4
  return tensor_5, tensor_6

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def Corm_p1_kernel(
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

def bench_Corm_p2():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 32, 4096, 1, dtype=torch.float32, device=dev)
  rand_arg_3 = torch.randn(4096, 4096, dtype=torch.float32, device=dev)
  avg_ms = triton.testing.do_bench(lambda: Corm_p2(rand_arg_0, rand_arg_1, rand_arg_2, rand_arg_3))
  print('[Corm_p2] avg_ms:', avg_ms)

def Corm_p2(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: torch.Tensor) -> torch.Tensor:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  tensor_3 = arg3
  empty_ptr_4 = torch.empty(1, 32, 4096, dtype=torch.bool, device=dev)
  grid = (1, 32, 32)
  Corm_p2_kernel[grid](tensor_0, tensor_1, tensor_2, tensor_3, empty_ptr_4, autotune_key)
  tensor_5 = empty_ptr_4
  return tensor_5

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def Corm_p2_kernel(
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
  const_11 = 4096
  const_12 = 128
  const_13 = 1
  mul_14 = pid_7 * const_12
  mul_15 = pid_6 * const_12
  mul_16 = mul_15 * const_11
  add_17 = mul_16 + mul_14
  block_ptr_18 = tl.make_block_ptr(
    base=arg_1 + add_17,
    shape=(128, 128,),
    strides=(1, 4096,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(0, 1,),
  )
  block_load_19 = tl.load(block_ptr_18)
  mul_20 = pid_7 * const_11
  add_21 = mul_15 + mul_20
  block_ptr_22 = tl.make_block_ptr(
    base=arg_4 + add_21,
    shape=(128,),
    strides=(1,),
    offsets=(0,),
    block_shape=(128,),
    order=(0,),
  )
  converted_23 = const_8
  mul_24 = block_load_19 * converted_23
  mul_24 = mul_24.to(tl.float16)
  zero_25 = tl.zeros([128], dtype=tl.int8)
  block_ptr_26 = tl.make_block_ptr(
    base=arg_0 + add_17,
    shape=(4096, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  block_ptr_27 = tl.make_block_ptr(
    base=arg_2 + add_21,
    shape=(4096,),
    strides=(1,),
    offsets=(0,),
    block_shape=(128,),
    order=(0,),
  )
  add_28 = mul_16 + mul_15
  block_ptr_29 = tl.make_block_ptr(
    base=arg_3 + add_28,
    shape=(4096, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  for i_30 in range(mul_15, const_11, const_12):
    block_load_31 = tl.load(block_ptr_26)
    block_load_32 = tl.load(block_ptr_27)
    block_load_33 = tl.load(block_ptr_29)
    where_34 = tl.zeros([128, 128], dtype=tl.float32)
    where_34 = tl.where(i_30 + tl.arange(0, 128)[:, None] >= mul_15 + tl.arange(0, 128)[None, :], where_34, float('-inf'))
    dot_35 = tl.dot(block_load_31, mul_24)
    add_36 = dot_35 + where_34
    exp2_37 = tl.math.exp2(add_36)
    unsqueeze_38 = block_load_32[:, None]
    div_39 = exp2_37 / unsqueeze_38
    ge_40 = div_39 >= block_load_33
    reduce_any_41 = tl.max(ge_40, axis=0, keep_dims=False).to(tl.int8)
    reduce_any_41 |= zero_25
    block_advance_42 = tl.advance(block_ptr_26, (128, 0,))
    block_advance_43 = tl.advance(block_ptr_27, (128,))
    block_advance_44 = tl.advance(block_ptr_29, (128, 0,))
    block_ptr_26 = block_advance_42
    block_ptr_27 = block_advance_43
    block_ptr_29 = block_advance_44
    zero_25 = reduce_any_41
  block_store_45 = tl.store(block_ptr_22, zero_25)

def Corm(arg_0, arg_1, arg_2, arg_3):
  k0_out_0, k0_out_1 = Corm_p1(arg_0, arg_2, arg_1)
  k1_out_0 = Corm_p2(arg_0, arg_1, k0_out_0, arg_3)
  return k0_out_1, k1_out_0


def bench_Corm():
    dev = torch.cuda.current_device()
    rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
    rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
    rand_arg_2 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
    rand_arg_3 = torch.randn(4096, 4096, dtype=torch.float32, device=dev)
    avg_ms = triton.testing.do_bench(lambda: Corm(rand_arg_0, rand_arg_1, rand_arg_2, rand_arg_3))
    print('[Corm] avg_ms:', avg_ms)
    
bench_Corm()