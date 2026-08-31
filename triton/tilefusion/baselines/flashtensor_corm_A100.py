import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Callable, Any, Optional, Tuple

def bench_Corm_p3():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_2 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  avg_ms = triton.testing.do_bench(lambda: Corm_p3(rand_arg_0, rand_arg_1, rand_arg_2))
  print('[Corm_p3] avg_ms:', avg_ms)

def Corm_p3(arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor) -> torch.Tensor:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  tensor_2 = arg2
  empty_ptr_3 = torch.empty(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  grid = (1, 32, 32)
  Corm_p3_kernel[grid](tensor_0, tensor_1, tensor_2, empty_ptr_3, autotune_key)
  tensor_4 = empty_ptr_3
  return tensor_4

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def Corm_p3_kernel(
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
  const_10 = 0
  const_11 = 1
  const_12 = 4096
  const_13 = 128
  mul_14 = pid_5 * const_13
  mul_15 = mul_14 * const_12
  mul_16 = pid_6 * const_13
  add_17 = mul_15 + mul_16
  block_ptr_18 = tl.make_block_ptr(
    base=arg_0 + add_17,
    shape=(128, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  block_load_19 = tl.load(block_ptr_18)
  block_ptr_20 = tl.make_block_ptr(
    base=arg_3 + add_17,
    shape=(128, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  converted_21 = const_7
  mul_22 = block_load_19 * converted_21
  mul_22 = mul_22.to(tl.float16)
  zero_23 = tl.zeros([128, 128], dtype=tl.float32)
  zero_24 = tl.zeros([128, 1], dtype=tl.float32)
  add_25 = mul_14 + const_13
  block_ptr_26 = tl.make_block_ptr(
    base=arg_2 + mul_16,
    shape=(128, 4096,),
    strides=(1, 4096,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(0, 1,),
  )
  block_ptr_27 = tl.make_block_ptr(
    base=arg_1 + mul_16,
    shape=(4096, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  for i_28 in range(const_10, add_25, const_13):
    block_load_29 = tl.load(block_ptr_26)
    block_load_30 = tl.load(block_ptr_27)
    dot_31 = tl.dot(mul_22, block_load_29)
    where_32 = tl.zeros([128, 128], dtype=tl.float32)
    where_32 = tl.where(mul_14 + tl.arange(0, 128)[:, None] >= i_28 + tl.arange(0, 128)[None, :], where_32, float('-inf'))
    add_33 = dot_31 + where_32
    exp2_34 = tl.math.exp2(add_33)
    reduce_sum_35 = tl.sum(exp2_34, axis=1, keep_dims=True).to(tl.float32)
    reduce_sum_35 += zero_24
    converted_36 = exp2_34.to(tl.float16)
    dot_37 = tl.dot(converted_36, block_load_30)
    add_38 = zero_23 + dot_37
    block_advance_39 = tl.advance(block_ptr_26, (0, 128,))
    block_advance_40 = tl.advance(block_ptr_27, (128, 0,))
    block_ptr_26 = block_advance_39
    block_ptr_27 = block_advance_40
    zero_23 = add_38
    zero_24 = reduce_sum_35
  div_41 = zero_23 / zero_24
  converted_42 = div_41.to(tl.float16)
  block_store_43 = tl.store(block_ptr_20, converted_42)

def bench_Corm_p4():
  dev = torch.cuda.current_device()
  rand_arg_0 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  rand_arg_1 = torch.randn(1, 4096, 32, 128, dtype=torch.float16, device=dev)
  avg_ms = triton.testing.do_bench(lambda: Corm_p4(rand_arg_0, rand_arg_1))
  print('[Corm_p4] avg_ms:', avg_ms)

def Corm_p4(arg0: torch.Tensor, arg1: torch.Tensor) -> torch.Tensor:
  dev = arg0.device
  autotune_key = torch.cuda.get_device_capability(dev)[0]
  tensor_0 = arg0
  tensor_1 = arg1
  empty_ptr_2 = torch.empty(1, 32, 4096, 1, dtype=torch.float32, device=dev)
  grid = (1, 32, 32)
  Corm_p4_kernel[grid](tensor_0, tensor_1, empty_ptr_2, autotune_key)
  tensor_3 = empty_ptr_2
  return tensor_3

@triton.autotune(configs=[
  triton.Config({}, num_warps=4),
  triton.Config({}, num_warps=8),
], key=['autotune_key'])
@triton.jit
def Corm_p4_kernel(
  arg_0,
  arg_1,
  arg_2,
  autotune_key,
):
  pid_3 = tl.program_id(0)
  pid_4 = tl.program_id(1)
  pid_5 = tl.program_id(2)
  const_6 = 1.275311e-01
  const_7 = float('-inf')
  const_8 = 0.000000e+00
  const_9 = 0
  const_10 = 1
  const_11 = 4096
  const_12 = 128
  mul_13 = pid_4 * const_12
  mul_14 = mul_13 * const_11
  mul_15 = pid_5 * const_12
  add_16 = mul_14 + mul_15
  block_ptr_17 = tl.make_block_ptr(
    base=arg_0 + add_16,
    shape=(128, 128,),
    strides=(4096, 1,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(1, 0,),
  )
  block_load_18 = tl.load(block_ptr_17)
  mul_19 = pid_5 * const_11
  add_20 = mul_13 + mul_19
  block_ptr_21 = tl.make_block_ptr(
    base=arg_2 + add_20,
    shape=(128, 1,),
    strides=(1, 1,),
    offsets=(0, 0,),
    block_shape=(128, 1,),
    order=(1, 0,),
  )
  converted_22 = const_6
  mul_23 = block_load_18 * converted_22
  mul_23 = mul_23.to(tl.float16)
  zero_24 = tl.zeros([128, 1], dtype=tl.float32)
  add_25 = mul_13 + const_12
  block_ptr_26 = tl.make_block_ptr(
    base=arg_1 + mul_15,
    shape=(128, 4096,),
    strides=(1, 4096,),
    offsets=(0, 0,),
    block_shape=(128, 128,),
    order=(0, 1,),
  )
  for i_27 in range(const_9, add_25, const_12):
    block_load_28 = tl.load(block_ptr_26)
    where_29 = tl.zeros([128, 128], dtype=tl.float32)
    where_29 = tl.where(mul_13 + tl.arange(0, 128)[:, None] >= i_27 + tl.arange(0, 128)[None, :], where_29, float('-inf'))
    dot_30 = tl.dot(mul_23, block_load_28)
    add_31 = dot_30 + where_29
    exp2_32 = tl.math.exp2(add_31)
    reduce_sum_33 = tl.sum(exp2_32, axis=1, keep_dims=True).to(tl.float32)
    reduce_sum_33 += zero_24
    block_advance_34 = tl.advance(block_ptr_26, (0, 128,))
    block_ptr_26 = block_advance_34
    zero_24 = reduce_sum_33
  block_store_35 = tl.store(block_ptr_21, zero_24)

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

k1 :
    gemm -> scalemask -> softmax_store -> gemm ouput(softmax_store_input, attn_output)

k2 : 
    softmax_store_input -> gemm -> scaleMask -> exp / softmax_store_input
    
    
k1 :
    gemm -> scalemask -> softmax -> gemm ouput(, attn_output)

k2 :
    gemm -> scalemask -> softmax_reduce ->out 
    
k3 : 
    out_input -> gemm -> scaleMask -> exp / softmax_store_input   


def Corm(arg_0, arg_1, arg_2, arg_3):
  k0_out_0 = Corm_p3(arg_0, arg_2, arg_1)
  k1_out_0 = Corm_p4(arg_0, arg_1)
  k2_out_0 = Corm_p2(arg_0, arg_1, k1_out_0, arg_3)
  return k0_out_0, k2_out_0
