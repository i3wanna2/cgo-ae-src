// MLA RoPE + FP8 Quant + KV-Cache Fused Kernel
// Source: vllm PR #25774 (rope + static fp8 quant + kvcache fusion)
// This file contains the CUDA kernel, host launcher, and pybind11 binding

#include <torch/all.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Simple LDG macro
#define VLLM_LDG(ptr) (*(ptr))

// Dispatch only half and bfloat16 (raw_kv_scalar_t = uint16_t requires 2-byte type)
#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                   \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                        \
      AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)               \
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__))

namespace vllm {

// ==========================================================================
// FP8 KV-Cache Data Type (from vllm quantization/w8a8/fp8/common.cuh)
// ==========================================================================
enum class Fp8KVCacheDataType {
  kAuto    = 0,  // same as compute dtype (no quantization)
  kFP8E4M3 = 1,
  kFP8E5M2 = 2,
};

// ==========================================================================
// FP8 Scaled Convert Helper (simplified inline version)
// Converts raw_kv_scalar_t (uint16_t backing of __half) to cache_t (uint8_t)
// ==========================================================================
namespace fp8 {

static constexpr float kFP8E4M3Max = 448.0f;
static constexpr float kFP8E5M2Max = 57344.0f;

template <typename cache_t, typename raw_kv_scalar_t, Fp8KVCacheDataType kv_dt>
__device__ __forceinline__ cache_t scaled_convert(raw_kv_scalar_t raw_val,
                                                   float scale) {
  static_assert(sizeof(cache_t) == 1, "cache_t must be 1 byte for fp8");
  static_assert(sizeof(raw_kv_scalar_t) == 2, "raw_kv_t must be 2 bytes (u16)");

  // Reinterpret the raw u16 bits back to __half for float conversion
  const __half h = *reinterpret_cast<const __half*>(&raw_val);
  float fval = __half2float(h);
  float scaled = fval * scale;

  if constexpr (kv_dt == Fp8KVCacheDataType::kFP8E4M3) {
    scaled = fmaxf(-kFP8E4M3Max, fminf(kFP8E4M3Max, scaled));
  } else {
    scaled = fmaxf(-kFP8E5M2Max, fminf(kFP8E5M2Max, scaled));
  }

  // Saturated int8 proxy (compatible with all GPU architectures)
  int ival = __float2int_rn(scaled);
  ival = max(-128, min(127, ival));
  return static_cast<cache_t>(static_cast<int8_t>(ival));
}

}  // namespace fp8

// ==========================================================================
// MLA RoPE + FP8 Quant + KV-Cache Fused Kernel
// Template parameters:
//   qk_t            - compute dtype (half / bfloat16)
//   IS_NEOX         - RoPE style (NeoX vs GPT-J)
//   raw_kv_scalar_t - raw backing type for qk_t (uint16_t)
//   cache_t         - storage dtype in kv_cache (uint8_t for fp8, or same as qk_t)
//   kv_dt           - Fp8KVCacheDataType enum value
// ==========================================================================
template <typename qk_t, bool IS_NEOX, typename raw_kv_scalar_t,
          typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_rope_quant_fused_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    qk_t* __restrict__ q_pe,        // [num_tokens, num_q_heads, rot_dim]
    qk_t* __restrict__ k_pe,        // [num_tokens, rot_dim]
    const qk_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const qk_t* __restrict__ rope_cos_sin_cache,  // [max_position, 2, rot_dim // 2]
    const int rot_dim, const int64_t q_pe_stride_token,
    const int64_t q_pe_stride_head, const int64_t k_pe_stride,
    const int64_t kv_c_stride, const int num_q_heads,
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank + rot_dim)]
    const int64_t* __restrict__ kv_cache_slot_mapping,  // [num_tokens]
    const int block_stride, const int entry_stride, const int kv_lora_rank,
    const int block_size, const float* kv_cache_quant_scale) {
  // Each thread block is responsible for one token.
  const int64_t token_idx = blockIdx.x;
  const int64_t pos = positions[token_idx];

  const qk_t* cos_sin_ptr = rope_cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;

  // Q ROPE (in-place, not quantized - Q stays in compute dtype)
  const int nq = num_q_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    int head_idx = i / embed_dim;
    int pair_idx = i % embed_dim;

    // NOTE: Would be nice to have interleaved sin/cos so we could just load
    // both at the same time.
    qk_t cos = VLLM_LDG(cos_sin_ptr + pair_idx);
    qk_t sin = VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim);

    qk_t* q_pe_head_ptr =
        q_pe + token_idx * q_pe_stride_token + head_idx * q_pe_stride_head;

    int pair_idx_x, pair_idx_y;
    if constexpr (IS_NEOX) {
      // GPT-NeoX style rotary embedding.
      pair_idx_x = pair_idx;
      pair_idx_y = embed_dim + pair_idx;
    } else {
      // GPT-J style rotary embedding.
      pair_idx_x = pair_idx * 2;
      pair_idx_y = pair_idx * 2 + 1;
    }

    qk_t x_src = q_pe_head_ptr[pair_idx_x];
    qk_t y_src = q_pe_head_ptr[pair_idx_y];

    qk_t x_dst = x_src * cos - y_src * sin;
    qk_t y_dst = y_src * cos + x_src * sin;

    q_pe_head_ptr[pair_idx_x] = x_dst;
    q_pe_head_ptr[pair_idx_y] = y_dst;
  }

  const int64_t slot_idx = kv_cache_slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / block_size;
  const int64_t entry_idx = slot_idx % block_size;

  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }

  // K ROPE + write to KV cache with optional FP8 quantization
  for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
    int pair_idx = i;

    qk_t cos = VLLM_LDG(cos_sin_ptr + pair_idx);
    qk_t sin = VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim);

    qk_t* k_pe_head_ptr = k_pe + token_idx * k_pe_stride;

    int pair_idx_x, pair_idx_y;
    if constexpr (IS_NEOX) {
      // GPT-NeoX style rotary embedding.
      pair_idx_x = pair_idx;
      pair_idx_y = embed_dim + pair_idx;
    } else {
      // GPT-J style rotary embedding.
      pair_idx_x = pair_idx * 2;
      pair_idx_y = pair_idx * 2 + 1;
    }

    qk_t x_src = k_pe_head_ptr[pair_idx_x];
    qk_t y_src = k_pe_head_ptr[pair_idx_y];

    qk_t x_dst = x_src * cos - y_src * sin;
    qk_t y_dst = y_src * cos + x_src * sin;

    k_pe_head_ptr[pair_idx_x] = x_dst;
    k_pe_head_ptr[pair_idx_y] = y_dst;

    // NOTE Why is this monster necessary?
    // When K is of type float16, the actual template replacement for
    // raw_kv_scalar_t will be u16. That's why it's used at the last moment
    // otherwise CUDA ALU would break.
    const raw_kv_scalar_t raw_x_value =
        *reinterpret_cast<const raw_kv_scalar_t*>(&x_dst);
    const raw_kv_scalar_t raw_y_value =
        *reinterpret_cast<const raw_kv_scalar_t*>(&y_dst);

    cache_t* kv_cache_ptr = kv_cache + block_idx * block_stride +
                            entry_idx * entry_stride + kv_lora_rank;

    // MLA Cache Store with optional FP8 quantization
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      kv_cache_ptr[pair_idx_x] = raw_x_value;
      kv_cache_ptr[pair_idx_y] = raw_y_value;
    } else {
      kv_cache_ptr[pair_idx_x] =
          fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
              raw_x_value, *kv_cache_quant_scale);
      kv_cache_ptr[pair_idx_y] =
          fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
              raw_y_value, *kv_cache_quant_scale);
    }
  }

  // NOPE (kv_c copy to kv_cache with optional fp8 quant)
  for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    const qk_t* src_ptr = kv_c + token_idx * kv_c_stride + i;
    const raw_kv_scalar_t src_value =
        *reinterpret_cast<const raw_kv_scalar_t*>(src_ptr);

    cache_t* kv_cache_ptr =
        kv_cache + block_idx * block_stride + entry_idx * entry_stride;

    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      kv_cache_ptr[i] = src_value;
    } else {
      kv_cache_ptr[i] = fp8::scaled_convert<cache_t, raw_kv_scalar_t, kv_dt>(
          src_value, *kv_cache_quant_scale);
    }
  }
}

}  // namespace vllm

// ==========================================================================
// Host launcher
// ==========================================================================

void concat_and_cache_mla_rope_quant_fused(
    torch::Tensor& positions,           // [num_tokens]
    torch::Tensor& q_pe,                // [num_tokens, num_q_heads, rot_dim]
    torch::Tensor& k_pe,                // [num_tokens, rot_dim]
    torch::Tensor& kv_c,                // [num_tokens, kv_lora_rank]
    torch::Tensor& rope_cos_sin_cache,  // [max_position, rot_dim]
    bool rope_is_neox,
    torch::Tensor& kv_cache_slot_mapping,  // [num_tokens]
    torch::Tensor& kv_cache,               // [num_blocks, block_size, (kv_lora_rank + rot_dim)]  uint8
    torch::Tensor& kv_cache_quant_scale) { // scalar float32
  const int64_t num_tokens = q_pe.size(0);
  const int num_q_heads = q_pe.size(1);
  const int rot_dim = q_pe.size(2);
  const int kv_lora_rank = kv_c.size(1);

  TORCH_CHECK_EQ(q_pe.dim(), 3);
  TORCH_CHECK_EQ(k_pe.dim(), 2);
  TORCH_CHECK_EQ(kv_c.dim(), 2);
  TORCH_CHECK_EQ(kv_cache.dim(), 3);
  TORCH_CHECK_EQ(kv_cache.size(2), kv_lora_rank + rot_dim);
  TORCH_CHECK_EQ(rope_cos_sin_cache.size(1), rot_dim);
  TORCH_CHECK_EQ(kv_cache_slot_mapping.scalar_type(), c10::ScalarType::Long);
  TORCH_CHECK_EQ(kv_cache.scalar_type(), c10::ScalarType::Byte);

  int64_t q_pe_stride_token = q_pe.stride(0);
  int64_t q_pe_stride_head  = q_pe.stride(1);
  int64_t k_pe_stride        = k_pe.stride(0);
  int64_t kv_c_stride        = kv_c.stride(0);

  int block_size   = kv_cache.size(1);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  int rope_block_size   = std::min(num_q_heads * rot_dim / 2, 512);
  int mla_block_size    = kv_lora_rank;
  int thread_block_size = std::min(std::max(rope_block_size, mla_block_size), 512);

  dim3 grid(num_tokens, 1, 1);
  dim3 block(thread_block_size, 1, 1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(positions));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using raw_kv_scalar_t = uint16_t;
  using cache_t         = uint8_t;
  constexpr auto kv_dt  = vllm::Fp8KVCacheDataType::kFP8E4M3;

  VLLM_DISPATCH_FLOATING_TYPES(q_pe.scalar_type(), "qk_scalar_type", [&] {
    using qk_t = scalar_t;
    if (rope_is_neox) {
      vllm::concat_and_cache_mla_rope_quant_fused_kernel<qk_t, true, raw_kv_scalar_t, cache_t, kv_dt>
          <<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(),
              q_pe.data_ptr<qk_t>(),
              k_pe.data_ptr<qk_t>(),
              kv_c.data_ptr<qk_t>(),
              rope_cos_sin_cache.data_ptr<qk_t>(),
              rot_dim, q_pe_stride_token, q_pe_stride_head,
              k_pe_stride, kv_c_stride, num_q_heads,
              kv_cache.data_ptr<cache_t>(),
              kv_cache_slot_mapping.data_ptr<int64_t>(),
              block_stride, entry_stride, kv_lora_rank, block_size,
              kv_cache_quant_scale.data_ptr<float>());
    } else {
      vllm::concat_and_cache_mla_rope_quant_fused_kernel<qk_t, false, raw_kv_scalar_t, cache_t, kv_dt>
          <<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(),
              q_pe.data_ptr<qk_t>(),
              k_pe.data_ptr<qk_t>(),
              kv_c.data_ptr<qk_t>(),
              rope_cos_sin_cache.data_ptr<qk_t>(),
              rot_dim, q_pe_stride_token, q_pe_stride_head,
              k_pe_stride, kv_c_stride, num_q_heads,
              kv_cache.data_ptr<cache_t>(),
              kv_cache_slot_mapping.data_ptr<int64_t>(),
              block_stride, entry_stride, kv_lora_rank, block_size,
              kv_cache_quant_scale.data_ptr<float>());
    }
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("concat_and_cache_mla_rope_quant_fused",
        &concat_and_cache_mla_rope_quant_fused,
        "MLA RoPE + FP8 Quant + KV-Cache fused kernel (CUDA)");
}
