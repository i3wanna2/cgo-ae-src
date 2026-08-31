#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Simple LDG macro
#define VLLM_LDG(ptr) (*(ptr))

// Dispatch floating types macro
#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                   \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                      \
      at::ScalarType::Half, at::ScalarType::BFloat16, TYPE, NAME, __VA_ARGS__)

namespace vllm {

template <typename qk_t, bool IS_NEOX>
__global__ void concat_and_cache_mla_rope_fused_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    qk_t* __restrict__ q_pe,        // [num_tokens, num_q_heads, rot_dim]
    qk_t* __restrict__ k_pe,        // [num_tokens, rot_dim]
    const qk_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const qk_t* __restrict__ rope_cos_sin_cache,  // [max_position, rot_dim]
    const int rot_dim, const int64_t q_pe_stride_token,
    const int64_t q_pe_stride_head, const int64_t k_pe_stride,
    const int64_t kv_c_stride, const int num_q_heads,
    qk_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank + rot_dim)]
    const int64_t* __restrict__ kv_cache_slot_mapping,  // [num_tokens]
    const int block_stride, const int entry_stride, const int kv_lora_rank,
    const int block_size) {
  // Each thread block is responsible for one token.
  const int64_t token_idx = blockIdx.x;
  const int64_t pos = positions[token_idx];

  const qk_t* cos_sin_ptr = rope_cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;

  // Q ROPE
  const int nq = num_q_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    int head_idx = i / embed_dim;
    int pair_idx = i % embed_dim;

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

  // K with 1 HEAD
  for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
    int pair_idx = i;

    qk_t cos = VLLM_LDG(cos_sin_ptr + pair_idx);
    qk_t sin = VLLM_LDG(cos_sin_ptr + pair_idx + embed_dim);

    qk_t* k_pe_head_ptr = k_pe + token_idx * k_pe_stride;

    int pair_idx_x, pair_idx_y;
    if constexpr (IS_NEOX) {
      pair_idx_x = pair_idx;
      pair_idx_y = embed_dim + pair_idx;
    } else {
      pair_idx_x = pair_idx * 2;
      pair_idx_y = pair_idx * 2 + 1;
    }

    qk_t x_src = k_pe_head_ptr[pair_idx_x];
    qk_t y_src = k_pe_head_ptr[pair_idx_y];

    qk_t x_dst = x_src * cos - y_src * sin;
    qk_t y_dst = y_src * cos + x_src * sin;

    k_pe_head_ptr[pair_idx_x] = x_dst;
    k_pe_head_ptr[pair_idx_y] = y_dst;

    // Write to KV cache
    qk_t* kv_cache_ptr = kv_cache + block_idx * block_stride +
                         entry_idx * entry_stride + kv_lora_rank;

    kv_cache_ptr[pair_idx_x] = x_dst;
    kv_cache_ptr[pair_idx_y] = y_dst;
  }

  // NOPE (kv_c copy)
  for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
    const qk_t* src_ptr = kv_c + token_idx * kv_c_stride + i;
    qk_t src_value = *src_ptr;

    qk_t* kv_cache_ptr =
        kv_cache + block_idx * block_stride + entry_idx * entry_stride;

    kv_cache_ptr[i] = src_value;
  }
}

}  // namespace vllm

// Executes RoPE on q_pe and k_pe, then writes k_pe and kv_c in the kv cache.
// q_pe and k_pe are modified in place.
void concat_and_cache_mla_rope_fused(
    torch::Tensor& positions,           // [num_tokens]
    torch::Tensor& q_pe,                // [num_tokens, num_q_heads, rot_dim]
    torch::Tensor& k_pe,                // [num_tokens, rot_dim]
    torch::Tensor& kv_c,                // [num_tokens, kv_lora_rank]
    torch::Tensor& rope_cos_sin_cache,  // [max_position, rot_dim]
    bool rope_is_neox,
    torch::Tensor& kv_cache_slot_mapping,  // [num_tokens]
    torch::Tensor& kv_cache) {             // [num_blocks, block_size, (kv_lora_rank + rot_dim)]
    
  const int64_t num_tokens = q_pe.size(0);
  const int num_q_heads = q_pe.size(1);
  const int rot_dim = q_pe.size(2);
  const int kv_lora_rank = kv_c.size(1);

  TORCH_CHECK(positions.size(0) >= num_tokens);
  TORCH_CHECK_EQ(positions.dim(), 1);
  TORCH_CHECK_EQ(positions.scalar_type(), c10::ScalarType::Long);

  TORCH_CHECK_EQ(q_pe.size(0), num_tokens);
  TORCH_CHECK_EQ(q_pe.size(1), num_q_heads);
  TORCH_CHECK_EQ(q_pe.size(2), rot_dim);
  TORCH_CHECK_EQ(q_pe.dim(), 3);

  TORCH_CHECK_EQ(k_pe.size(0), num_tokens);
  TORCH_CHECK_EQ(k_pe.size(1), rot_dim);
  TORCH_CHECK_EQ(k_pe.dim(), 2);

  TORCH_CHECK_EQ(kv_c.size(0), num_tokens);
  TORCH_CHECK_EQ(kv_c.size(1), kv_lora_rank);
  TORCH_CHECK_EQ(kv_c.dim(), 2);

  TORCH_CHECK_EQ(rope_cos_sin_cache.size(1), rot_dim);

  TORCH_CHECK_EQ(kv_cache_slot_mapping.size(0), num_tokens);
  TORCH_CHECK_EQ(kv_cache_slot_mapping.scalar_type(), c10::ScalarType::Long);

  TORCH_CHECK_EQ(kv_cache.size(2), kv_lora_rank + rot_dim);
  TORCH_CHECK_EQ(kv_cache.dim(), 3);

  int64_t q_pe_stride_token = q_pe.stride(0);
  int64_t q_pe_stride_head = q_pe.stride(1);
  int64_t k_pe_stride = k_pe.stride(0);
  int64_t kv_c_stride = kv_c.stride(0);

  int block_size = kv_cache.size(1);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  int rope_block_size = std::min(num_q_heads * rot_dim / 2, 512);
  int mla_block_size = kv_lora_rank;
  int thread_block_size = std::min(std::max(rope_block_size, mla_block_size), 512);

  dim3 grid(num_tokens, 1, 1);
  dim3 block(thread_block_size, 1, 1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(positions));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_FLOATING_TYPES(q_pe.scalar_type(), "qk_scalar_type", [&] {
    using qk_t = scalar_t;
    if (rope_is_neox) {
      vllm::concat_and_cache_mla_rope_fused_kernel<qk_t, true>
          <<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(), q_pe.data_ptr<qk_t>(),
              k_pe.data_ptr<qk_t>(), kv_c.data_ptr<qk_t>(),
              rope_cos_sin_cache.data_ptr<qk_t>(), rot_dim,
              q_pe_stride_token, q_pe_stride_head, k_pe_stride, kv_c_stride,
              num_q_heads, kv_cache.data_ptr<qk_t>(),
              kv_cache_slot_mapping.data_ptr<int64_t>(), block_stride,
              entry_stride, kv_lora_rank, block_size);
    } else {
      vllm::concat_and_cache_mla_rope_fused_kernel<qk_t, false>
          <<<grid, block, 0, stream>>>(
              positions.data_ptr<int64_t>(), q_pe.data_ptr<qk_t>(),
              k_pe.data_ptr<qk_t>(), kv_c.data_ptr<qk_t>(),
              rope_cos_sin_cache.data_ptr<qk_t>(), rot_dim,
              q_pe_stride_token, q_pe_stride_head, k_pe_stride, kv_c_stride,
              num_q_heads, kv_cache.data_ptr<qk_t>(),
              kv_cache_slot_mapping.data_ptr<int64_t>(), block_stride,
              entry_stride, kv_lora_rank, block_size);
    }
  });
}

#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("concat_and_cache_mla_rope_fused", &concat_and_cache_mla_rope_fused,
        "MLA RoPE fused kernel (CUDA)");
}
