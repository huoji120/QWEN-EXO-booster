/*
 * Copyright (c) 2025 by SGLang team.
 * Copyright (c) 2024-2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <sgl_kernel/tensor.h>   // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>    // For CHECK_HOST
#include <sgl_kernel/utils.cuh>  // For LaunchKernel and CHECK_CUDA

#include <flashinfer/sampling.cuh>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <limits>

namespace {

// Keep the dense target-only kernel's block, vector, scan and reduction layout.
// Only its initially-zero draft-probability tensor is replaced: a topk=1 chain
// can reject at most one candidate at the final node, and none at a bonus node.
// The FlashInfer vector/sampling primitives below deliberately match
// sgl-kernel/csrc/speculative/speculative_sampling.cuh, including FP32 order.
template <uint32_t VEC_SIZE>
__global__ void chain_target_only_kernel(
    int32_t* predicts,
    int32_t* accept_index,
    int32_t* accept_token_num,
    const int64_t* candidates,
    const int64_t* retrieve_index,
    const int64_t* retrieve_next_token,
    const float* uniform_samples,
    const float* uniform_samples_for_final_sampling,
    const float* target_probs,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens,
    uint32_t d,
    float threshold_single,
    float threshold_acc) {
  using namespace flashinfer;
  using namespace flashinfer::sampling;
  using namespace cub;
  constexpr uint32_t BLOCK_THREADS = 1024;
  extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>&>(smem_sampling);
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  float prob_acc = 0.0;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  float coin = uniform_samples[bx * num_draft_tokens];
  int64_t last_accepted_retrieve_idx = retrieve_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrieve_idx;
  uint32_t num_accepted_tokens = 0;
  int64_t cur_index = 0;
  int64_t rejected_token = -1;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrieve_next_token[bx * num_draft_tokens + cur_index];
    if (cur_index == -1) break;
    const int64_t draft_index = retrieve_index[bx * num_draft_tokens + cur_index];
    const int64_t draft_token_id = candidates[bx * num_draft_tokens + cur_index];
    const float target_prob_single = target_probs[cur_prob_offset + draft_token_id];
    prob_acc += target_prob_single;
    if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single) {
      prob_acc = 0.;
      cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
      coin = uniform_samples[bx * num_draft_tokens + cur_index];
      predicts[last_accepted_retrieve_idx] = draft_token_id;
      ++num_accepted_tokens;
      accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
      last_accepted_retrieve_idx = draft_index;
    } else {
      // The generic kernel writes q[token] to the otherwise-zero p row.
      // There are no siblings on a topk=1 chain, so this is the final node.
      rejected_token = draft_token_id;
      break;
    }
  }
  accept_token_num[bx] = num_accepted_tokens;
  coin = uniform_samples_for_final_sampling[bx];

  float sum_relu_q_minus_p(0);
  vec_t<float, VEC_SIZE> q_vec, p_vec;
  float relu_q_minus_p[VEC_SIZE];
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(float(0));
    p_vec.fill(float(0));
    const uint32_t offset = (i * BLOCK_THREADS + tx) * VEC_SIZE;
    if (offset < d) {
      q_vec.load(target_probs + cur_prob_offset + offset);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          if (offset + j == rejected_token) p_vec[j] = q_vec[j];
        }
      }
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], float(0));
    }
    sum_relu_q_minus_p += BlockReduce<float, BLOCK_THREADS, REDUCE_ALGO>(temp_storage.block_prim.reduce)
                              .Sum<VEC_SIZE>(relu_q_minus_p);
    __syncthreads();
  }
  if (tx == 0) temp_storage.block_aggregate.value = sum_relu_q_minus_p;
  temp_storage.sampled_id = d;
  temp_storage.last_valid_id = -1;
  __syncthreads();
  sum_relu_q_minus_p = temp_storage.block_aggregate.value;
  float u = coin * sum_relu_q_minus_p;
  float aggregate_relu_q_minus_p(0);
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    q_vec.fill(float(0));
    p_vec.fill(float(0));
    const uint32_t offset = (i * BLOCK_THREADS + tx) * VEC_SIZE;
    if (offset < d) {
      q_vec.load(target_probs + cur_prob_offset + offset);
      if (num_accepted_tokens != num_speculative_tokens - 1) {
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          if (offset + j == rejected_token) p_vec[j] = q_vec[j];
        }
      }
    }
    vec_t<float, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      relu_q_minus_p_vec[j] = max(q_vec[j] - p_vec[j], float(0));
    }
    DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, true>(
        i, d, [&](float x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p, &temp_storage);
    if (aggregate_relu_q_minus_p > u) break;
  }
  __syncthreads();
  int sampled_id = temp_storage.sampled_id;
  if (sampled_id == d) {
    sampled_id = temp_storage.last_valid_id == -1 ? d - 1 : temp_storage.last_valid_id;
  }
  predicts[last_accepted_retrieve_idx] = sampled_id;
}

template <uint32_t VEC_SIZE>
void chain_target_only(
    tvm::ffi::TensorView predicts,
    tvm::ffi::TensorView accept_index,
    tvm::ffi::TensorView accept_token_num,
    tvm::ffi::TensorView candidates,
    tvm::ffi::TensorView retrieve_index,
    tvm::ffi::TensorView retrieve_next_token,
    tvm::ffi::TensorView uniform_samples,
    tvm::ffi::TensorView uniform_samples_for_final_sampling,
    tvm::ffi::TensorView target_probs,
    double threshold_single,
    double threshold_acc) {
  using namespace host;
  using namespace flashinfer::sampling;
  using namespace cub;
  SymbolicSize B{"batch_size"}, W{"draft_width"}, S{"accept_width"}, V{"vocab_size"};
  SymbolicDevice dev;
  dev.set_options<kDLCUDA>();
  TensorMatcher({B, W, V}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(target_probs);
  TensorMatcher({B, W})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(dev)
      .verify(candidates)
      .verify(retrieve_index)
      .verify(retrieve_next_token);
  TensorMatcher({B, W}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(uniform_samples);
  TensorMatcher({B}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(uniform_samples_for_final_sampling);
  TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(accept_token_num);
  TensorMatcher({B, S}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(accept_index);
  CHECK_HOST(B.unwrap() > 0 && W.unwrap() > 0 && V.unwrap() > 0) << "Expected nonempty target probabilities";
  CHECK_HOST(S.unwrap() > 0 && S.unwrap() <= W.unwrap()) << "Invalid chain acceptance width";
  CHECK_HOST(B.unwrap() <= std::numeric_limits<int32_t>::max() / W.unwrap()) << "Chain indices exceed int32";
  const int64_t rows = B.unwrap() * W.unwrap();
  SymbolicSize N{"prediction_count"};
  N.set_value(rows);
  TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(predicts);
  CHECK_HOST(V.unwrap() <= std::numeric_limits<uint32_t>::max() / rows) << "Probability offsets exceed uint32";
  CHECK_HOST(V.unwrap() <= std::numeric_limits<int32_t>::max()) << "Sampled token IDs exceed int32";
  CHECK_HOST(V.unwrap() % VEC_SIZE == 0) << "Vocabulary must be divisible by vector width";
  CHECK_HOST(threshold_single >= 0 && threshold_single <= 1) << "threshold_single must be in [0, 1]";
  CHECK_HOST(threshold_acc >= 0 && threshold_acc <= 1) << "threshold_acc must be in [0, 1]";
  constexpr uint32_t kBlockThreads = 1024;
  const size_t smem_size = sizeof(SamplingTempStorage<kBlockThreads, SCAN_ALGO, REDUCE_ALGO>);
  auto kernel = chain_target_only_kernel<VEC_SIZE>;
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  LaunchKernel(B.unwrap(), kBlockThreads, dev.unwrap(), smem_size)(
      kernel,
      static_cast<int32_t*>(predicts.data_ptr()),
      static_cast<int32_t*>(accept_index.data_ptr()),
      static_cast<int32_t*>(accept_token_num.data_ptr()),
      static_cast<const int64_t*>(candidates.data_ptr()),
      static_cast<const int64_t*>(retrieve_index.data_ptr()),
      static_cast<const int64_t*>(retrieve_next_token.data_ptr()),
      static_cast<const float*>(uniform_samples.data_ptr()),
      static_cast<const float*>(uniform_samples_for_final_sampling.data_ptr()),
      static_cast<const float*>(target_probs.data_ptr()),
      static_cast<uint32_t>(S.unwrap()),
      static_cast<uint32_t>(W.unwrap()),
      static_cast<uint32_t>(V.unwrap()),
      static_cast<float>(threshold_single),
      fmaxf(static_cast<float>(threshold_acc), 1e-9f));
}

}  // namespace
