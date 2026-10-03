// QWEN-EXO Engram: gather n-gram table rows from pinned host memory and dequantize.
//
// The table (FP8 E4M3 rows + one fp32 scale per row) lives in a cudaHostRegister'ed
// host buffer and is read zero-copy over PCIe, so it never occupies VRAM. One block
// handles one token: kHeads rows of kRowBytes bytes, one 16-byte chunk per thread.
// out[t, h, :] = bf16_rn(float(fp8(data[rows[t, h], :])) * scale[rows[t, h]]), which is
// bit-identical to the torch reference (fp8 -> fp32 is exact, one fp32 multiply).

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>

namespace {

constexpr uint32_t kChunkBytes = 16;
constexpr uint32_t kMaxBlocks = 65535;

template <uint32_t kHeads, uint32_t kRowBytes>
__global__ void __launch_bounds__(kHeads* kRowBytes / kChunkBytes) engram_gather_dequant_kernel(
    const uint8_t* __restrict__ data,
    const float* __restrict__ scale,
    const int64_t* __restrict__ rows,
    __nv_bfloat16* __restrict__ out,
    uint32_t num_tokens) {
  constexpr uint32_t kThreadsPerRow = kRowBytes / kChunkBytes;
  __shared__ int64_t s_row[kHeads];
  __shared__ float s_scale[kHeads];
  const uint32_t head = threadIdx.x / kThreadsPerRow;
  const uint32_t chunk = threadIdx.x % kThreadsPerRow;

  for (uint32_t token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    if (threadIdx.x < kHeads) {
      const int64_t row = rows[static_cast<int64_t>(token) * kHeads + threadIdx.x];
      s_row[threadIdx.x] = row;
      s_scale[threadIdx.x] = scale[row];
    }
    __syncthreads();

    const uint4 packed = *reinterpret_cast<const uint4*>(data + s_row[head] * kRowBytes + chunk * kChunkBytes);
    const float row_scale = s_scale[head];
    const auto* bytes = reinterpret_cast<const uint8_t*>(&packed);
    __align__(16) __nv_bfloat16 values[kChunkBytes];
#pragma unroll
    for (uint32_t i = 0; i < kChunkBytes; ++i) {
      __nv_fp8_e4m3 value;
      value.__x = bytes[i];
      values[i] = __float2bfloat16_rn(static_cast<float>(value) * row_scale);
    }
    auto* dst = reinterpret_cast<uint4*>(
        out + (static_cast<int64_t>(token) * kHeads + head) * kRowBytes + chunk * kChunkBytes);
    dst[0] = reinterpret_cast<const uint4*>(values)[0];
    dst[1] = reinterpret_cast<const uint4*>(values)[1];
    __syncthreads();  // s_row / s_scale are rewritten by the next token
  }
}

template <uint32_t kHeads, uint32_t kRowBytes>
struct QwenExoEngramKernel {
  static_assert(kRowBytes % kChunkBytes == 0, "Engram row bytes must be a multiple of 16");

  static void gather_dequant(
      const tvm::ffi::TensorView data,
      const tvm::ffi::TensorView scale,
      const tvm::ffi::TensorView rows,
      const tvm::ffi::TensorView out) {
    using namespace host;

    auto R = SymbolicSize{"table rows"};
    auto M = SymbolicSize{"gathered rows"};
    auto device = SymbolicDevice{};
    TensorMatcher({R, static_cast<int64_t>(kRowBytes)})  //
        .with_dtype<uint8_t>()
        .with_device<kDLGPUHost, kDLCPU, kDLGPU>()
        .verify(data);
    TensorMatcher({R})  //
        .with_dtype<float>()
        .with_device<kDLGPUHost, kDLCPU, kDLGPU>()
        .verify(scale);
    TensorMatcher({M})  //
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device)
        .verify(rows);
    TensorMatcher({M, static_cast<int64_t>(kRowBytes)})  //
        .with_dtype<bf16_t>()
        .with_device<kDLGPU>(device)
        .verify(out);

    const auto num_rows = static_cast<uint64_t>(M.unwrap());
    RuntimeCheck(num_rows % kHeads == 0, "Engram gather: rows must be a multiple of the head count");
    const auto num_tokens = static_cast<uint32_t>(num_rows / kHeads);
    if (num_tokens == 0) {
      return;
    }
    const auto grid = std::min<uint32_t>(num_tokens, kMaxBlocks);
    LaunchKernel(grid, kHeads * kRowBytes / kChunkBytes, device.unwrap())(
        engram_gather_dequant_kernel<kHeads, kRowBytes>,
        static_cast<const uint8_t*>(data.data_ptr()),
        static_cast<const float*>(scale.data_ptr()),
        static_cast<const int64_t*>(rows.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()),
        num_tokens);
  }
};

}  // namespace
