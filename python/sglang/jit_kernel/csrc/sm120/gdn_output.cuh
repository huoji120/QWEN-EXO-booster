// Adapted from SSHdotCodes/qwen-3.8-flash-next-pro6000 qf_gdnout.cu,
// revision 1233bf628f42940ec8e4fb060fa35c1a30eecfc2. Modified for SGLang JIT/FFI,
// explicit BF16/INT8 dispatch, and ordinary stream ordering (no unsafe PDL wait).
#include <sgl_kernel/tensor.h>     // For TensorMatcher and symbolic dimensions/devices
#include <sgl_kernel/utils.cuh>    // For BF16 types, LaunchKernel and CUDA error checks
#include <sgl_kernel/utils.h>      // For CHECK_HOST
#include <tvm/ffi/container/tensor.h>
#include <algorithm>
#include <cmath>

namespace {
constexpr int kHeads = 48, kHeadDim = 128, kK = 6144, kN = 2560;
constexpr int kThreads = 256, kWarps = 8, kWarpK = kK / kWarps;
constexpr int kStride = kK + 32;

SGL_DEVICE void gdn_mma(float (&d)[4], uint4 a0, uint4 a1, uint4 b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0.x), "r"(a1.x), "r"(a0.y), "r"(a1.y), "r"(b.x), "r"(b.y));
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0.z), "r"(a1.z), "r"(a0.w), "r"(a1.w), "r"(b.z), "r"(b.w));
}

template <bool kInt8>
SGL_DEVICE uint4 gdn_weight8(const void* weight, const bf16_t* scale, int row, int k) {
  if constexpr (!kInt8) {
    return __ldcs(reinterpret_cast<const uint4*>(static_cast<const bf16_t*>(weight) + row * kK + k));
  } else {
    const auto packed = __ldcs(reinterpret_cast<const uint2*>(static_cast<const int8_t*>(weight) + row * kK + k));
    const auto* bytes = reinterpret_cast<const int8_t*>(&packed);
    const float s = __bfloat162float(scale[row * (kK / 128) + k / 128]);
    uint4 result;
    auto* pairs = reinterpret_cast<bf16x2_t*>(&result);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // This rounding is part of the quantized target, shared with the large-M GEMM.
      pairs[j] = __floats2bfloat162_rn(float(bytes[2 * j]) * s, float(bytes[2 * j + 1]) * s);
    }
    return result;
  }
}

template <int kRows, bool kInt8>
__global__ void __launch_bounds__(kThreads, 1) gdn_output_kernel(
    const bf16_t* __restrict__ core, const bf16_t* __restrict__ gate,
    const bf16_t* __restrict__ norm_weight, const void* __restrict__ weight,
    const bf16_t* __restrict__ scale, bf16_t* __restrict__ output, float eps, bool swish) {
  extern __shared__ __align__(16) unsigned char scratch[];
  auto* x = reinterpret_cast<bf16_t*>(scratch);
  auto* partial = reinterpret_cast<float (*)[16][8]>(scratch);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int sub = lane >> 3, sl = lane & 7;
  const int n0 = blockIdx.x * 16;
  // Ordinary stream launch: the existing recurrent producer is Triton and does
  // not promise PDL signalling. Never wait on it or release this kernel early.
  for (int p = warp * 4 + sub; p < kRows * kHeads; p += kWarps * 4) {
    float values[16], gates[16], ss = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int d = sl * 16 + j;
      values[j] = __bfloat162float(core[p * kHeadDim + d]);
      gates[j] = __bfloat162float(gate[p * kHeadDim + d]);
      ss += values[j] * values[j];
    }
    ss += __shfl_xor_sync(0xffffffff, ss, 4);
    ss += __shfl_xor_sync(0xffffffff, ss, 2);
    ss += __shfl_xor_sync(0xffffffff, ss, 1);
    const float rstd = rsqrtf(ss / kHeadDim + eps);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int d = sl * 16 + j;
      const float g = 1.f / (1.f + expf(-gates[j]));
      const float v = values[j] * rstd * __bfloat162float(norm_weight[d]) * (swish ? gates[j] * g : g);
      x[(p / kHeads) * kStride + (p % kHeads) * kHeadDim + d] = __float2bfloat16_rn(v);
    }
  }
  __syncthreads();
  const int group = lane >> 2, tq = lane & 3;
  const int kb = warp * kWarpK + tq * 8;
  const uint4 zero = make_uint4(0, 0, 0, 0);
  float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int kc = 0; kc < kWarpK / 32; ++kc) {
    const int k = kb + kc * 32;
    const uint4 a0 = gdn_weight8<kInt8>(weight, scale, n0 + group, k);
    const uint4 a1 = gdn_weight8<kInt8>(weight, scale, n0 + group + 8, k);
    const uint4 b = group < kRows ? *reinterpret_cast<const uint4*>(x + group * kStride + k) : zero;
    gdn_mma(d, a0, a1, b);
  }
  __syncthreads();  // All normalized values have been consumed before scratch reuse.
  partial[warp][group][2 * tq] = d[0];
  partial[warp][group][2 * tq + 1] = d[1];
  partial[warp][group + 8][2 * tq] = d[2];
  partial[warp][group + 8][2 * tq + 1] = d[3];
  __syncthreads();
  if (tid < 16 * kRows) {
    const int row = tid / kRows, token = tid % kRows;
    float sum = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) sum += partial[w][row][token];
    output[token * kN + n0 + row] = __float2bfloat16_rn(sum);
  }
}

template <int kRows, bool kInt8>
void gdn_output(tvm::ffi::TensorView output, tvm::ffi::TensorView core,
                tvm::ffi::TensorView gate, tvm::ffi::TensorView norm_weight,
                tvm::ffi::TensorView weight, tvm::ffi::TensorView scale, double eps, bool swish) {
  using namespace host;
  static_assert(kRows >= 1 && kRows <= 8);
  SymbolicDevice device;
  TensorMatcher({kRows * kHeads, kHeadDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device)
      .verify(core).verify(gate);
  TensorMatcher({kHeadDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(norm_weight);
  TensorMatcher({kRows, kN}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(output);
  if constexpr (kInt8) {
    TensorMatcher({kN, kK}).with_dtype<int8_t>().with_device<kDLCUDA>(device).verify(weight);
    TensorMatcher({kN, kK / 128}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(scale);
  } else {
    TensorMatcher({kN, kK}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(weight);
    // No scale tensor is needed in the BF16 specialization; wrapper passes norm_weight.
    TensorMatcher({kHeadDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(scale);
  }
  CHECK_HOST(std::isfinite(eps) && eps > 0) << "GDN norm epsilon must be positive and finite";
  constexpr int smem = std::max(kRows * kStride * int(sizeof(bf16_t)), kWarps * 16 * 8 * int(sizeof(float)));
  static const bool configured = [] {
    CHECK_CUDA(cudaFuncSetAttribute(gdn_output_kernel<kRows, kInt8>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    return true;
  }();
  (void)configured;
  LaunchKernel(kN / 16, kThreads, device.unwrap(), smem)(
      gdn_output_kernel<kRows, kInt8>, static_cast<const bf16_t*>(core.data_ptr()),
      static_cast<const bf16_t*>(gate.data_ptr()), static_cast<const bf16_t*>(norm_weight.data_ptr()),
      weight.data_ptr(), static_cast<const bf16_t*>(scale.data_ptr()),
      static_cast<bf16_t*>(output.data_ptr()), float(eps), swish);
}
}  // namespace
