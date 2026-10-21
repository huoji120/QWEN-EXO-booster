// SPDX-License-Identifier: Apache-2.0
// Adapted from SSHdotCodes/qwen-3.8-flash-next-pro6000 qf_w8g.cu,
// revision 1233bf628f42940ec8e4fb060fa35c1a30eecfc2.
// Modified: TVM-FFI validation/launch, no globals or env switches, G128 tails,
// and BF16 rounding of q*scale before MMA (matching the prefill unpack).
#include <sgl_kernel/tensor.h>  // For TensorMatcher and symbolic dimensions/device
#include <sgl_kernel/utils.h>  // For CHECK_HOST and div_ceil
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, BF16 types and PDL primitives
#include <tvm/ffi/container/tensor.h>
#include <cstdint>

namespace {

SGL_DEVICE uint32_t sm120_word(const uint4& value, int i) {
  return i == 0 ? value.x : i == 1 ? value.y : i == 2 ? value.z : value.w;
}

SGL_DEVICE void sm120_dequant4(uint32_t bytes, float scale, uint32_t& lo, uint32_t& hi) {
  // Byte permutation produces exact signed INT8 values in FP32; their upper
  // halves are exact BF16 integers. Scale each pair before MMA, preserving
  // the same BF16(q * scale) rounding used by the large-row unpack path.
  bytes ^= 0x80808080u;
  const float f0 = __uint_as_float(__byte_perm(bytes, 0x4B000000u, 0x7440)) - 8388736.f;
  const float f1 = __uint_as_float(__byte_perm(bytes, 0x4B000000u, 0x7441)) - 8388736.f;
  const float f2 = __uint_as_float(__byte_perm(bytes, 0x4B000000u, 0x7442)) - 8388736.f;
  const float f3 = __uint_as_float(__byte_perm(bytes, 0x4B000000u, 0x7443)) - 8388736.f;
  lo = __byte_perm(__float_as_uint(f0), __float_as_uint(f1), 0x7632);
  hi = __byte_perm(__float_as_uint(f2), __float_as_uint(f3), 0x7632);
  const auto scales = __float2bfloat162_rn(scale);
  const auto low = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&lo), scales);
  const auto high = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&hi), scales);
  lo = *reinterpret_cast<const uint32_t*>(&low);
  hi = *reinterpret_cast<const uint32_t*>(&high);
}

SGL_DEVICE void sm120_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                        uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A CTA owns 16 output channels, a warp a disjoint K slice. The same
// permutation of K is used for A and B fragments; token columns 0..7 are
// zero-padded to cover M=1..8 without launching distinct graph kernels.
template <int NB, int NW>
__global__ __launch_bounds__(NW * 32) void sm120_w8_mma_kernel(
    const bf16_t* __restrict__ x, const int8_t* __restrict__ q,
    const bf16_t* __restrict__ scales, bf16_t* __restrict__ y, int n, int m) {
  constexpr int K = NW * NB * 64;
  constexpr int NG = K / 128;
  __shared__ float reduction[NW][16][9];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t = lane & 3, first = blockIdx.x * 16;
  const size_t ra = min(first + g, n - 1), rb = min(first + g + 8, n - 1);
  const int kw = warp * NB * 64 + t * 16;
  uint4 wa[NB], wb[NB];
  float sa[NB], sb[NB];
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    wa[b] = __ldcs(reinterpret_cast<const uint4*>(q + ra * K + kw + b * 64));
    wb[b] = __ldcs(reinterpret_cast<const uint4*>(q + rb * K + kw + b * 64));
    const int group = (warp * NB * 64 + b * 64) / 128;
    sa[b] = __bfloat162float(__ldg(scales + ra * NG + group));
    sb[b] = __bfloat162float(__ldg(scales + rb * NG + group));
  }
  // Only immutable weights may be read before the primary producer completes.
  device::PDLWaitPrimary<true>();
  uint4 xv[NB][2];
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    if (g < m) {
      const auto* xp = reinterpret_cast<const uint4*>(x + size_t(g) * K + kw + b * 64);
      xv[b][0] = __ldcg(xp);
      xv[b][1] = __ldcg(xp + 1);
    } else {
      xv[b][0] = make_uint4(0, 0, 0, 0);
      xv[b][1] = make_uint4(0, 0, 0, 0);
    }
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int b = 0; b < NB; ++b) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint32_t a0, a1, a2, a3;
      sm120_dequant4(sm120_word(wa[b], j), sa[b], a0, a2);
      sm120_dequant4(sm120_word(wb[b], j), sb[b], a1, a3);
      const uint4& token = xv[b][j >> 1];
      sm120_mma(acc, a0, a1, a2, a3, sm120_word(token, (j & 1) * 2),
                sm120_word(token, (j & 1) * 2 + 1));
    }
  }
  reduction[warp][g][2 * t] = acc[0];
  reduction[warp][g][2 * t + 1] = acc[1];
  reduction[warp][g + 8][2 * t] = acc[2];
  reduction[warp][g + 8][2 * t + 1] = acc[3];
  __syncthreads();
  // NW=2 has only 64 threads, while M=8 needs 128 outputs.
  for (int index = threadIdx.x; index < 16 * m; index += blockDim.x) {
    const int row = index & 15, token = index >> 4;
    float value = 0.f;
#pragma unroll
    for (int w = 0; w < NW; ++w) value += reduction[w][row][token];
    if (first + row < n) y[size_t(token) * n + first + row] = __float2bfloat16_rn(value);
  }
  device::PDLTriggerSecondary<true>();
}

template <int K>
void sm120_int8_gemm(tvm::ffi::TensorView y, tvm::ffi::TensorView x,
                    tvm::ffi::TensorView q, tvm::ffi::TensorView scales) {
  using namespace host;
  SymbolicSize M{"M"}, N{"N"};
  SymbolicDevice dev;
  TensorMatcher({M, K}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).verify(x);
  TensorMatcher({N, K}).with_dtype<int8_t>().with_device<kDLCUDA>(dev).verify(q);
  TensorMatcher({N, K / 128}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).verify(scales);
  TensorMatcher({M, N}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).verify(y);
  CHECK_HOST(M.unwrap() >= 1 && M.unwrap() <= 8 && N.unwrap() > 0) << "requires 1..8 rows and N>0";
  CHECK_HOST(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
             reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0) << "requires 16-byte-aligned x and q";
  constexpr int NW = K == 640 ? 2 : 8;
  constexpr int NB = K / (NW * 64);
  LaunchKernel(div_ceil(N.unwrap(), int64_t(16)), NW * 32, dev.unwrap()).enable_pdl(true)(
      sm120_w8_mma_kernel<NB, NW>, static_cast<const bf16_t*>(x.data_ptr()),
      static_cast<const int8_t*>(q.data_ptr()), static_cast<const bf16_t*>(scales.data_ptr()),
      static_cast<bf16_t*>(y.data_ptr()), int(N.unwrap()), int(M.unwrap()));
}

__global__ void sm120_unpack_kernel(bf16_t* out, const int8_t* q, const bf16_t* scales,
                                    int64_t total, int k, int groups) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += int64_t(gridDim.x) * blockDim.x) {
    const auto row = i / k;
    const auto group = (i % k) / 128;
    out[i] = __float2bfloat16_rn(float(q[i]) * __bfloat162float(scales[row * groups + group]));
  }
}

void sm120_int8_unpack(tvm::ffi::TensorView out, tvm::ffi::TensorView q,
                      tvm::ffi::TensorView scales) {
  using namespace host;
  SymbolicSize N{"N"}, K{"K"};
  SymbolicDevice dev;
  TensorMatcher({N, K}).with_dtype<int8_t>().with_device<kDLCUDA>(dev).verify(q);
  TensorMatcher({N, K}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).verify(out);
  const auto groups = div_ceil(K.unwrap(), int64_t(128));
  TensorMatcher({N, groups}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).verify(scales);
  CHECK_HOST(K.unwrap() > 0 && N.unwrap() > 0) << "requires nonempty weights";
  const auto total = N.unwrap() * K.unwrap();
  LaunchKernel(div_ceil(total, int64_t(256)), 256, dev.unwrap())(
      sm120_unpack_kernel, static_cast<bf16_t*>(out.data_ptr()),
      static_cast<const int8_t*>(q.data_ptr()), static_cast<const bf16_t*>(scales.data_ptr()),
      total, int(K.unwrap()), int(groups));
}

}  // namespace
