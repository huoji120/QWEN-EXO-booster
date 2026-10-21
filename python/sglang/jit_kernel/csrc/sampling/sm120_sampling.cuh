// Adapted from SSHdotCodes/qwen-3.8-flash-next-pro6000, revision
// 1233bf628f42940ec8e4fb060fa35c1a30eecfc2, qf_topk.cu and qf_sample.cu.
// Modified: SGLang JIT ABI, full-vocabulary proposals, complete-tie device
// eligibility, and a genuine full-dense radix fallback without host decisions.
#include <sgl_kernel/tensor.h>  // For TensorMatcher and symbolic dimensions
#include <sgl_kernel/utils.cuh>  // For LaunchKernel and CUDA scalar types
#include <sgl_kernel/utils.h>  // For CHECK_HOST and div_ceil
#include <tvm/ffi/container/tensor.h>
#include <cmath>
#include <cstdint>

namespace {

constexpr int THREADS = 256, NB = 64, KMAX = 64;

__device__ __forceinline__ uint32_t f2key(float f) {
    const uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float key2f(uint32_t k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k);
}

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_f<bf16_t>(bf16_t v) { return static_cast<float>(v); }

// Block-wide radix select over keys held in registers (EPT per thread, invalid = 0). Returns the k-th largest
// key (thresh) and how many elements equal to it belong to the top-k (need_eq).
template <int EPT>
__device__ void radix_select(const uint32_t (&key)[EPT], int k, uint32_t & thresh, int & need_eq,
                             int * hist, int * s_bucket) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    uint32_t prefix = 0, mask = 0;
#pragma unroll 1
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[tid] = 0;
        __syncthreads();
#pragma unroll
        for (int e = 0; e < EPT; ++e)
            if ((key[e] & mask) == prefix && key[e] != 0u) atomicAdd(&hist[(key[e] >> shift) & 255], 1);
        __syncthreads();
        if (warp == 0) {
            // lane l owns bins 255-8l .. 248-8l (descending); suffix counts from the top
            int c[8], tot = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) { c[j] = hist[255 - 8 * lane - j]; tot += c[j]; }
            int incl = tot;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffff, incl, o);
                if (lane >= o) incl += v;
            }
            int above = incl - tot;   // elements in higher bins than this lane's first bin
            if (above < k && k <= incl) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (above < k && k <= above + c[j]) { s_bucket[0] = 255 - 8 * lane - j; s_bucket[1] = k - above; }
                    above += c[j];
                }
            }
        }
        __syncthreads();
        const uint32_t b = (uint32_t) s_bucket[0];
        k = s_bucket[1];
        prefix |= b << shift;
        mask |= 0xFFu << shift;
        __syncthreads();
    }
    thresh = prefix;
    need_eq = k;
}

template <typename T, int EPT>
__global__ void __launch_bounds__(THREADS) topk_stage1(const T * __restrict__ x, int V, long ldx, int seg, int k,
                                                       uint32_t * __restrict__ ckey, int * __restrict__ cidx) {
    __shared__ int hist[256];
    __shared__ int s_bucket[2];
    __shared__ int s_cnt[2];
    const int b = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
    const int lo = b * seg, hi = min(V, lo + seg);
    const T * row = x + (size_t) r * ldx;
    uint32_t key[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = lo + e * THREADS + tid;
        key[e] = i < hi ? f2key(to_f<T>(row[i])) : 0u;
    }
    uint32_t th; int need;
    radix_select<EPT>(key, k, th, need, hist, s_bucket);
    if (tid < 2) s_cnt[tid] = 0;
    __syncthreads();
    uint32_t * ok = ckey + ((size_t) r * NB + b) * k;
    int * oi = cidx + ((size_t) r * NB + b) * k;
    const int ngt = k - need;
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = lo + e * THREADS + tid;
        if (key[e] > th) {
            const int s = atomicAdd(&s_cnt[0], 1);
            ok[s] = key[e]; oi[s] = i;
        } else if (key[e] == th && key[e] != 0u) {
            const int s = atomicAdd(&s_cnt[1], 1);
            if (s < need) { ok[ngt + s] = key[e]; oi[ngt + s] = i; }
        }
    }
    __syncthreads();
    // fewer than k valid elements in the segment: pad
    const int filled = s_cnt[0] + min(s_cnt[1], need);
    for (int s = filled + tid; s < k; s += THREADS) { ok[s] = 0u; oi[s] = -1; }
}

template <typename TO>
__global__ void __launch_bounds__(THREADS) topk_stage2(const uint32_t * __restrict__ ckey, const int * __restrict__ cidx,
                                                       int k, TO * __restrict__ vals, int64_t * __restrict__ idx) {
    constexpr int EPT = NB * KMAX / THREADS;   // 16
    __shared__ int hist[256];
    __shared__ int s_bucket[2];
    __shared__ int s_cnt[2];
    __shared__ uint32_t s_key[KMAX];
    __shared__ int s_idx[KMAX];
    const int r = blockIdx.x, tid = threadIdx.x, n = NB * k;
    const uint32_t * ck = ckey + (size_t) r * n;
    const int * ci = cidx + (size_t) r * n;
    uint32_t key[EPT];
    int ix[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int i = e * THREADS + tid;
        key[e] = i < n ? ck[i] : 0u;
        ix[e] = i < n ? ci[i] : -1;
    }
    uint32_t th; int need;
    radix_select<EPT>(key, k, th, need, hist, s_bucket);
    if (tid < 2) s_cnt[tid] = 0;
    __syncthreads();
    const int ngt = k - need;
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        if (key[e] > th) {
            const int s = atomicAdd(&s_cnt[0], 1);
            s_key[s] = key[e]; s_idx[s] = ix[e];
        } else if (key[e] == th && key[e] != 0u) {
            const int s = atomicAdd(&s_cnt[1], 1);
            if (s < need) { s_key[ngt + s] = key[e]; s_idx[ngt + s] = ix[e]; }
        }
    }
    __syncthreads();
    if (tid < k) {
        const uint32_t mk = s_key[tid];
        const int mi = s_idx[tid];
        int rank = 0;
        for (int j = 0; j < k; ++j) {
            const uint32_t oj = s_key[j];
            rank += (oj > mk) || (oj == mk && s_idx[j] < mi);
        }
        vals[(size_t) r * k + rank] = (TO) key2f(mk);
        idx[(size_t) r * k + rank] = mi;
    }
}

}  // namespace

template <typename T>
void sm120_topk(tvm::ffi::TensorView x, tvm::ffi::TensorView vals,
                tvm::ffi::TensorView idx, tvm::ffi::TensorView keys,
                tvm::ffi::TensorView ids) {
  using namespace host;
  SymbolicSize R{"rows"}, V{"vocab"}, K{"k"};
  SymbolicDevice dev;
  dev.set_options<kDLCUDA>();
  TensorMatcher({R, V}).with_dtype<T>().template with_device<kDLCUDA>(dev).verify(x);
  TensorMatcher({R, K}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(vals);
  TensorMatcher({R, K}).with_dtype<int64_t>().with_device<kDLCUDA>(dev).verify(idx);
  TensorMatcher({R, NB, K}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(keys).verify(ids);
  const int r = R.unwrap(), v = V.unwrap(), k = K.unwrap();
  const int seg = div_ceil(v, NB);
  CHECK_HOST(k >= 1 && k <= KMAX && seg >= k && seg <= 4096)
      << "unsupported radix top-k shape";
  if (!r) return;
  auto ck = reinterpret_cast<uint32_t*>(keys.data_ptr());
  auto ci = static_cast<int32_t*>(ids.data_ptr());
  LaunchKernel(dim3(NB, r), THREADS, dev.unwrap())(
      topk_stage1<T, 16>, static_cast<const T*>(x.data_ptr()), v, v, seg, k, ck, ci);
  LaunchKernel(r, THREADS, dev.unwrap())(
      topk_stage2<float>, ck, ci, k, static_cast<float*>(vals.data_ptr()),
      static_cast<int64_t*>(idx.data_ptr()));
}

