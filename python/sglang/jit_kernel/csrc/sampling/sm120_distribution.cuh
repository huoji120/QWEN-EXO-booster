// Adapted from SSHdotCodes/qwen-3.8-flash-next-pro6000 revision
// 1233bf628f42940ec8e4fb060fa35c1a30eecfc2 (qf_topk.cu/qf_sample.cu).
// Modified: full-vocabulary q, tie-complete device gating and streamed dense
// radix fallback. No hot-token map and no device-to-host branch decisions.
#include "sm120_sampling.cuh"

namespace {
__device__ double block_sum(double value, double* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int n = blockDim.x / 2; n; n /= 2) {
    if (threadIdx.x < n) scratch[threadIdx.x] += scratch[threadIdx.x + n];
    __syncthreads();
  }
  return scratch[0];
}

__global__ void eligibility_kernel(const float* vals, const int32_t* ks,
                                   const float* temps, int32_t* eligible,
                                   int depth, int vocab) {
  const int b = blockIdx.x;
  if (threadIdx.x) return;
  const int k = ks[b];
  bool ok = k >= 1 && k <= 56 && temps[b] > 0 && isfinite(temps[b]);
  for (int d = 0; d < depth && ok; ++d) {
    const float* v = vals + (b * depth + d) * 64;
    // Strict inequality proves that ALL tokens tied at the cutoff fit. A
    // fixed number of spare candidates does not establish this property.
    const float tail_weight = expf(v[63] / temps[b] - v[0] / temps[b]);
    const float cutoff_weight = expf(v[k - 1] / temps[b] - v[0] / temps[b]);
    // Conservatively decline near-equalities that can collapse to the same
    // FP32 softmax probability, not just raw-logit cutoff equalities.
    ok = isfinite(v[0]) && isfinite(v[k - 1]) &&
         (vocab == 64 || tail_weight < cutoff_weight * (1.f - 1.e-5f));
  }
  eligible[b] = ok;
}

// 64 sorted candidates, top-k ties retained for p; draft q deliberately uses
// a bounded rank-prefix proposal. Top-p uses a probability pivot (ties kept).
__global__ void sparse_distribution_kernel(
    const float* vals, const int64_t* ids, const float* temps,
    const int32_t* ks, const float* ps, const int32_t* eligible,
    float* sparse, float* dense, const float* expo, float* px, int64_t* x,
    int depth, int vocab, bool proposal) {
  const int row = blockIdx.x, b = row / depth, i = threadIdx.x;
  if (!proposal && !eligible[b]) return;
  __shared__ float weights[64];
  __shared__ float probabilities[64];
  __shared__ float scores[64];
  __shared__ float norm;
  const float* v = vals + row * 64;
  const int k = proposal ? (ks[b] > 0 ? min(ks[b], 64) : 64) : ks[b];
  const float threshold = v[k - 1] / temps[b];
  float w = (proposal ? i < k : v[i] / temps[b] >= threshold)
                ? expf(v[i] / temps[b] - v[0] / temps[b]) : 0.f;
  weights[i] = w;
  __syncthreads();
  if (!i) {
    double sum = 0;
    for (int j = 0; j < 64; ++j) sum += weights[j];
    norm = sum;
  }
  __syncthreads();
  probabilities[i] = w / norm;
  __syncthreads();
  if (!i) {
    double cdf = 0;
    float pivot = 0;
    if (ps[b] < 1.f) {
      for (int j = 0; j < 64; ++j) {
        cdf += probabilities[j];
        if (cdf >= ps[b]) { pivot = probabilities[j]; break; }
      }
    }
    double sum = 0;
    for (int j = 0; j < 64; ++j) {
      if (probabilities[j] < pivot) probabilities[j] = 0;
      sum += probabilities[j];
    }
    norm = sum;
  }
  __syncthreads();
  const float p = probabilities[i] / norm;
  sparse[row * 64 + i] = p;
  if (!proposal) return;
  const int64_t token = ids[row * 64 + i];
  dense[row * vocab + token] = p;
  scores[i] = p > 0 ? p / fmaxf(expo[row * 64 + i], 1.17549435e-38f) : -1.f;
  __syncthreads();
  if (!i) {
    int best = 0;
    for (int j = 1; j < 64; ++j)
      if (scores[j] > scores[best]) best = j;
    x[row] = ids[row * 64 + best];
    px[row] = sparse[row * 64 + best];
  }
}

// Only requests rejected by eligibility execute this kernel. It constructs the
// FULL dense target p, including arbitrarily large cutoff ties and arbitrary
// top-k/top-p settings. Histogram passes stream the vocabulary, avoiding a
// vocabulary-sized local array or host readback. Weighted radix selection
// finds the top-p pivot; the final >= comparison includes its complete tie.
__global__ void dense_distribution_kernel(
    const float* logits, const float* vals, const float* temps,
    const int32_t* ks, const float* ps, const int32_t* eligible,
    float* dense, int depth, int vocab) {
  const int row = blockIdx.x, b = row / depth, tid = threadIdx.x;
  if (eligible[b]) return;
  __shared__ int counts[256];
  __shared__ double hist[256], scratch[256];
  __shared__ uint32_t prefix, mask, cutoff;
  __shared__ int remaining;
  __shared__ double mass;
  const float* z = logits + row * vocab;
  const float temp = temps[b];
  const float maximum = vals[row * 64] / temp;
  if (!tid) { prefix = 0; mask = 0; remaining = ks[b]; cutoff = f2key(-INFINITY); }
  __syncthreads();
  if (ks[b] > 0 && ks[b] < vocab) {
    for (int shift = 24; shift >= 0; shift -= 8) {
      counts[tid] = 0;
      __syncthreads();
      for (int j = tid; j < vocab; j += 256) {
        const uint32_t key = f2key(z[j] / temp);
        if ((key & mask) == prefix) atomicAdd(counts + ((key >> shift) & 255), 1);
      }
      __syncthreads();
      if (!tid) {
        for (int bin = 255; bin >= 0; --bin) {
          if (remaining <= counts[bin]) { prefix |= uint32_t(bin) << shift; break; }
          remaining -= counts[bin];
        }
        mask |= 255u << shift;
      }
      __syncthreads();
    }
    if (!tid) cutoff = prefix;
    __syncthreads();
  }
  const float threshold = key2f(cutoff);
  double sum = 0;
  for (int j = tid; j < vocab; j += 256)
    if (z[j] / temp >= threshold) sum += expf(z[j] / temp - maximum);
  const double total = block_sum(sum, scratch);
  if (!tid) { mass = double(ps[b]) * total; prefix = 0; mask = 0; }
  __syncthreads();
  if (ps[b] < 1.f) {
    for (int shift = 24; shift >= 0; shift -= 8) {
      hist[tid] = 0;
      __syncthreads();
      for (int j = tid; j < vocab; j += 256) {
        const float value = z[j] / temp;
        const uint32_t key = f2key(value);
        if (value >= threshold && (key & mask) == prefix)
          atomicAdd(hist + ((key >> shift) & 255), double(expf(value - maximum)));
      }
      __syncthreads();
      if (!tid) {
        for (int bin = 255; bin >= 0; --bin) {
          if (mass <= hist[bin] && hist[bin] > 0) { prefix |= uint32_t(bin) << shift; break; }
          mass -= hist[bin];
        }
        mask |= 255u << shift;
      }
      __syncthreads();
    }
    if (!tid) cutoff = prefix;
    __syncthreads();
  }
  const float pivot = key2f(cutoff);
  sum = 0;
  for (int j = tid; j < vocab; j += 256)
    if (z[j] / temp >= pivot) sum += expf(z[j] / temp - maximum);
  const double kept = block_sum(sum, scratch);
  for (int j = tid; j < vocab; j += 256)
    dense[row * vocab + j] = z[j] / temp >= pivot ? float(expf(z[j] / temp - maximum) / kept) : 0.f;
}

// Same chain accept/residual law and token-ID-order CDF as the dense Triton
// implementation. The target support is complete by device-side eligibility.
__global__ void sparse_chain_kernel(
    const float* p, const int64_t* ids, const float* q, const int32_t* eligible,
    const int64_t* candidates, const int64_t* retrieve, const float* coins,
    const float* final_coins, int32_t* predicts, int32_t* accept_index,
    int32_t* accept_num, int depth, int vocab, int qdepth, int adepth) {
  const int b = blockIdx.x;
  if (threadIdx.x || !eligible[b]) return;
  int current = 0, accepted = 0, last = retrieve[b * depth];
  accept_index[b * adepth] = last;
  bool all = true;
  for (int step = 1; step < depth; ++step) {
    const int token = candidates[b * depth + step];
    const float* rowp = p + (b * depth + current) * 64;
    const int64_t* rowids = ids + (b * depth + current) * 64;
    float target = 0;
    for (int j = 0; j < 64; ++j) if (rowids[j] == token) target = rowp[j];
    const float proposal = q[(b * qdepth + current) * vocab + token];
    if (coins[b * depth + step - 1] * proposal < target) {
      predicts[last] = token;
      current = step;
      last = retrieve[b * depth + step];
      accept_index[b * adepth + ++accepted] = last;
    } else { all = false; break; }
  }
  accept_num[b] = accepted;
  const float* rowp = p + (b * depth + current) * 64;
  const int64_t* rowids = ids + (b * depth + current) * 64;
  float weights[64];
  int order[64], n = 0;
  for (int j = 0; j < 64; ++j) {
    if (!(rowp[j] > 0)) continue;
    const float proposal = all ? 0 : q[(b * qdepth + current) * vocab + rowids[j]];
    const float value = all ? rowp[j] : fmaxf(rowp[j] - proposal, 0.f);
    int i = n++;
    while (i && rowids[order[i - 1]] > rowids[j]) {
      order[i] = order[i - 1]; weights[i] = weights[i - 1]; --i;
    }
    order[i] = j; weights[i] = value;
  }
  float norm = 0;
  for (int j = 0; j < n; ++j) norm += weights[j];
  const float u = final_coins[b] * norm;
  float cdf = 0;
  int token = vocab - 1;
  for (int j = 0; j < n; ++j) {
    cdf += weights[j];
    if (cdf > u) { token = rowids[order[j]]; break; }
  }
  predicts[last] = token;
}
}  // namespace

void sm120_distribution(tvm::ffi::TensorView logits, tvm::ffi::TensorView vals,
    tvm::ffi::TensorView ids, tvm::ffi::TensorView temps, tvm::ffi::TensorView ks,
    tvm::ffi::TensorView ps, tvm::ffi::TensorView sparse, tvm::ffi::TensorView dense,
    tvm::ffi::TensorView eligible, tvm::ffi::TensorView expo,
    tvm::ffi::TensorView px, tvm::ffi::TensorView x, int64_t depth, bool proposal) {
  using namespace host;
  SymbolicSize R{"rows"}, V{"vocab"}, B{"requests"};
  SymbolicDevice dev; dev.set_options<kDLCUDA>();
  TensorMatcher({R, V}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(logits).verify(dense);
  TensorMatcher({R, 64}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(vals).verify(sparse).verify(expo);
  TensorMatcher({R, 64}).with_dtype<int64_t>().with_device<kDLCUDA>(dev).verify(ids);
  TensorMatcher({B}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(temps).verify(ps);
  TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(ks).verify(eligible);
  TensorMatcher({R, 1}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(px);
  TensorMatcher({R, 1}).with_dtype<int64_t>().with_device<kDLCUDA>(dev).verify(x);
  const int rows = R.unwrap(), vocab = V.unwrap(), batch = B.unwrap();
  CHECK_HOST(depth >= 1 && depth <= 8 && rows == batch * depth && vocab >= 64)
      << "invalid sampling dimensions";
  CHECK_HOST(!proposal || depth == 1) << "proposal requires one row per request";
  if (!rows) return;
  auto t = static_cast<const float*>(temps.data_ptr());
  auto k = static_cast<const int32_t*>(ks.data_ptr());
  auto tp = static_cast<const float*>(ps.data_ptr());
  auto v = static_cast<const float*>(vals.data_ptr());
  auto i = static_cast<const int64_t*>(ids.data_ptr());
  auto e = static_cast<int32_t*>(eligible.data_ptr());
  auto d = static_cast<float*>(dense.data_ptr());
  auto s = static_cast<float*>(sparse.data_ptr());
  if (!proposal) LaunchKernel(batch, 32, dev.unwrap())(eligibility_kernel, v, k, t, e, int(depth), vocab);
  LaunchKernel(rows, 64, dev.unwrap())(sparse_distribution_kernel, v, i, t, k, tp, e,
      s, d, static_cast<const float*>(expo.data_ptr()), static_cast<float*>(px.data_ptr()),
      static_cast<int64_t*>(x.data_ptr()), int(depth), vocab, proposal);
  if (!proposal) LaunchKernel(rows, 256, dev.unwrap())(dense_distribution_kernel,
      static_cast<const float*>(logits.data_ptr()), v, t, k, tp, e, d, int(depth), vocab);
}

void sm120_sparse_chain(tvm::ffi::TensorView p, tvm::ffi::TensorView ids,
    tvm::ffi::TensorView q, tvm::ffi::TensorView eligible,
    tvm::ffi::TensorView candidates, tvm::ffi::TensorView retrieve,
    tvm::ffi::TensorView coins, tvm::ffi::TensorView final_coins,
    tvm::ffi::TensorView predicts, tvm::ffi::TensorView accept_index,
    tvm::ffi::TensorView accept_num) {
  using namespace host;
  SymbolicSize B{"batch"}, D{"depth"}, Q{"qdepth"}, V{"vocab"}, A{"acceptdepth"}, R{"rows"};
  SymbolicDevice dev; dev.set_options<kDLCUDA>();
  TensorMatcher({B, D}).with_dtype<int64_t>().with_device<kDLCUDA>(dev).verify(candidates).verify(retrieve);
  TensorMatcher({B, D, 64}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(p);
  TensorMatcher({B, D, 64}).with_dtype<int64_t>().with_device<kDLCUDA>(dev).verify(ids);
  TensorMatcher({B, Q, V}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(q);
  TensorMatcher({B}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(eligible).verify(accept_num);
  TensorMatcher({B, D}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(coins);
  TensorMatcher({B}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(final_coins);
  TensorMatcher({R}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(predicts);
  TensorMatcher({B, A}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(accept_index);
  CHECK_HOST(Q.unwrap() >= D.unwrap() - 1 && A.unwrap() >= D.unwrap()
      && R.unwrap() == B.unwrap() * D.unwrap()) << "invalid chain dimensions";
  if (!B.unwrap()) return;
  LaunchKernel(B.unwrap(), 32, dev.unwrap())(sparse_chain_kernel,
      static_cast<const float*>(p.data_ptr()), static_cast<const int64_t*>(ids.data_ptr()),
      static_cast<const float*>(q.data_ptr()), static_cast<const int32_t*>(eligible.data_ptr()),
      static_cast<const int64_t*>(candidates.data_ptr()), static_cast<const int64_t*>(retrieve.data_ptr()),
      static_cast<const float*>(coins.data_ptr()), static_cast<const float*>(final_coins.data_ptr()),
      static_cast<int32_t*>(predicts.data_ptr()), static_cast<int32_t*>(accept_index.data_ptr()),
      static_cast<int32_t*>(accept_num.data_ptr()), int(D.unwrap()), int(V.unwrap()),
      int(Q.unwrap()), int(A.unwrap()));
}
