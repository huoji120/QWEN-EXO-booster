// Adapted from SSHdotCodes/qwen-3.8-flash-next-pro6000, qf_a16moe.cu
// revision 1233bf628f42940ec8e4fb060fa35c1a30eecfc2 (phase 4).
// Modified: native ModelOpt Marlin payload, routed-only output, masked experts,
// per-invocation scratch initialization and device-local occupancy sizing.
#include <sgl_kernel/tensor.h>  // For TensorMatcher and symbolic shapes/devices
#include <sgl_kernel/utils.cuh>  // For LaunchKernel, BF16 and CUDA checks
#include <sgl_kernel/runtime.cuh>  // For occupancy and device properties
#include <sgl_kernel/utils.h>  // For CHECK_HOST
#include <cuda_bf16.h>
#include <tvm/ffi/container/tensor.h>
#include <algorithm>

namespace {
constexpr int H = 2560, I = 640, TOPK = 10, MAXT = 8, MAXSLOT = MAXT * TOPK;
constexpr int THREADS = 256, WARPS = 8;
constexpr int KT1 = H / 16, KT2 = I / 16;            // 160, 40 k-tiles
constexpr int ROW1 = (2 * I) / 2, ROW2 = H / 2;      // int4 per k-tile row: 640 (w13), 1280 (w2)
constexpr int NG1 = I / 64;                          // 10 gate (and up) column groups
constexpr int NT2 = H / 128;                         // 20 output tiles of gemm2
constexpr int XS1 = H + 8, XS2 = I + 8;              // bf16 row strides in shared memory (conflict-free B reads)
constexpr int D1 = 4, D2 = 5;                        // k-tiles prefetched per warp

SGL_DEVICE uint4 ldg_nc(const uint4 * p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
SGL_DEVICE uint2 ldg_nc2(const uint2 * p) {
    uint2 v;
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}

// Marlin dequant<nv_bfloat162, kFE2M1f, skip_flop=true>: e2m1 nibbles -> bf16 bit patterns (scaled by 2^-126;
// the processed group/global scales carry the compensation). f[0] <- nibbles 8,24 ; f[1] <- nibbles 12,28.
SGL_DEVICE void dq_fp4(uint32_t q, uint32_t (&f)[2]) {
    const uint32_t o1 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
    q <<= 4;
    const uint32_t o2 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
    f[0] = o2; f[1] = o1;
}
// Marlin dequant_fp8_scales<nv_bfloat162, kFE4M3fn>: s[0] <- bytes 0,2 ; s[1] <- bytes 1,3.
SGL_DEVICE void dq_s8(uint32_t q, uint32_t & s0, uint32_t & s1) {
    const uint32_t o1 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
    q <<= 8;
    const uint32_t o2 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
    s0 = o2; s1 = o1;
}
SGL_DEVICE uint32_t bmul2(uint32_t a, uint32_t b) {
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162 *>(&a), *reinterpret_cast<__nv_bfloat162 *>(&b));
    return *reinterpret_cast<uint32_t *>(&r);
}
SGL_DEVICE uint32_t bdup_lo(uint32_t s) { return (s & 0xFFFFu) | (s << 16); }
SGL_DEVICE uint32_t bdup_hi(uint32_t s) { return (s >> 16) | (s & 0xFFFF0000u); }

SGL_DEVICE void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One 16 x 64 weight tile (this lane's int4 w + its lane group's 8 scale bytes sc) against the token fragment
// (b0, b1); acc[j] = {C[16j+grp][2tq], C[16j+grp][2tq+1], C[16j+8+grp][2tq], C[16j+8+grp][2tq+1]}.
SGL_DEVICE void tile_mma(float (&acc)[4][4], const uint4 & w, const uint2 & sc, uint32_t b0, uint32_t b1) {
    uint32_t s[4];
    dq_s8(sc.x, s[0], s[1]);
    dq_s8(sc.y, s[2], s[3]);
    const uint32_t q[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t f0[2], f1[2];
        dq_fp4(q[j] << 8, f0);      // columns 16j + grp
        dq_fp4(q[j], f1);           // columns 16j + 8 + grp
        const uint32_t s0 = bdup_lo(s[j]), s1 = bdup_hi(s[j]);
        f0[0] = bmul2(f0[0], s0); f0[1] = bmul2(f0[1], s0);
        f1[0] = bmul2(f1[0], s1); f1[1] = bmul2(f1[1], s1);
        mma(acc[j], f0[0], f1[0], f0[1], f1[1], b0, b1);
    }
}

SGL_DEVICE float bf16r(float v) { return __bfloat162float(__float2bfloat16(v)); }
// Shared slot bookkeeping: ids in shared memory; first[s] = 1 when slot s is the first with its expert.
struct SlotInfo {
    int ids[MAXSLOT];
    unsigned char first[MAXSLOT];
    int n_unique;
};

SGL_DEVICE void load_slots(SlotInfo & si, const int * __restrict__ topk_ids, int nslots) {
    const int tid = threadIdx.x;
    if (tid < nslots) si.ids[tid] = topk_ids[tid];
    __syncthreads();
    if (tid < nslots) {
        const int e = si.ids[tid];
        bool f = e >= 0 && e < 512;
        for (int s = 0; s < tid; ++s) f &= si.ids[s] != e;
        si.first[tid] = f;
    }
    __syncthreads();
    if (tid < 32) {
        int n = 0;
        for (int s = tid; s < nslots; s += 32) n += si.first[s];
#pragma unroll
        for (int o = 16; o; o >>= 1) n += __shfl_xor_sync(0xffffffffu, n, o);
        if (tid == 0) si.n_unique = n;
    }
    __syncthreads();
}


SGL_DEVICE float bf16_bits(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }

constexpr int QC = 0, CNT = 8, DONE = 32, HALF = 128, CTRL = 928;
constexpr int KH = KT1 / 2, DH = 5;

SGL_DEVICE unsigned ld_acq(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__global__ void initialize_moe(unsigned* ctrl, __nv_bfloat16* out, int M) {
    for (int i = threadIdx.x; i < CTRL; i += blockDim.x) ctrl[i] = 0;
    for (int i = threadIdx.x; i < M * H; i += blockDim.x) out[i] = __float2bfloat16(0.f);
}

// Queue order is producer-before-consumer: all gate/up halves precede all down
// tiles. The grid is bounded by occupancy, so every predecessor is resident or
// completed before a dependent item can wait. Each unique expert is read once
// for all physical token rows, including duplicate expert selections.
__global__ void __launch_bounds__(THREADS, 2) a16_moe_persist_kernel(
        const __nv_bfloat16* __restrict__ x,
        const int* __restrict__ topk_ids, const float* __restrict__ topk_w,
        const uint4* __restrict__ w13, const uint8_t* __restrict__ s13,
        const uint16_t* __restrict__ gs13,
        const uint4* __restrict__ w2, const uint8_t* __restrict__ s2,
        const uint16_t* __restrict__ gs2,
        __nv_bfloat16* __restrict__ hbuf, __nv_bfloat16* __restrict__ ybuf,
        float* __restrict__ pbuf, unsigned* __restrict__ ctrl,
        __nv_bfloat16* __restrict__ out, int M) {
    __shared__ __align__(16) float smem[WARPS * 32 * 33];
    auto* xs = reinterpret_cast<__nv_bfloat16*>(smem);
    float* red = smem;
    __shared__ SlotInfo si;
    __shared__ int ulist[MAXSLOT];
    __shared__ int s_item;
    __shared__ unsigned s_last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int grp = lane >> 2, tq = lane & 3;
    constexpr int XSH = KH * 16 + 8;
    load_slots(si, topk_ids, M * TOPK);
    if (warp == 0) {
        int base = 0;
        for (int s0 = 0; s0 < M * TOPK; s0 += 32) {
            const int s = s0 + lane;
            const int f = s < M * TOPK ? si.first[s] : 0;
            int incl = f;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            if (f) ulist[base + incl - 1] = si.ids[s];
            base += __shfl_sync(0xffffffffu, incl, 31);
        }
    }
    __syncthreads();
    const int U = si.n_unique, b2 = 0, b4 = U * NG1 * 2;
    const int bend = b4 + U * NT2;
    const unsigned tile_target = U - 1;
    auto final_tile = [&](int c) {
        // Every producer publishes its complete tile before incrementing CNT.
        __threadfence();
        __syncthreads();
        for (int q = tid; q < M * 128; q += THREADS) {
            const int t = q / 128, n = 128 * c + q % 128;
            float acc = 0.f;
#pragma unroll
            for (int kp = 0; kp < TOPK; ++kp) {
                const int e = si.ids[t * TOPK + kp];
                if (e >= 0 && e < 512)
                    acc += __bfloat162float(__ldcg(ybuf + ((size_t)t * TOPK + kp) * H + n));
            }
            out[(size_t)t * H + n] = __float2bfloat16(acc);
        }
    };
    int item = blockIdx.x;
    while (true) {
        if (item < b4) {
            // ---------------- gemm1 half item ----------------
            const int it = item - b2, u = it / (2 * NG1), g = (it >> 1) % NG1, kh = it & 1, e = ulist[u];
            const int k0 = kh * KH;
            const uint4 * wb = w13 + (size_t) e * KT1 * ROW1 + lane;
            const uint8_t * sb = s13 + (size_t) e * KT1 * (2 * I) + grp * 8;
            uint4 wg[DH], wu[DH];
            uint2 sg[DH], su[DH];
#pragma unroll
            for (int d = 0; d < DH; ++d) {
                const int kt = k0 + warp + d * WARPS;
                wg[d] = ldg_nc(wb + kt * ROW1 + g * 32);
                wu[d] = ldg_nc(wb + kt * ROW1 + (NG1 + g) * 32);
                sg[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * (2 * I) + g * 64));
                su[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * (2 * I) + (NG1 + g) * 64));
            }
            for (int c = tid; c < MAXT * (KH * 2); c += THREADS) {            // half of each token row: KH*16 bf16
                const int t = c / (KH * 2), v = c % (KH * 2);
                uint4 val = make_uint4(0, 0, 0, 0);
                if (t < M) val = *reinterpret_cast<const uint4 *>(x + (size_t) t * H + k0 * 16 + v * 8);
                *reinterpret_cast<uint4 *>(xs + t * XSH + v * 8) = val;
            }
            __syncthreads();
            float ag[4][4], au[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) ag[j][i] = au[j][i] = 0.f;
            const uint32_t * xr = reinterpret_cast<const uint32_t *>(xs + grp * XSH) + tq;
#pragma unroll
            for (int i2 = 0; i2 < KH / WARPS; ++i2) {
                const int d = i2 % DH;
                const int ktl = warp + i2 * WARPS;                 // k-tile within the half
                const uint32_t b0 = xr[ktl * 8], b1 = xr[ktl * 8 + 4];
                const uint4 cwg = wg[d], cwu = wu[d];
                const uint2 csg = sg[d], csu = su[d];
                const int nk = ktl + DH * WARPS;
                if (nk < KH) {
                    const int kt = k0 + nk;
                    wg[d] = ldg_nc(wb + kt * ROW1 + g * 32);
                    wu[d] = ldg_nc(wb + kt * ROW1 + (NG1 + g) * 32);
                    sg[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * (2 * I) + g * 64));
                    su[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * (2 * I) + (NG1 + g) * 64));
                }
                tile_mma(ag, cwg, csg, b0, b1);
                tile_mma(au, cwu, csu, b0, b1);
            }
            __syncthreads();
            float * r = red + (warp * 32 + lane) * 33;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) { r[j * 4 + i] = ag[j][i]; r[16 + j * 4 + i] = au[j][i]; }
            __syncthreads();
            // reduce over warps: thread -> 4 of the 1024 (lane, 32) values; this half's partial
            float part[4];
            float * pb = pbuf + ((size_t) (u * NG1 + g) * 2) * 1024;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int idx = tid + p * THREADS;          // 0..1023 = ln * 32 + v
                const int ln = idx >> 5, v = idx & 31;
                float s = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) s += red[(w * 32 + ln) * 33 + v];
                part[p] = s;
                __stcg(pb + kh * 1024 + idx, s);
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) s_last = atomicAdd(ctrl + HALF + u * NG1 + g, 1u) == 1u;
            __syncthreads();
            if (s_last) {
                __threadfence();
                const float gs = bf16_bits(gs13[e]);
                // both halves summed in fixed order (half 0 + half 1) so the result does not depend on timing
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const int idx = tid + p * THREADS;
                    const float other = __ldcg(pb + (1 - kh) * 1024 + idx);
                    red[idx] = kh == 0 ? part[p] + other : other + part[p];
                }
                __syncthreads();
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int idx = tid * 2 + p;            // (ln, ji) gate at ln*32+ji, up at ln*32+16+ji
                    const int ln = idx >> 4, ji = idx & 15, j = ji >> 2, i = ji & 3;
                    const float sg_ = red[ln * 32 + ji], su_ = red[ln * 32 + 16 + ji];
                    const int n = 16 * j + (ln >> 2) + ((i & 2) ? 8 : 0);
                    const int t = 2 * (ln & 3) + (i & 1);
                    if (t < M) {
                        const float gv = bf16r(bf16r(sg_) * gs), uv = bf16r(bf16r(su_) * gs);
                        hbuf[((size_t) u * MAXT + t) * I + g * 64 + n] = __float2bfloat16(gv / (1.f + __expf(-gv)) * uv);
                    }
                }
                __threadfence();
                __syncthreads();
                if (tid == 0) { ctrl[HALF + u * NG1 + g] = 0u; atomicAdd(ctrl + DONE + u, 1u); }
            }

        } else if (item >= b4 && item < bend) {
            // ---------------- gemm2 item ----------------
            const int j2 = item - b4, u = j2 / NT2, c = j2 % NT2, e = ulist[u];
            const uint4 * wb = w2 + (size_t) e * KT2 * ROW2 + lane;
            const uint8_t * sb = s2 + (size_t) e * KT2 * H + grp * 8;
            uint4 wa[D2], wc[D2];
            uint2 sa[D2], sc[D2];
#pragma unroll
            for (int d = 0; d < D2; ++d) {
                const int kt = warp + d * WARPS;
                wa[d] = ldg_nc(wb + kt * ROW2 + (2 * c) * 32);
                wc[d] = ldg_nc(wb + kt * ROW2 + (2 * c + 1) * 32);
                sa[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * H + (2 * c) * 64));
                sc[d] = ldg_nc2(reinterpret_cast<const uint2 *>(sb + kt * H + (2 * c + 1) * 64));
            }
            if (tid == 0) {
                while (ld_acq(ctrl + DONE + u) < (unsigned) NG1) __nanosleep(32);
            }
            __syncthreads();

            uint32_t hb[D2][2];
            {
                const unsigned * hrow = reinterpret_cast<const unsigned *>(hbuf + ((size_t) u * MAXT + grp) * I) + tq;
#pragma unroll
                for (int d = 0; d < D2; ++d) {
                    const int kt = warp + d * WARPS;
                    hb[d][0] = grp < M ? __ldcg(hrow + kt * 8) : 0u;
                    hb[d][1] = grp < M ? __ldcg(hrow + kt * 8 + 4) : 0u;
                }
            }
            float aa[4][4], ac[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) aa[j][i] = ac[j][i] = 0.f;
#pragma unroll
            for (int d = 0; d < D2; ++d) {
                tile_mma(aa, wa[d], sa[d], hb[d][0], hb[d][1]);
                tile_mma(ac, wc[d], sc[d], hb[d][0], hb[d][1]);
            }
            __syncthreads();
            float * r = red + (warp * 32 + lane) * 33;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int i = 0; i < 4; ++i) { r[j * 4 + i] = aa[j][i]; r[16 + j * 4 + i] = ac[j][i]; }
            __syncthreads();
            const float gsv = bf16_bits(gs2[e]);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int idx = tid * 4 + p;
                const int ln = idx >> 5, hj = idx & 31, half = hj >> 4, ji = hj & 15, j = ji >> 2, i = ji & 3;
                float s = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) s += red[(w * 32 + ln) * 33 + hj];
                const int n = 128 * c + 64 * half + 16 * j + (ln >> 2) + ((i & 2) ? 8 : 0);
                const int t = 2 * (ln & 3) + (i & 1);
                if (t < M) {
#pragma unroll
                    for (int kp = 0; kp < TOPK; ++kp) {
                        if (si.ids[t * TOPK + kp] == e) {
                            const float tw = bf16r(gsv * bf16r(topk_w[t * TOPK + kp]));
                            ybuf[((size_t) t * TOPK + kp) * H + n] = __float2bfloat16(bf16r(s) * tw);
                        }
                    }
                }
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) s_last = atomicAdd(ctrl + CNT + c, 1u) == tile_target;
            __syncthreads();

            if (s_last) final_tile(c);
        } else {
            break;
        }
        if (tid == 0) s_item = (int) (gridDim.x + atomicAdd(ctrl + QC, 1u));
        __syncthreads();
        item = s_item;
        __syncthreads();
    }

}

void persistent_moe(tvm::ffi::TensorView out, tvm::ffi::TensorView x,
        tvm::ffi::TensorView ids, tvm::ffi::TensorView weights,
        tvm::ffi::TensorView w13, tvm::ffi::TensorView s13, tvm::ffi::TensorView gs13,
        tvm::ffi::TensorView w2, tvm::ffi::TensorView s2, tvm::ffi::TensorView gs2,
        tvm::ffi::TensorView hbuf, tvm::ffi::TensorView ybuf,
        tvm::ffi::TensorView pbuf, tvm::ffi::TensorView ctrl) {
    using namespace host;
    SymbolicSize M{"rows"};
    SymbolicDevice dev;
    TensorMatcher({M, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).verify(x).verify(out);
    TensorMatcher({M, TOPK}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(ids);
    TensorMatcher({M, TOPK}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(weights);
    TensorMatcher({512, KT1, ROW1 * 4}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(w13);
    TensorMatcher({512, KT2, ROW2 * 4}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(w2);
    TensorMatcher({512, KT1, 2 * I}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(dev).verify(s13);
    TensorMatcher({512, KT2, H}).with_dtype<fp8_e4m3_t>().with_device<kDLCUDA>(dev).verify(s2);
    TensorMatcher({512}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).verify(gs13).verify(gs2);
    TensorMatcher({MAXSLOT, MAXT, I}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).verify(hbuf);
    TensorMatcher({MAXT, TOPK, H}).with_dtype<bf16_t>().with_device<kDLCUDA>(dev).verify(ybuf);
    TensorMatcher({MAXSLOT, NG1, 2, 1024}).with_dtype<float>().with_device<kDLCUDA>(dev).verify(pbuf);
    TensorMatcher({CTRL}).with_dtype<int32_t>().with_device<kDLCUDA>(dev).verify(ctrl);
    CHECK_HOST(M.unwrap() >= 1 && M.unwrap() <= MAXT) << "persistent MoE requires 1..8 physical rows";
    CHECK_HOST(runtime::get_sm_version(dev.unwrap().device_id) == 120) << "persistent MoE requires SM120";
    const auto resident = runtime::get_blocks_per_sm(a16_moe_persist_kernel, THREADS);
    CHECK_HOST(resident > 0) << "persistent MoE kernel has no resident CTAs";
    const auto blocks = std::min(resident * runtime::get_sm_count(dev.unwrap().device_id), uint32_t(MAXSLOT * 40));
    LaunchKernel(1, THREADS, dev.unwrap())(initialize_moe,
        static_cast<unsigned*>(ctrl.data_ptr()), static_cast<bf16_t*>(out.data_ptr()), int(M.unwrap()));
    LaunchKernel(blocks, THREADS, dev.unwrap())(a16_moe_persist_kernel,
        static_cast<const bf16_t*>(x.data_ptr()), static_cast<const int*>(ids.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<const uint4*>(w13.data_ptr()), static_cast<const uint8_t*>(s13.data_ptr()),
        static_cast<const uint16_t*>(gs13.data_ptr()),
        static_cast<const uint4*>(w2.data_ptr()), static_cast<const uint8_t*>(s2.data_ptr()),
        static_cast<const uint16_t*>(gs2.data_ptr()),
        static_cast<bf16_t*>(hbuf.data_ptr()), static_cast<bf16_t*>(ybuf.data_ptr()),
        static_cast<float*>(pbuf.data_ptr()), static_cast<unsigned*>(ctrl.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()), int(M.unwrap()));
}
}  // namespace
