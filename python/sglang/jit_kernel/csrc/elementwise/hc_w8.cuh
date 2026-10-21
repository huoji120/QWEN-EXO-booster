// Adapted from SSHdotCodes/qwen-3.8-flash-next-pro6000 qf_hc8.cu at
// 1233bf628f42940ec8e4fb060fa35c1a30eecfc2. Modified for SGLang JIT tensor
// wrappers, model-bound dispatch, and per-call workspace ownership.
#include <sgl_kernel/runtime.cuh>  // For resident SM count
#include <sgl_kernel/tensor.h>  // For TensorMatcher and symbolic dimensions
#include <sgl_kernel/utils.h>  // For CHECK_HOST
#include <sgl_kernel/utils.cuh>  // For LaunchKernel and CUDA type aliases
#include <sgl_kernel/vec.cuh>  // For aligned BF16 vector loads and stores

#include <cstdint>
#include <tvm/ffi/container/tensor.h>

namespace sglang {
namespace {
constexpr int HC=4, HS=2560, K=HC*HS, R=320, NINJ=4, NROWS=R+NINJ;
constexpr int THREADS=256, WARPS=THREADS/32, MAXM=8;
constexpr int KQ_VEC=HS/8, GD=128, GU=64, WD_NG=K/GD, WU_NG=R/GU;
constexpr int WU8_VEC=R/16, WU8_VEC_PER_J=HC*WU8_VEC, MAXJ=16;

__device__ __forceinline__ void bf16x8_to_f32(const uint4& v,float(&f)[8]) {
  const __nv_bfloat162* h=reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for(int i=0;i<4;++i){ const float2 t=__bfloat1622float2(h[i]); f[2*i]=t.x; f[2*i+1]=t.y; }
}
__device__ __forceinline__ uint4 i8x8_to_bf16(uint32_t lo,uint32_t hi,float sc) {
  float f[8]; lo^=0x80808080u; hi^=0x80808080u;
#pragma unroll
  for(int i=0;i<4;++i){ f[i]=__int_as_float(__byte_perm(lo,0x4B000000u,0x7440+i))-8388736.f; f[4+i]=__int_as_float(__byte_perm(hi,0x4B000000u,0x7440+i))-8388736.f; }
  uint4 o; __nv_bfloat162* op=reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
  for(int i=0;i<4;++i) op[i]=__floats2bfloat162_rn(f[2*i]*sc,f[2*i+1]*sc);
  return o;
}
__device__ __forceinline__ float bf16_bits(uint32_t b){ return __uint_as_float(b<<16); }
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p){ unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(v):"l"(p):"memory"); return v; }
__device__ __forceinline__ void mma16816(float(&d)[4],uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};":"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
constexpr int XS_STRIDE=HS+32, WU8_STRIDE=WU8_VEC+1, TS_STRIDE=R+32;
template<int M> constexpr int xs_elems(){return (M*XS_STRIDE>8*TS_STRIDE)?M*XS_STRIDE:8*TS_STRIDE;}

// Standalone combine retains the materialized BF16 boundary consumed by
// PLE/EXO. Match eager FP32 sigmoid, multiply, then add (not an FMA).
__global__ void hc_combine_w8_kernel(
    const bf16_t* __restrict__ block_output,
    const bf16_t* __restrict__ residual,
    const float* __restrict__ inject,
    bf16_t* __restrict__ out) {
  constexpr int kVec = 8;
  const int row = blockIdx.z;
  const int branch = blockIdx.y;
  const int vec = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec >= HS / kVec) return;

  float gate = 0.0f;
  if ((threadIdx.x & 31) == 0) {
    const float scaled = __fmul_rn(inject[row * HC + branch], 1.0f / HC);
    const float sigmoid = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-scaled)));
    gate = __fmul_rn(2.0f, sigmoid);
  }
  gate = __shfl_sync(0xffffffff, gate, 0);

  device::AlignedVector<bf16_t, kVec> y, x, combined;
  const int offset = vec * kVec;
  y.load(block_output + row * HS + offset);
  x.load(residual + row * K + branch * HS + offset);
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    const float injection = __fmul_rn(static_cast<float>(y[i]), gate);
    combined[i] = __float2bfloat16_rn(__fadd_rn(static_cast<float>(x[i]), injection));
  }
  combined.store(out + row * K + branch * HS + offset);
}

template<int M,bool COMB>
__global__ __launch_bounds__(THREADS,1) void hc_mix_fused_w8_kernel(
 const __nv_bfloat16* x,const __nv_bfloat16* w_norm,const int8_t* wd,const uint16_t* wd_s,const __nv_bfloat16* winj,
 const int8_t* wu,const uint16_t* wu_s,__nv_bfloat16* out,float* inj_out,__nv_bfloat16* normed,float* partial,unsigned* bar,
 int rows,float eps,const __nv_bfloat16* yb,const float* inj_prev,__nv_bfloat16* xout){
 extern __shared__ __align__(16) uint4 smem[];
 uint4* wus=smem;
 uint16_t* wuss=reinterpret_cast<uint16_t*>(smem+MAXJ*HC*WU8_STRIDE);
 __nv_bfloat16* xs=reinterpret_cast<__nv_bfloat16*>(smem+MAXJ*HC*WU8_STRIDE+48);
 __shared__ float s_red[WARPS][M]; __shared__ float s_rstd[M]; __shared__ float s_c[WARPS][16][8];
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,grp=lane>>2,tq=lane&3;
 const int G=gridDim.x,cta=blockIdx.x,nrows=winj!=nullptr?NROWS:R;
 const int per_q=G/HC,q=cta/per_q,qi=cta%per_q,n_row=qi+grp*per_q;
 const bool has_row=grp<7&&n_row<nrows,row8=n_row<R;
 uint4 wa[10];
 {
  const int kofs=q*HS+warp*320+tq*8;
  if(has_row&&row8){ const int8_t* wrow=wd+(size_t)n_row*K+kofs;
#pragma unroll
   for(int kc=0;kc<10;++kc){ const uint2 v=__ldcs(reinterpret_cast<const uint2*>(wrow+kc*32)); const uint32_t sb=__ldg(wd_s+(size_t)n_row*WD_NG+(q*HS+warp*320+kc*32)/GD); wa[kc]=make_uint4(v.x,v.y,sb,0u); }
  } else if(has_row){ const __nv_bfloat16* wrow=winj+(size_t)(n_row-R)*K+kofs;
#pragma unroll
   for(int kc=0;kc<10;++kc) wa[kc]=__ldcs(reinterpret_cast<const uint4*>(wrow+kc*32));
  } else {
#pragma unroll
   for(int kc=0;kc<10;++kc) wa[kc]=make_uint4(0,0,0,0);
  }
 }
 const int j0=(int)((long)cta*HS/G),j1=(int)((long)(cta+1)*HS/G),nj=j1-j0;
 for(int c=tid;c<nj*WU8_VEC_PER_J;c+=THREADS){ const int jl=c/WU8_VEC_PER_J,v=c%WU8_VEC_PER_J,g=v/WU8_VEC,k=v%WU8_VEC; const void* src=wu+((size_t)(g*HS+j0+jl))*R+k*16; const unsigned dst=(unsigned)__cvta_generic_to_shared(wus+(jl*HC+g)*WU8_STRIDE+k); asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(dst),"l"(src):"memory"); }
 asm volatile("cp.async.commit_group;":: :"memory");
 for(int c=tid;c<nj*HC*WU_NG;c+=THREADS){ const int rr=c/WU_NG,gg=c%WU_NG,jl=rr/HC,g=rr%HC; wuss[rr*WU_NG+gg]=__ldg(wu_s+((size_t)(g*HS+j0+jl))*WU_NG+gg); }
 uint4 wn[2]; wn[0]=__ldg(reinterpret_cast<const uint4*>(w_norm+q*HS)+tid); wn[1]=tid<KQ_VEC-THREADS?__ldg(reinterpret_cast<const uint4*>(w_norm+q*HS)+tid+THREADS):make_uint4(0,0,0,0);
 asm volatile("griddepcontrol.wait;":: :"memory");
 {
  uint4 xv[M][2]; float ss[M];
#pragma unroll
  for(int m=0;m<M;++m){ const uint4* src=reinterpret_cast<const uint4*>(x+(size_t)m*K+q*HS); xv[m][0]=m<rows?__ldcg(src+tid):make_uint4(0,0,0,0); xv[m][1]=(m<rows&&tid<KQ_VEC-THREADS)?__ldcg(src+tid+THREADS):make_uint4(0,0,0,0); }
  if constexpr(COMB){ uint4 yv[M][2]; float ip[M];
#pragma unroll
   for(int m=0;m<M;++m){ const uint4* ys=reinterpret_cast<const uint4*>(yb+(size_t)m*HS); ip[m]=m<rows?__ldcg(inj_prev+m*NINJ+q):0.f; yv[m][0]=m<rows?__ldcg(ys+tid):make_uint4(0,0,0,0); yv[m][1]=(m<rows&&tid<KQ_VEC-THREADS)?__ldcg(ys+tid+THREADS):make_uint4(0,0,0,0); }
#pragma unroll
   for(int m=0;m<M;++m){ const float a=2.f/(1.f+expf(-ip[m]/HC));
#pragma unroll
    for(int h=0;h<2;++h){ const int kk=tid+h*THREADS; float rf[8],yf[8]; bf16x8_to_f32(xv[m][h],rf); bf16x8_to_f32(yv[m][h],yf); uint4 o; __nv_bfloat162* op=reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
     for(int e=0;e<4;++e) op[e]=__floats2bfloat162_rn(rf[2*e]+a*yf[2*e],rf[2*e+1]+a*yf[2*e+1]); xv[m][h]=o; if(m<rows&&kk<KQ_VEC&&kk%per_q==qi) reinterpret_cast<uint4*>(xout+(size_t)m*K+q*HS)[kk]=o; }
   }
  }
#pragma unroll
  for(int m=0;m<M;++m){ float s=0.f;
#pragma unroll
   for(int h=0;h<2;++h){float f[8];bf16x8_to_f32(xv[m][h],f);
#pragma unroll
    for(int e=0;e<8;++e)s+=f[e]*f[e];}
#pragma unroll
   for(int o=16;o>0;o>>=1)s+=__shfl_xor_sync(0xffffffff,s,o); ss[m]=s; }
  if(lane==0){
#pragma unroll
   for(int m=0;m<M;++m)s_red[warp][m]=ss[m]; }
  __syncthreads(); if(tid<M){float s=0.f;
#pragma unroll
   for(int w=0;w<WARPS;++w)s+=s_red[w][tid]; s_rstd[tid]=rsqrtf(s/HS+eps);}
  __syncthreads();
#pragma unroll
  for(int h=0;h<2;++h){const int kk=tid+h*THREADS;if(kk>=KQ_VEC)break;float wf[8];bf16x8_to_f32(wn[h],wf);
#pragma unroll
   for(int m=0;m<M;++m){float f[8];bf16x8_to_f32(xv[m][h],f);const float rr=s_rstd[m];uint4 o;__nv_bfloat162* op=reinterpret_cast<__nv_bfloat162*>(&o);
#pragma unroll
    for(int e=0;e<4;++e)op[e]=__floats2bfloat162_rn(f[2*e]*rr*(1.f+wf[2*e]),f[2*e+1]*rr*(1.f+wf[2*e+1]));reinterpret_cast<uint4*>(xs+m*XS_STRIDE)[kk]=o;if(kk%per_q==qi&&m<rows)reinterpret_cast<uint4*>(normed+(size_t)m*K+q*HS)[kk]=o;}}
  __syncthreads();
 }
 {
  if(has_row&&row8){
#pragma unroll
   for(int kc=0;kc<10;++kc)wa[kc]=i8x8_to_bf16(wa[kc].x,wa[kc].y,bf16_bits(wa[kc].z));}
  float d[4]={0,0,0,0};const bool brow=grp<M;const __nv_bfloat16* xb=xs+(brow?grp:0)*XS_STRIDE+warp*320+tq*8;
#pragma unroll
  for(int kc=0;kc<10;++kc){const uint4 bv=brow?*reinterpret_cast<const uint4*>(xb+kc*32):make_uint4(0,0,0,0);mma16816(d,wa[kc].x,0,wa[kc].y,0,bv.x,bv.y);mma16816(d,wa[kc].z,0,wa[kc].w,0,bv.z,bv.w);}
  s_c[warp][grp][2*tq]=d[0];s_c[warp][grp][2*tq+1]=d[1];__syncthreads();
  if(tid<7*M){const int rr=tid/M,m=tid%M,n=qi+rr*per_q;if(n<nrows){float v=0.f;
#pragma unroll
   for(int w=0;w<WARPS;++w)v+=s_c[w][rr][m];partial[((size_t)q*MAXM+m)*NROWS+n]=v;}}
 }
 __syncthreads();if(tid==0){__threadfence();atomicAdd(bar,1u);while(ld_acquire(bar)<(unsigned)G){}}
 __syncthreads();asm volatile("griddepcontrol.launch_dependents;":: :"memory");
 float nv[2][2]={{0,0},{0,0}};if(warp<4&&warp*4<nj){
#pragma unroll
  for(int h=0;h<2;++h){const int rr=warp*16+grp+8*h,jl=rr>>2,g=rr&3;
#pragma unroll
   for(int c=0;c<2;++c){const int m=2*tq+c;if(rr<nj*HC&&m<rows)nv[h][c]=__bfloat162float(__ldcg(normed+(size_t)m*K+g*HS+j0+jl));}}}
 __nv_bfloat16* ts=xs;constexpr int TV=(8*R+THREADS-1)/THREADS;float pv[TV][HC];
#pragma unroll
 for(int i=0;i<TV;++i){const int c=tid+i*THREADS,m=c/R,r=c%R;
#pragma unroll
  for(int g=0;g<HC;++g)pv[i][g]=(c<8*R&&m<M)?__ldcg(partial+((size_t)g*MAXM+m)*NROWS+r):0.f;}
 for(int i=0;i<TV;++i){const int c=tid+i*THREADS,m=c/R,r=c%R;float a=(pv[i][0]+pv[i][1])+(pv[i][2]+pv[i][3]);a*=1.f/HC;if(c<8*R)ts[m*TS_STRIDE+r]=m<M?__float2bfloat16(a*(1.f/(1.f+__expf(-a)))):__float2bfloat16(0.f);}
 if(cta==0&&winj!=nullptr&&tid<M*NINJ){const int m=tid/NINJ,g=tid%NINJ;if(m<rows){float a=0.f;
#pragma unroll
  for(int qq=0;qq<HC;++qq)a+=__ldcg(partial+((size_t)qq*MAXM+m)*NROWS+R+g);inj_out[m*NINJ+g]=a;}}
 asm volatile("cp.async.wait_all;":: :"memory");__syncthreads();
 if(warp<4&&warp*4<nj){float d[4]={0,0,0,0};const int r0=warp*16+grp,r1=r0+8;const bool v0=r0<nj*HC,v1=r1<nj*HC;const uint8_t* a0p=reinterpret_cast<const uint8_t*>(wus+r0*WU8_STRIDE)+tq*8;const uint8_t* a1p=reinterpret_cast<const uint8_t*>(wus+r1*WU8_STRIDE)+tq*8;const __nv_bfloat16* bp=ts+grp*TS_STRIDE+tq*8;
#pragma unroll
  for(int kc=0;kc<R/32;++kc){uint4 av0=make_uint4(0,0,0,0),av1=make_uint4(0,0,0,0);if(v0){const uint2 w=*reinterpret_cast<const uint2*>(a0p+kc*32);av0=i8x8_to_bf16(w.x,w.y,bf16_bits(wuss[r0*WU_NG+kc*32/GU]));}if(v1){const uint2 w=*reinterpret_cast<const uint2*>(a1p+kc*32);av1=i8x8_to_bf16(w.x,w.y,bf16_bits(wuss[r1*WU_NG+kc*32/GU]));}const uint4 bv=*reinterpret_cast<const uint4*>(bp+kc*32);mma16816(d,av0.x,av1.x,av0.y,av1.y,bv.x,bv.y);mma16816(d,av0.z,av1.z,av0.w,av1.w,bv.z,bv.w);}
#pragma unroll
  for(int h=0;h<2;++h){const int rr=h?r1:r0,jl=rr>>2,g=rr&3,j=j0+jl;const bool valid=h?v1:v0;float o[2];
#pragma unroll
   for(int c=0;c<2;++c){const int m=2*tq+c;float v=0.f;if(valid&&m<rows){const float gate=1.f/(1.f+__expf(-d[2*h+c]));v=gate*nv[h][c];}v+=__shfl_xor_sync(0xffffffff,v,4);v+=__shfl_xor_sync(0xffffffff,v,8);o[c]=v;}if(valid&&g==0){
#pragma unroll
    for(int c=0;c<2;++c){const int m=2*tq+c;if(m<rows)out[(size_t)m*HS+j]=__float2bfloat16(o[c]*(1.f/HC));}}}}
 if(tid==0){const unsigned ticket=atomicAdd(bar+1,1u);if(ticket==(unsigned)G-1){bar[0]=0;bar[1]=0;__threadfence();}}
}

template<int M,bool COMB> void launch_mix(const tvm::ffi::TensorView& x,const tvm::ffi::TensorView& wn,const tvm::ffi::TensorView& wd,const tvm::ffi::TensorView& wds,const tvm::ffi::TensorView& wi,const tvm::ffi::TensorView& wu,const tvm::ffi::TensorView& wus,const tvm::ffi::TensorView& out,const tvm::ffi::TensorView& inj,const tvm::ffi::TensorView& normed,const tvm::ffi::TensorView& partial,const tvm::ffi::TensorView& bar,int rows,float eps,const tvm::ffi::TensorView& y,const tvm::ffi::TensorView& ip,const tvm::ffi::TensorView& xo,DLDevice dev,bool pdl){
 const int smem=(MAXJ*HC*WU8_STRIDE+48)*16+xs_elems<M>()*2;auto k=hc_mix_fused_w8_kernel<M,COMB>;if(smem>48*1024)host::RuntimeDeviceCheck(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));const int sms=host::runtime::get_sm_count(dev.device_id);const int grid=(sms/HC)*HC;CHECK_HOST(grid*MAXJ>=HS&&(grid/HC)*7>=NROWS)<<"too few SMs for hc_mix_w8";host::LaunchKernel(grid,THREADS,dev,smem).enable_pdl(pdl)(k,static_cast<const __nv_bfloat16*>(x.data_ptr()),static_cast<const __nv_bfloat16*>(wn.data_ptr()),static_cast<const int8_t*>(wd.data_ptr()),static_cast<const uint16_t*>(wds.data_ptr()),wi.size(0)?static_cast<const __nv_bfloat16*>(wi.data_ptr()):nullptr,static_cast<const int8_t*>(wu.data_ptr()),static_cast<const uint16_t*>(wus.data_ptr()),static_cast<__nv_bfloat16*>(out.data_ptr()),static_cast<float*>(inj.data_ptr()),static_cast<__nv_bfloat16*>(normed.data_ptr()),static_cast<float*>(partial.data_ptr()),static_cast<unsigned*>(bar.data_ptr()),rows,eps,COMB?static_cast<const __nv_bfloat16*>(y.data_ptr()):nullptr,COMB?static_cast<const float*>(ip.data_ptr()):nullptr,COMB?static_cast<__nv_bfloat16*>(xo.data_ptr()):nullptr);
}

template<bool COMB> void dispatch(const tvm::ffi::TensorView& x,const tvm::ffi::TensorView& wn,const tvm::ffi::TensorView& wd,const tvm::ffi::TensorView& wds,const tvm::ffi::TensorView& wi,const tvm::ffi::TensorView& wu,const tvm::ffi::TensorView& wus,const tvm::ffi::TensorView& out,const tvm::ffi::TensorView& inj,const tvm::ffi::TensorView& normed,const tvm::ffi::TensorView& partial,const tvm::ffi::TensorView& bar,int rows,float eps,const tvm::ffi::TensorView& y,const tvm::ffi::TensorView& ip,const tvm::ffi::TensorView& xo,DLDevice dev,bool pdl){if(rows==1)launch_mix<1,COMB>(x,wn,wd,wds,wi,wu,wus,out,inj,normed,partial,bar,rows,eps,y,ip,xo,dev,pdl);else if(rows==2)launch_mix<2,COMB>(x,wn,wd,wds,wi,wu,wus,out,inj,normed,partial,bar,rows,eps,y,ip,xo,dev,pdl);else if(rows<=4)launch_mix<4,COMB>(x,wn,wd,wds,wi,wu,wus,out,inj,normed,partial,bar,rows,eps,y,ip,xo,dev,pdl);else launch_mix<8,COMB>(x,wn,wd,wds,wi,wu,wus,out,inj,normed,partial,bar,rows,eps,y,ip,xo,dev,pdl);}

template<bool COMB> void run_impl(const tvm::ffi::TensorView& x,const tvm::ffi::TensorView& wn,const tvm::ffi::TensorView& wd,const tvm::ffi::TensorView& wds,const tvm::ffi::TensorView& wi,const tvm::ffi::TensorView& wu,const tvm::ffi::TensorView& wus,double eps,const tvm::ffi::TensorView& out,const tvm::ffi::TensorView& inj,const tvm::ffi::TensorView& normed,const tvm::ffi::TensorView& partial,const tvm::ffi::TensorView& bar,const tvm::ffi::TensorView& y,const tvm::ffi::TensorView& ip,const tvm::ffi::TensorView& xo,bool pdl){using namespace host;auto M=SymbolicSize{"rows"};auto dev=SymbolicDevice{};dev.set_options<kDLCUDA>();TensorMatcher({M,K}).with_dtype<bf16_t>().with_device(dev).verify(x);TensorMatcher({K}).with_dtype<bf16_t>().with_device(dev).verify(wn);TensorMatcher({R,K}).with_dtype<int8_t>().with_device(dev).verify(wd);TensorMatcher({R,WD_NG}).with_dtype<bf16_t>().with_device(dev).verify(wds);TensorMatcher({K,R}).with_dtype<int8_t>().with_device(dev).verify(wu);TensorMatcher({K,WU_NG}).with_dtype<bf16_t>().with_device(dev).verify(wus);if(wi.size(0))TensorMatcher({HC,K}).with_dtype<bf16_t>().with_device(dev).verify(wi);TensorMatcher({M,HS}).with_dtype<bf16_t>().with_device(dev).verify(out);TensorMatcher({M,NINJ}).with_dtype<float>().with_device(dev).verify(inj);TensorMatcher({M,K}).with_dtype<bf16_t>().with_device(dev).verify(normed);TensorMatcher({HC*MAXM*NROWS}).with_dtype<float>().with_device(dev).verify(partial);TensorMatcher({2}).with_dtype<int32_t>().with_device(dev).verify(bar);if constexpr(COMB){TensorMatcher({M,HS}).with_dtype<bf16_t>().with_device(dev).verify(y);TensorMatcher({M,NINJ}).with_dtype<float>().with_device(dev).verify(ip);TensorMatcher({M,K}).with_dtype<bf16_t>().with_device(dev).verify(xo);}CHECK_HOST(M.unwrap()>0&&M.unwrap()<=MAXM)<<"hc_mix_w8 supports 1..8 rows";CHECK_HOST(M.unwrap()<=runtime::get_sm_count(dev.unwrap().device_id))<<"HC grid must be resident";dispatch<COMB>(x,wn,wd,wds,wi,wu,wus,out,inj,normed,partial,bar,(int)M.unwrap(),(float)eps,y,ip,xo,dev.unwrap(),pdl);}
}
struct HcW8Kernel{static void run(const tvm::ffi::TensorView x,const tvm::ffi::TensorView wn,const tvm::ffi::TensorView wd,const tvm::ffi::TensorView wds,const tvm::ffi::TensorView wi,const tvm::ffi::TensorView wu,const tvm::ffi::TensorView wus,double eps,const tvm::ffi::TensorView out,const tvm::ffi::TensorView inj,const tvm::ffi::TensorView normed,const tvm::ffi::TensorView partial,const tvm::ffi::TensorView bar,bool pdl){tvm::ffi::TensorView z=x;run_impl<false>(x,wn,wd,wds,wi,wu,wus,eps,out,inj,normed,partial,bar,z,z,z,pdl);}};
struct HcCombineMixW8Kernel{static void run(const tvm::ffi::TensorView y,const tvm::ffi::TensorView x,const tvm::ffi::TensorView ip,const tvm::ffi::TensorView wn,const tvm::ffi::TensorView wd,const tvm::ffi::TensorView wds,const tvm::ffi::TensorView wi,const tvm::ffi::TensorView wu,const tvm::ffi::TensorView wus,double eps,const tvm::ffi::TensorView out,const tvm::ffi::TensorView inj,const tvm::ffi::TensorView normed,const tvm::ffi::TensorView partial,const tvm::ffi::TensorView bar,const tvm::ffi::TensorView xo,bool pdl){run_impl<true>(x,wn,wd,wds,wi,wu,wus,eps,out,inj,normed,partial,bar,y,ip,xo,pdl);}};

struct HcCombineW8Kernel {
  static void run(
      tvm::ffi::TensorView block_output,
      tvm::ffi::TensorView residual,
      tvm::ffi::TensorView inject,
      tvm::ffi::TensorView out) {
    using namespace host;
    auto rows = SymbolicSize{"rows"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();
    TensorMatcher({rows, K})
        .with_dtype<bf16_t>()
        .with_device(dev)
        .verify(residual)
        .verify(out);
    TensorMatcher({rows, HS}).with_dtype<bf16_t>().with_device(dev).verify(block_output);
    TensorMatcher({rows, HC}).with_dtype<float>().with_device(dev).verify(inject);
    CHECK_HOST(rows.unwrap() >= 1 && rows.unwrap() <= MAXM)
        << "hc_combine_w8 supports only 1..8 decode rows";
    CHECK_HOST(
        (reinterpret_cast<uintptr_t>(block_output.data_ptr()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0))
        << "hc_combine_w8 requires 16-byte-aligned BF16 tensors";
    constexpr int kThreads = 128;
    const dim3 grid(div_ceil(HS / 8, kThreads), HC, rows.unwrap());
    LaunchKernel(grid, kThreads, dev.unwrap())(
        hc_combine_w8_kernel,
        static_cast<const bf16_t*>(block_output.data_ptr()),
        static_cast<const bf16_t*>(residual.data_ptr()),
        static_cast<const float*>(inject.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()));
  }
};
} // namespace sglang
