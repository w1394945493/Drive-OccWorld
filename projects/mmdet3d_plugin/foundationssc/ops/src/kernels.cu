// Local FP32 fused operators. All launches use the input device/current CUDA stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

using torch::Tensor;
constexpr int THREADS = 256;
static int blocks(int64_t n) { return static_cast<int>(std::min<int64_t>((n + THREADS - 1) / THREADS, 4096)); }

template<bool BACKWARD>
__global__ void pool_kernel(const float* context, const float* depth, const int64_t* idx,
                            float* out, const float* grad, float* gc, float* gd,
                            int64_t B, int64_t C, int64_t D, int64_t S, int64_t V) {
  int64_t count = B * D * S * C;
  for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < count; t += blockDim.x * (int64_t)gridDim.x) {
    int64_t channel = t % C, ray = t / C, b = ray / (D * S), s = ray % S;
    int64_t voxel = idx[ray];
    if (voxel < 0 || voxel >= V) continue;  // invalid geometry is never clamped into the volume
    int64_t ci = (b * C + channel) * S + s, oi = (b * C + channel) * V + voxel;
    if (BACKWARD) {
      float g = grad[oi];
      atomicAdd(gc + ci, g * depth[ray]);
      atomicAdd(gd + ray, g * context[ci]);
    } else {
      atomicAdd(out + oi, context[ci] * depth[ray]);
    }
  }
}

// One thread per output channel: sum over sample points and four/eight neighbours.
// It fuses interpolation and attention reduction instead of allocating [B,Q,heads,P,C].
template<bool DEPTH, bool BACKWARD>
__global__ void attn_kernel(const float* value, const float* depth, const float* loc, const float* weights,
                            float* out, const float* grad, float* gv, float* gd, float* gl, float* gw,
                            int64_t B, int64_t Q, int64_t M, int64_t C, int64_t H, int64_t W, int64_t D, int64_t P) {
  int64_t count = B * Q * M * C;
  constexpr int K = DEPTH ? 3 : 2;
  for (int64_t t = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; t < count; t += blockDim.x * (int64_t)gridDim.x) {
    int64_t channel = t % C, m = (t / C) % M, b = t / (C * M * Q);
    int64_t qhead = t / C;
    float result = 0.f, go = BACKWARD ? grad[t] : 0.f;
    for (int64_t p = 0; p < P; ++p) {
      int64_t wi = qhead * P + p, li = wi * K;
      float x = loc[li] * W - .5f, y = loc[li+1] * H - .5f;
      float z = DEPTH ? loc[li+2] * D - .5f : 0.f;
      if (!isfinite(x) || !isfinite(y) || !isfinite(z) || x < -1.f || x >= W || y < -1.f || y >= H || (DEPTH && (z < -1.f || z >= D))) continue;
      int64_t x0 = (int64_t)floorf(x), y0 = (int64_t)floorf(y), z0 = (int64_t)floorf(z);
      float fx = x-x0, fy = y-y0, fz = z-z0;
      float sample = 0.f, gx = 0.f, gy = 0.f, gz = 0.f;
      for (int dx = 0; dx < 2; ++dx) for (int dy = 0; dy < 2; ++dy) for (int dz = 0; dz < (DEPTH ? 2 : 1); ++dz) {
        int64_t xx = x0+dx, yy = y0+dy, zz = z0+dz;
        if (xx < 0 || xx >= W || yy < 0 || yy >= H || (DEPTH && (zz < 0 || zz >= D))) continue;
        float wx = dx ? fx : 1.f-fx, wy = dy ? fy : 1.f-fy, wz = DEPTH ? (dz ? fz : 1.f-fz) : 1.f;
        int64_t vi = ((b*M+m)*C+channel)*H*W + yy*W+xx;
        int64_t di = (b*D+zz)*H*W + yy*W+xx;
        float val = value[vi], dep = DEPTH ? depth[di] : 1.f;
        float coeff = wx*wy*wz, product = val*dep;
        sample += coeff*product;
        if (BACKWARD) {
          float g = go*weights[wi];
          atomicAdd(gv+vi, g*coeff*dep);
          if (DEPTH) atomicAdd(gd+di, g*coeff*val);
          gx += (dx ? 1.f : -1.f)*wy*wz*product;
          gy += wx*(dy ? 1.f : -1.f)*wz*product;
          if (DEPTH) gz += wx*wy*(dz ? 1.f : -1.f)*product;
        }
      }
      if (BACKWARD) {
        atomicAdd(gw+wi, go*sample);
        float g = go*weights[wi];
        atomicAdd(gl+li, g*gx*W);
        atomicAdd(gl+li+1, g*gy*H);
        if (DEPTH) atomicAdd(gl+li+2, g*gz*D);
      } else result += sample*weights[wi];
    }
    if (!BACKWARD) out[t] = result;
  }
}

Tensor pool_forward_cuda(Tensor c, Tensor d, Tensor idx, int64_t V) {
  c10::cuda::CUDAGuard guard(c.device());
  auto out = torch::zeros({c.size(0), c.size(1), V}, c.options());
  int64_t n = d.numel()*c.size(1);
  pool_kernel<false><<<blocks(n), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      c.data_ptr<float>(), d.data_ptr<float>(), idx.data_ptr<int64_t>(), out.data_ptr<float>(), nullptr, nullptr, nullptr,
      c.size(0), c.size(1), d.size(1), c.size(2), V);
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return out;
}
std::vector<Tensor> pool_backward_cuda(Tensor g, Tensor c, Tensor d, Tensor idx) {
  c10::cuda::CUDAGuard guard(c.device());
  auto gc = torch::zeros_like(c), gd = torch::zeros_like(d);
  int64_t n = d.numel()*c.size(1);
  pool_kernel<true><<<blocks(n), THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
      c.data_ptr<float>(), d.data_ptr<float>(), idx.data_ptr<int64_t>(), nullptr, g.data_ptr<float>(), gc.data_ptr<float>(), gd.data_ptr<float>(),
      c.size(0), c.size(1), d.size(1), c.size(2), g.size(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK(); return {gc,gd};
}

template<bool BACKWARD>
void launch_attn(Tensor v, Tensor d, Tensor l, Tensor w, Tensor out, Tensor g, Tensor gv, Tensor gd, Tensor gl, Tensor gw) {
  int64_t B=v.size(0), M=v.size(1), C=v.size(2), H=v.size(3), W=v.size(4), Q=l.size(1), P=l.size(3);
  int64_t n=B*Q*M*C;
  if (!n) return;  // empty query set, including backward
  auto stream=at::cuda::getCurrentCUDAStream();
  float *op=BACKWARD?nullptr:out.data_ptr<float>(), *gvp=BACKWARD?gv.data_ptr<float>():nullptr;
  float *glp=BACKWARD?gl.data_ptr<float>():nullptr, *gwp=BACKWARD?gw.data_ptr<float>():nullptr;
  const float* gp=BACKWARD?g.data_ptr<float>():nullptr;
  if (l.size(4)==3) {
    attn_kernel<true,BACKWARD><<<blocks(n),THREADS,0,stream>>>(v.data_ptr<float>(),d.data_ptr<float>(),l.data_ptr<float>(),w.data_ptr<float>(),
        op,gp,gvp,BACKWARD?gd.data_ptr<float>():nullptr,glp,gwp,B,Q,M,C,H,W,d.size(1),P);
  } else {
    attn_kernel<false,BACKWARD><<<blocks(n),THREADS,0,stream>>>(v.data_ptr<float>(),nullptr,l.data_ptr<float>(),w.data_ptr<float>(),
        op,gp,gvp,nullptr,glp,gwp,B,Q,M,C,H,W,1,P);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
Tensor attn_forward_cuda(Tensor v, Tensor d, Tensor l, Tensor w) {
  c10::cuda::CUDAGuard guard(v.device());
  auto out=torch::zeros({v.size(0),l.size(1),v.size(1),v.size(2)},v.options());
  launch_attn<false>(v,d,l,w,out,Tensor(),Tensor(),Tensor(),Tensor(),Tensor()); return out;
}
std::vector<Tensor> attn_backward_cuda(Tensor g, Tensor v, Tensor d, Tensor l, Tensor w) {
  c10::cuda::CUDAGuard guard(v.device());
  auto gv=torch::zeros_like(v), gd=torch::zeros_like(d), gl=torch::zeros_like(l), gw=torch::zeros_like(w);
  launch_attn<true>(v,d,l,w,Tensor(),g,gv,gd,gl,gw); return {gv,gd,gl,gw};
}
