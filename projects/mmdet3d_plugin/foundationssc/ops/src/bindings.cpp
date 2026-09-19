#include <torch/extension.h>
#include <vector>

using torch::Tensor;
Tensor pool_forward_cuda(Tensor, Tensor, Tensor, int64_t);
std::vector<Tensor> pool_backward_cuda(Tensor, Tensor, Tensor, Tensor);
Tensor attn_forward_cuda(Tensor, Tensor, Tensor, Tensor);
std::vector<Tensor> attn_backward_cuda(Tensor, Tensor, Tensor, Tensor, Tensor);

static void check_float(const Tensor& t, const Tensor& ref) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "expected contiguous CUDA tensor");
  TORCH_CHECK(t.scalar_type() == at::kFloat, "only float32 is supported");
  TORCH_CHECK(t.device() == ref.device(), "input devices must match");
}
static void check_pool(const Tensor& c, const Tensor& d, const Tensor& idx) {
  check_float(c, c); check_float(d, c);
  TORCH_CHECK(c.dim() == 3 && d.dim() == 3, "expected context[B,C,S], depth[B,D,S]");
  TORCH_CHECK(c.size(0) == d.size(0) && c.size(2) == d.size(2), "pool shapes disagree");
  TORCH_CHECK(c.size(0) > 0 && c.size(1) > 0 && c.size(2) > 0 && d.size(1) > 0, "empty pool input dimensions");
  TORCH_CHECK(idx.is_cuda() && idx.is_contiguous() && idx.device() == c.device(), "invalid index device/layout");
  TORCH_CHECK(idx.scalar_type() == at::kLong && idx.sizes() == d.sizes(), "indices must be int64[B,D,S]");
}
static void check_attn(const Tensor& v, const Tensor& d, const Tensor& l, const Tensor& w) {
  check_float(v, v); check_float(d, v); check_float(l, v); check_float(w, v);
  TORCH_CHECK(v.dim() == 5 && l.dim() == 5 && w.dim() == 4, "invalid attention dimensions");
  TORCH_CHECK(v.size(0) > 0 && v.size(1) > 0 && v.size(2) > 0 && v.size(3) > 0 && v.size(4) > 0, "empty value dimensions");
  TORCH_CHECK(l.size(0) == v.size(0) && l.size(2) == v.size(1), "batch/head mismatch");
  TORCH_CHECK(l.size(3) > 0 && (l.size(4) == 2 || l.size(4) == 3), "locations must end in 2 or 3");
  for (int i = 0; i < 4; ++i) TORCH_CHECK(w.size(i) == l.size(i), "attention weight shape mismatch");
  if (l.size(4) == 3) {
    TORCH_CHECK(d.dim() == 4 && d.size(0) == v.size(0) && d.size(1) > 0 &&
                d.size(2) == v.size(3) && d.size(3) == v.size(4), "depth must be [B,D,H,W]");
  } else {
    TORCH_CHECK(d.numel() == 0, "2D attention takes an empty depth tensor");
  }
}
Tensor pool_forward(Tensor c, Tensor d, Tensor idx, int64_t voxels) {
  check_pool(c,d,idx); TORCH_CHECK(voxels > 0, "voxel_count must be positive");
  return pool_forward_cuda(c,d,idx,voxels);
}
std::vector<Tensor> pool_backward(Tensor g, Tensor c, Tensor d, Tensor idx) {
  check_pool(c,d,idx); check_float(g,c);
  TORCH_CHECK(g.dim() == 3 && g.size(0) == c.size(0) && g.size(1) == c.size(1) && g.size(2) > 0, "invalid pool grad");
  return pool_backward_cuda(g,c,d,idx);
}
Tensor attn_forward(Tensor v, Tensor d, Tensor l, Tensor w) {
  check_attn(v,d,l,w); return attn_forward_cuda(v,d,l,w);
}
std::vector<Tensor> attn_backward(Tensor g, Tensor v, Tensor d, Tensor l, Tensor w) {
  check_attn(v,d,l,w); check_float(g,v);
  TORCH_CHECK(g.dim() == 4 && g.size(0) == v.size(0) && g.size(1) == l.size(1) &&
              g.size(2) == v.size(1) && g.size(3) == v.size(2), "invalid attention grad");
  return attn_backward_cuda(g,v,d,l,w);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pool_forward", &pool_forward);
  m.def("pool_backward", &pool_backward);
  m.def("attn_forward", &attn_forward);
  m.def("attn_backward", &attn_backward);
}
