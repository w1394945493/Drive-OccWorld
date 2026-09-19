"""三类融合算子的 autograd 包装，当前 FP32，支持一阶反向。"""
import importlib
import torch
from torch.autograd.function import once_differentiable

_extension = None


def require_extension():
    global _extension
    if _extension is None:
        try:
            _extension = importlib.import_module('._C', __package__)
        except ImportError as exc:
            raise ImportError(
                'FoundationSSC CUDA 扩展未编译或与当前 PyTorch/CUDA 不兼容。请在 '
                'projects/mmdet3d_plugin/foundationssc/ops 下执行 '
                '`python setup.py build_ext --inplace`。不会静默回退到 PyTorch。') from exc
    return _extension


def use_cuda(backend, tensor):
    if backend not in ('pytorch', 'cuda', 'auto'):
        raise ValueError(f'不支持 ops_backend={backend}')
    if backend == 'cuda' and not tensor.is_cuda:
        raise ValueError('ops_backend=cuda 要求 CUDA tensor；CPU 测试请选 pytorch')
    enabled = backend == 'cuda' or (backend == 'auto' and tensor.is_cuda)
    if enabled:
        require_extension()  #* auto 仅按设备选择；GPU 上缺少扩展仍明确报错。
    return enabled


def _check_float(*tensors):
    if any(not x.is_cuda or x.dtype != torch.float32 for x in tensors):
        raise ValueError('本地 CUDA 算子要求 FP32 CUDA tensor；当前 voxel 前端已采用 FP32')
    if any(x.device != tensors[0].device for x in tensors):
        raise ValueError('算子输入必须在同一设备')


class _LiftPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, context, depth, indices, voxel_count):
        _check_float(context, depth)
        context, depth, indices = context.contiguous(), depth.contiguous(), indices.contiguous()
        ctx.save_for_backward(context, depth, indices)
        return require_extension().pool_forward(context, depth, indices, voxel_count)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        context, depth, indices = ctx.saved_tensors
        gc, gd = require_extension().pool_backward(grad.contiguous(), context, depth, indices)
        return gc, gd, None, None


def lift_pool(context, depth, indices, voxel_count):
    """context[B,C,S], depth[B,D,S], local voxel indices[B,D,S]（-1=无效）。

    输出[B,C,V]；乘深度概率和 voxel 求和融合，不构造[B,C,D,S]中间张量。
    标定/离散 indices 不求导，与 PyTorch floor/index_add 路径一致。
    """
    return _LiftPool.apply(context, depth, indices, voxel_count)


class _DeformAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, depth, locations, weights):
        _check_float(value, depth, locations, weights)
        tensors = tuple(x.contiguous() for x in (value, depth, locations, weights))
        ctx.save_for_backward(*tensors)
        return require_extension().attn_forward(*tensors)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        return tuple(require_extension().attn_backward(grad.contiguous(), *ctx.saved_tensors))


def deform_attention(value, locations, weights, depth=None):
    """value[B,heads,C,H,W], locations[B,Q,heads,P,2或3], weights[B,Q,heads,P]。

    2D: 双线性采样与权重求和；3D: value×depth 隐式体三线性采样与权重求和。
    输出[B,Q,heads,C]；offset、权重、图像特征和深度分布均保留梯度。
    """
    if depth is None:
        if locations.shape[-1] != 2:
            raise ValueError('3D 注意力必须传入 depth')
        depth = value.new_empty(0)
    elif locations.shape[-1] != 3:
        raise ValueError('带 depth 的注意力要求 3D locations')
    return _DeformAttention.apply(value, depth, locations, weights)
