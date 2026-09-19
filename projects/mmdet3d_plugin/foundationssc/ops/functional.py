"""仅适配布局/导入，不重写 CUDA 或 autograd；计算调用 FoundationSSC 原封装。"""
import importlib
import math
import torch


def require_extension(include_mmcv=True):
    """显式检查本地两项扩展与原工作依赖的 MMCV 2D attention。"""
    try:
        pool = importlib.import_module('.bev_pool.bev_pool_ext', __package__)
        dfa = importlib.import_module('.dfa3D._ext', __package__)
    except ImportError as exc:
        raise ImportError('请在 foundationssc/ops 中执行 python setup.py build_ext --inplace；旧 _C.so 不再使用。') from exc
    modules = {'bev_pool': pool, 'DFA3D': dfa}
    if include_mmcv:
        module = importlib.import_module('.multi_scale_deformable_attn_function', __package__)
        modules['MMCV attention'] = module.ext_module
    return modules


def use_cuda(backend, tensor):
    if backend not in ('cuda', 'pytorch', 'auto'):
        raise ValueError(f'未知算子后端：{backend}')
    if backend == 'pytorch':
        return False
    if backend == 'cuda' and not tensor.is_cuda:
        raise ValueError('cuda 后端要求 CUDA 输入')
    return tensor.is_cuda


def _check_float(*tensors):
    device = tensors[0].device
    if any(not t.is_cuda or t.device != device or t.dtype != torch.float32 for t in tensors):
        raise ValueError('本项目算子适配入口要求同一 CUDA 设备的 FP32 张量')


def lift_pool(context, depth, indices, voxel_shape):
    """[B,C,S] × [B,D,S] → [B,C,X*Y*Z]，indices 是本项目 Z 最快的局部线性索引。"""
    _check_float(context, depth)
    b, c, s = context.shape
    shape = (voxel_shape, 1, 1) if isinstance(voxel_shape, int) else tuple(voxel_shape)
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('voxel_shape 必须是正整数 X,Y,Z')
    x, y, z = shape
    if depth.ndim != 3 or depth.shape[0] != b or depth.shape[2] != s or indices.shape != depth.shape:
        raise ValueError('depth/indices 应为 [B,D,S]，并与 context 对应')
    if indices.device != context.device or indices.dtype != torch.long:
        raise ValueError('indices 必须是同设备 int64')
    #* 原 bev_pool 内核使用默认 CUDA stream；保留原源码，不假装支持非默认流。
    if torch.cuda.current_stream(context.device) != torch.cuda.default_stream(context.device):
        raise RuntimeError('原 FoundationSSC bev_pool 仅支持默认 CUDA stream')
    from .bev_pool import bev_pool
    valid = (indices >= 0) & (indices < math.prod(shape))
    #* 恢复原 LSS 的 Lift→过滤→排序分段汇聚；不再使用此前自写的融合内核。
    lifted = (context[:, :, None] * depth[:, None]).permute(0, 2, 3, 1)
    feats = lifted[valid].contiguous()
    if feats.shape[0] == 0:
        #* 原封装会访问 interval_lengths[-1]，空输入在外围处理，保持零梯度链。
        return context.new_zeros((b, c, x*y*z)) + feats.sum()
    ids = indices[valid]
    batch = torch.arange(b, device=context.device)[:, None, None].expand_as(indices)[valid]
    coords = torch.stack((ids // (y*z), (ids // z) % y, ids % z, batch), -1).int()
    with torch.cuda.device(context.device):
        pooled = bev_pool(feats, coords, b, z, x, y)  # 原输出 [B,C,Z,X,Y]。
    return pooled.permute(0, 1, 3, 4, 2).contiguous().reshape(b, c, x*y*z)


def deform_attention(value, locations, weights, depth=None):
    """当前单尺度布局 → 原多尺度接口；返回 [B,Q,heads,C_per_head]。

    value[B,heads,C,H,W]，locations[B,Q,heads,P,2/3]，weights[B,Q,heads,P]。
    DFA3D 原实现先得到四邻域 depth_score，再由 weighted attention 汇聚；
    不再调用本地自写三线性 CUDA 内核。2D 分支调用原 MMCV autograd 封装。
    """
    tensors = (value, locations, weights) if depth is None else (value, locations, weights, depth)
    _check_float(*tensors)
    b, heads, channels, h, w = value.shape
    dim = 2 if depth is None else 3
    if locations.ndim != 5 or locations.shape[0] != b or locations.shape[2] != heads or locations.shape[-1] != dim:
        raise ValueError('locations 维度不匹配')
    if weights.shape != locations.shape[:-1]:
        raise ValueError('weights 应与 locations 的采样点对应')
    if depth is not None and (depth.ndim != 4 or depth.shape[0] != b or depth.shape[-2:] != (h, w)):
        raise ValueError('depth 应为 [B,D,H,W]')
    q = locations.shape[1]
    if q == 0:
        return value.new_zeros((b, 0, heads, channels)) + sum(t.sum() * 0 for t in tensors)
    #* 原接口：value[B,HW,heads,C]，loc[B,Q,heads,level=1,P,dim]。
    values = value.permute(0, 3, 4, 1, 2).reshape(b, h*w, heads, channels).contiguous()
    loc = locations.unsqueeze(3).contiguous()
    attn = weights.unsqueeze(3).contiguous()
    starts = torch.zeros(1, dtype=torch.long, device=value.device)
    #* 原 im2col_step=64；需整除 batch，非整除时以单样本分块，不改变数值语义。
    step = min(b, 64) if b % min(b, 64) == 0 else 1
    with torch.cuda.device(value.device):
        if depth is None:
            from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
            shapes = starts.new_tensor([[h, w]])
            output = MultiScaleDeformableAttnFunction_fp32.apply(values, shapes, starts, loc, attn, step)
        else:
            from .multi_scale_3ddeformable_attn_function import MultiScale3DDeformableAttnFunction_fp32
            d = depth.shape[1]
            #* 对齐原 cross attention：每个 head 共用一份深度分布。
            distribution = depth.permute(0, 2, 3, 1).reshape(b, h*w, 1, d).repeat(1, 1, heads, 1).contiguous()
            shapes = starts.new_tensor([[h, w, d]])
            output, _ = MultiScale3DDeformableAttnFunction_fp32.apply(values, distribution, shapes, starts, loc, attn, step)
    return output.reshape(b, q, heads, channels)
