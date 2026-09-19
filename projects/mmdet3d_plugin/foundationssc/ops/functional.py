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


def prepare_attention(value, depth=None, batch_repeats=1):
    """每层仅准备一次原算子布局；返回的张量由所有 query 分块共享，不 detach。

    value[B,heads,C,H,W] → values[B,HW,heads,C]。
    自注意力 batch_repeats=2 表示原双队列；DFA3D 深度分布按 head 展开一次。
    返回对象只在当前 forward 内使用，不能跨迭代缓存，否则会持有旧计算图。
    """
    _check_float(value) if depth is None else _check_float(value, depth)
    b, heads, channels, h, w = value.shape
    if batch_repeats < 1 or not isinstance(batch_repeats, int):
        raise ValueError('batch_repeats 必须是正整数')
    if depth is not None and (depth.ndim != 4 or depth.shape[0] != b or depth.shape[-2:] != (h, w)):
        raise ValueError('depth 应为 [B,D,H,W]')
    values = value.permute(0, 3, 4, 1, 2).reshape(b, h*w, heads, channels).contiguous()
    if batch_repeats != 1:
        values = values.repeat_interleave(batch_repeats, dim=0)
    distribution = None
    shape = [h, w]
    if depth is not None:
        d = depth.shape[1]
        distribution = depth.permute(0, 2, 3, 1).reshape(b, h*w, 1, d).repeat(1, 1, heads, 1).contiguous()
        if batch_repeats != 1:
            distribution = distribution.repeat_interleave(batch_repeats, dim=0)
        shape.append(d)
    shapes = torch.tensor([shape], dtype=torch.long, device=value.device)
    starts = torch.zeros(1, dtype=torch.long, device=value.device)
    return values, distribution, shapes, starts


def deform_attention_prepared(memory, locations, weights):
    """仅处理当前 query 块；memory 中的大张量直接传入原封装，不复制、不修改。"""
    values, distribution, shapes, starts = memory
    b, _, heads, channels = values.shape
    dim = 2 if distribution is None else 3
    _check_float(values, locations, weights)
    if locations.ndim != 5 or locations.shape[0] != b or locations.shape[2] != heads or locations.shape[-1] != dim:
        raise ValueError('locations 维度不匹配')
    if weights.shape != locations.shape[:-1]:
        raise ValueError('weights 应与 locations 的采样点对应')
    q = locations.shape[1]
    if q == 0:
        tensors = (values, locations, weights) if distribution is None else (values, locations, weights, distribution)
        return values.new_zeros((b, 0, heads, channels)) + sum(t.sum() * 0 for t in tensors)
    loc = locations.unsqueeze(3).contiguous()
    attn = weights.unsqueeze(3).contiguous()
    step = min(b, 64) if b % min(b, 64) == 0 else 1
    #* 原封装会 save_for_backward(values/distribution)。各分块保存同一存储的引用，
    #* 而不是每块保留一份完整特征；所有块的梯度仍累加到原来的 value/depth。
    with torch.cuda.device(values.device):
        if distribution is None:
            from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction_fp32.apply(values, shapes, starts, loc, attn, step)
        else:
            from .multi_scale_3ddeformable_attn_function import MultiScale3DDeformableAttnFunction_fp32
            output, _ = MultiScale3DDeformableAttnFunction_fp32.apply(values, distribution, shapes, starts, loc, attn, step)
    return output.reshape(b, q, heads, channels)


def deform_attention(value, locations, weights, depth=None):
    """单次调用兼容入口；分块循环请先 prepare_attention，再复用 memory。"""
    return deform_attention_prepared(prepare_attention(value, depth), locations, weights)
