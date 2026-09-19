#!/usr/bin/env python3
"""验证原扩展源码、适配层与原封装的前后向一致性，以及完整体素模块接线。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import torch


def load_local_package():
    #* 独立加载本地体素包，避免算子测试依赖整个训练插件入口。
    root = Path(__file__).resolve().parents[1] / 'projects/mmdet3d_plugin/foundationssc/voxel'
    parent = types.ModuleType('_foundation_test')
    parent.__path__ = [str(root.parent)]
    sys.modules[parent.__name__] = parent
    spec = importlib.util.spec_from_file_location('_foundation_test.voxel', root / '__init__.py', submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def cameras(batch=1):
    #* 小尺寸体素测试使用的人工标定，不读取真实数据集。
    rot = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])[None, None].repeat(batch, 1, 1, 1)
    trans = torch.zeros(batch, 1, 3)
    k = torch.eye(4)[None, None].repeat(batch, 1, 1, 1)
    k[..., 0, 0] = k[..., 1, 1] = 16
    k[..., 0, 2] = k[..., 1, 2] = 16
    return [rot, trans, k, torch.eye(3)[None, None].repeat(batch, 1, 1, 1),
            torch.zeros(batch, 1, 3), torch.eye(4)[None].repeat(batch, 1, 1)]


def check_sources(reference_root=None):
    root = Path(__file__).resolve().parents[1] / 'projects/mmdet3d_plugin/foundationssc/ops'
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((root / item['local']).read_bytes()).hexdigest() == item['local_sha256'], item['local']
        if reference_root:
            source = Path(reference_root) / item['source']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item['source_sha256'], str(source)
    print(f"通过：{len(manifest['files'])} 个原始文件源码校验（仅两处 Python 文件调整本地导入）")


def compare(name, cuda_fn, ref_fn, inputs, atol=3e-4, rtol=3e-4):
    a = [x.detach().requires_grad_(True) for x in inputs]
    b = [x.detach().clone().requires_grad_(True) for x in inputs]
    actual, expected = cuda_fn(*a), ref_fn(*b)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    grad = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, a, grad)
    gb = torch.autograd.grad(expected, b, grad)
    for i, (x, y) in enumerate(zip(ga, gb)):
        assert torch.isfinite(x).all(), (name, i)
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol, msg=lambda msg: f'{name} 输入{i}梯度不一致: {msg}')
    print(f'通过：{name} 前向及 {len(inputs)} 项输入梯度')


def pool_reference(context, depth, indices, count):
    b, c, s = context.shape
    values = (context[:, :, None] * depth[:, None]).permute(0, 2, 3, 1)
    valid = (indices >= 0) & (indices < count)
    index = indices + torch.arange(b, device=context.device)[:, None, None] * count
    out = context.new_zeros((b * count, c)).index_add(0, index[valid], values[valid])
    return out.reshape(b, count, c).transpose(1, 2)


def attention_reference(value, locations, weights, depth=None):
    #* 直接按 FoundationSSC 调用原封装，检查适配层是否正确排列维度和传递梯度。
    from _foundation_test.ops.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
    from _foundation_test.ops.multi_scale_3ddeformable_attn_function import MultiScale3DDeformableAttnFunction_fp32
    b, m, c, h, w = value.shape
    v = value.flatten(-2).permute(0,3,1,2).contiguous()
    loc, attn = locations.unsqueeze(3).contiguous(), weights.unsqueeze(3).contiguous()
    starts = torch.zeros(1, dtype=torch.long, device=value.device)
    if depth is None:
        out = MultiScaleDeformableAttnFunction_fp32.apply(v, starts.new_tensor([[h,w]]), starts, loc, attn, 64)
    else:
        dist = depth.flatten(-2).transpose(1,2).unsqueeze(2).repeat(1,1,m,1).contiguous()
        out, _ = MultiScale3DDeformableAttnFunction_fp32.apply(v, dist, starts.new_tensor([[h,w,depth.shape[1]]]), starts, loc, attn, 64)
    return out.reshape(b, locations.shape[1], m, c)


def test_ops(device):
    from _foundation_test.ops import lift_pool, deform_attention
    for batch in (1, 2):
        context = torch.randn(batch, 4, 15, device=device)
        depth = torch.rand(batch, 7, 15, device=device)
        #* 重复 voxel、无效点以及越界索引均覆盖；使用非连续的 context。
        indices = torch.randint(-2, 26, (batch, 7, 15), device=device)
        c = context.transpose(1,2).contiguous().transpose(1,2)
        #* 非立方形空间检查原 [B,C,Z,X,Y] 到本项目 [B,C,X,Y,Z] 的轴顺序。
        compare(f'原 LSS vs index_add batch={batch}', lambda c,d: lift_pool(c,d,indices,(3,4,2)),
                lambda c,d: pool_reference(c,d,indices,24), [c,depth])
        indices.fill_(-1)
        compare(f'LSS 全无效 batch={batch}', lambda c,d: lift_pool(c,d,indices,11),
                lambda c,d: pool_reference(c,d,indices,11), [c,depth])
        for dim in (2, 3):
            value = torch.randn(batch,2,4,5,6,device=device)
            locations = torch.rand(batch,9,2,3,dim,device=device)*1.6-.3
            #* 包括归一化边界和有部分邻居越界的半像素位置。
            locations[:,0] = 0; locations[:,1] = 1
            locations[:,2,:,:,0] = -.5/6
            locations[:,3,:,:,0] = 2.5/6  # 整数像素中心；以原扩展的梯度约定为准。
            weights = torch.randn(batch,9,2,3,device=device)
            if dim == 2:
                compare(f'MMCV 2D 适配层 vs 原封装 batch={batch}', deform_attention, attention_reference, [value,locations,weights])
            else:
                distribution = torch.rand(batch,7,5,6,device=device)
                compare(f'DFA3D 适配层 vs 原封装 batch={batch}',
                        lambda v,d,l,w: deform_attention(v,l,w,d),
                        lambda v,d,l,w: attention_reference(v,l,w,d), [value,distribution,locations,weights])
            #* 零候选也必须能 backward，不能以零个 block 发起 CUDA launch。
            empty = locations[:,:0].contiguous().requires_grad_()
            value.requires_grad_()
            out = deform_attention(value,empty,weights[:,:0], None if dim == 2 else distribution)
            out.sum().backward()
            assert out.shape == (batch,0,2,4) and torch.count_nonzero(value.grad) == 0
    print('通过：空 query 前向/反向')


def test_encoder(device):
    from _foundation_test.voxel.encoder import FoundationVoxelEncoder
    model = FoundationVoxelEncoder(
        point_cloud_range=(0,-4,-2,8,4,2), voxel_shape=(8,8,4), input_size=(32,32),
        depth_bound=(1,9,1), input_channels=32, channels=32, disparity_channels=8,
        depth_cfg=dict(dformer_layers=1,mixer_layers=1), fusion_groups=4,
        refiner_cfg=dict(cross_layers=1,self_layers=1,heads=8,points=2,ffn_channels=64,
                         self_layout=(16,16),query_chunk=64,dropout=0.), ops_backend='cuda').to(device).eval()
    batch = 2
    cam = [x.to(device) for x in cameras(batch)]
    inputs = [torch.empty(batch,2,3,32,32,device=device)] + [x.repeat(1,2,*([1]*(x.ndim-2))) for x in cam[:5]] + [cam[5]]
    disp = [torch.rand(batch,8,8,8,device=device).softmax(1),torch.full((batch,1,32,32),2.,device=device)]
    feature = torch.randn(batch,1,32,4,4,device=device,requires_grad=True)
    meta = dict(baseline=torch.full((batch,),.5,device=device))
    out = model(dict(img_feats=feature,disparity=disp),inputs,meta)
    assert out['voxel_feats'].shape == (batch,32,8,8,4)
    for key, value in out.items():
        assert torch.isfinite(value).all(), key
    target = torch.randn_like(out['voxel_feats'])
    (out['voxel_feats']*target).mean().backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    grads = {name:p.grad for name,p in model.named_parameters() if p.grad is not None}
    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), name
    for prefix in ('refiner.cross.0.offset_uv', 'refiner.cross.0.offset_d', 'refiner.self_attention.0.offset'):
        assert any(name.startswith(prefix) and grad.abs().sum() > 0 for name,grad in grads.items()), prefix
    #* 本次基准改为原 CUDA/MMCV 封装；不要求完整网络在插值折点的梯度等于 grid_sample。
    print(f'通过：batch=2 原扩展完整三维前端前向/反向，{len(grads)} 项参数梯度有限')


def benchmark(device):
    from _foundation_test.ops import lift_pool, deform_attention
    c = torch.randn(1,128,48*160,device=device)
    d = torch.rand(1,112,48*160,device=device)
    idx = torch.randint(-1,128*128*16,(1,112,48*160),device=device)
    v = torch.randn(1,8,16,48,160,device=device)
    loc = torch.rand(1,2048,8,8,3,device=device)
    w = torch.rand(1,2048,8,8,device=device)
    tests = [
        ('LSS/PyTorch',lambda:pool_reference(c,d,idx,128*128*16)),
        ('LSS/原CUDA',lambda:lift_pool(c,d,idx,(128,128,16))),
        ('DFA3D/原封装',lambda:attention_reference(v,loc,w,d.reshape(1,112,48,160))),
        ('DFA3D/CUDA',lambda:deform_attention(v,loc,w,d.reshape(1,112,48,160))),
    ]
    for name,func in tests:
        with torch.no_grad():
            for _ in range(2): func()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            start=time.perf_counter()
            for _ in range(5): func()
            torch.cuda.synchronize(device)
        print(f'{name}: 前向 {(time.perf_counter()-start)*200:.2f} ms，额外峰值 {(torch.cuda.max_memory_allocated(device)-baseline)/2**20:.1f} MiB')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--import-only',action='store_true',help='仅检查本地两项扩展和 MMCV 加载，不代表通过 CUDA 数值测试')
    parser.add_argument('--source-only',action='store_true',help='仅核对源码，无需 GPU/MMCV')
    parser.add_argument('--reference-root',help='可选：原 FoundationSSC 仓库路径，用于核对源文件 SHA256；运行模型不依赖它')
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    check_sources(args.reference_root)
    if args.source_only:
        raise SystemExit(0)
    load_local_package()
    from _foundation_test.ops import require_extension
    for name, ext in require_extension().items():
        print(f'扩展加载成功：{name}: {ext.__file__}')
    if not args.import_only:
        device=torch.device(args.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA 数值验证需要可用 GPU；无 GPU 仅可使用 --import-only')
        torch.cuda.set_device(device)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        #* 原 bev_pool 在默认流 launch；不改源内核，因此使用默认流验证。
        test_ops(device)
        test_encoder(device)
        torch.cuda.synchronize()
        print('全部原扩展适配检查通过。')
        if args.benchmark: benchmark(device)
