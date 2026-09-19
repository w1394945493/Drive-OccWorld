#!/usr/bin/env python3
"""就地 CUDA 扩展验证：与 PyTorch 比较前向/所有输入梯度，并检查三维模块接线。"""
import argparse
import copy
import time

import torch
from torch.nn import functional as F
from test_foundationssc_local_voxel import load_local_package, cameras


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
    b, m, c, h, w = value.shape
    q, p = locations.shape[1], locations.shape[3]
    if depth is None:
        grid = locations.permute(0, 2, 1, 3, 4).reshape(b*m, q, p, 2)
        samples = F.grid_sample(value.reshape(b*m,c,h,w), grid*2-1, align_corners=False)
        samples = samples.reshape(b,m,c,q,p).permute(0,3,1,4,2)
    else:
        from _foundation_test.voxel.refiner import sample_depth_weighted
        samples = sample_depth_weighted(value, depth, locations)
    return (samples * weights[..., None]).sum(-2)


def test_ops(device):
    from _foundation_test.ops import lift_pool, deform_attention
    for batch in (1, 2):
        context = torch.randn(batch, 4, 15, device=device)
        depth = torch.rand(batch, 7, 15, device=device)
        #* 重复 voxel、无效点以及越界索引均覆盖；使用非连续的 context。
        indices = torch.randint(-2, 13, (batch, 7, 15), device=device)
        c = context.transpose(1,2).contiguous().transpose(1,2)
        compare(f'LSS batch={batch}', lambda c,d: lift_pool(c,d,indices,11),
                lambda c,d: pool_reference(c,d,indices,11), [c,depth])
        indices.fill_(-1)
        compare(f'LSS 全无效 batch={batch}', lambda c,d: lift_pool(c,d,indices,11),
                lambda c,d: pool_reference(c,d,indices,11), [c,depth])
        for dim in (2, 3):
            value = torch.randn(batch,2,4,5,6,device=device)
            locations = torch.rand(batch,9,2,3,dim,device=device)*1.6-.3
            #* 包括归一化边界和有部分邻居越界的半像素位置。
            locations[:,0] = 0; locations[:,1] = 1
            locations[:,2,:,:,0] = -.5/6
            weights = torch.randn(batch,9,2,3,device=device)
            if dim == 2:
                compare(f'2D 注意力 batch={batch}', deform_attention, attention_reference, [value,locations,weights])
            else:
                distribution = torch.rand(batch,7,5,6,device=device)
                compare(f'DFA3D batch={batch}',
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
    from _foundation_test.voxel.refiner import CrossLayer, SelfLayer
    model = FoundationVoxelEncoder(
        point_cloud_range=(0,-4,-2,8,4,2), voxel_shape=(8,8,4), input_size=(32,32),
        depth_bound=(1,9,1), input_channels=32, channels=32, disparity_channels=8,
        depth_cfg=dict(dformer_layers=1,mixer_layers=1), fusion_groups=4,
        refiner_cfg=dict(cross_layers=1,self_layers=1,heads=8,points=2,ffn_channels=64,
                         self_layout=(16,16),query_chunk=64,dropout=0.), ops_backend='pytorch').to(device).eval()
    fast = copy.deepcopy(model)
    fast.geometry.ops_backend = fast.ops_backend = 'cuda'
    for module in fast.modules():
        if isinstance(module,(CrossLayer,SelfLayer)):
            module.ops_backend = 'cuda'
    batch = 2
    cam = [x.to(device) for x in cameras(batch)]
    inputs = [torch.empty(batch,2,3,32,32,device=device)] + [x.repeat(1,2,*([1]*(x.ndim-2))) for x in cam[:5]] + [cam[5]]
    disp = [torch.rand(batch,8,8,8,device=device).softmax(1),torch.full((batch,1,32,32),2.,device=device)]
    feature = torch.randn(batch,1,32,4,4,device=device,requires_grad=True)
    f2 = feature.detach().clone().requires_grad_()
    meta = dict(baseline=torch.full((batch,),.5,device=device))
    ref = model(dict(img_feats=feature,disparity=disp),inputs,meta)
    out = fast(dict(img_feats=f2,disparity=disp),inputs,meta)
    for key in ref:
        torch.testing.assert_close(out[key],ref[key],atol=1e-3,rtol=1e-3)
    target = torch.randn_like(out['voxel_feats'])
    (out['voxel_feats']*target).mean().backward()
    (ref['voxel_feats']*target).mean().backward()
    torch.testing.assert_close(f2.grad,feature.grad,atol=3e-4,rtol=2e-3)
    for (name,p), (name2,q) in zip(model.named_parameters(),fast.named_parameters()):
        assert name == name2
        if p.grad is None:
            assert q.grad is None, name
        else:
            assert q.grad is not None and torch.isfinite(q.grad).all(), name
            torch.testing.assert_close(q.grad,p.grad,atol=3e-4,rtol=2e-3,msg=lambda msg: f'{name}: {msg}')
    print('通过：batch=2 完整三维前端 PyTorch/CUDA 中间结果、最终特征、输入和参数梯度对照')


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
        ('LSS/CUDA',lambda:lift_pool(c,d,idx,128*128*16)),
        ('DFA3D/PyTorch',lambda:attention_reference(v,loc,w,d.reshape(1,112,48,160))),
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
    parser.add_argument('--import-only',action='store_true',help='无 GPU 时仅检查就地 .so 加载，不代表通过 CUDA 数值测试')
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    load_local_package()
    from _foundation_test.ops import require_extension
    ext=require_extension()
    print('就地扩展加载成功：',ext.__file__)
    if not args.import_only:
        device=torch.device(args.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('CUDA 数值验证需要可用 GPU；无 GPU 仅可使用 --import-only')
        torch.cuda.set_device(device)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        #* 非默认流验证 device guard / current stream，不仅测试默认 stream。
        stream=torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            test_ops(device)
            test_encoder(device)
        stream.synchronize()
        print('全部 CUDA 对照检查通过（含非默认 CUDA stream）。')
        if args.benchmark: benchmark(device)
