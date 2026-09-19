#!/usr/bin/env python3
"""第三阶段 CPU 回归检查：无原仓库、无权重、无 MMCV/CUDA 扩展。"""
import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch.nn import functional as F


def load_local_package():
    root = Path(__file__).resolve().parents[1] / 'projects/mmdet3d_plugin/foundationssc/voxel'
    parent = types.ModuleType('_foundation_test')
    parent.__path__ = [str(root.parent)]
    sys.modules[parent.__name__] = parent
    spec = importlib.util.spec_from_file_location('_foundation_test.voxel', root / '__init__.py', submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def cameras(batch=1):
    rot = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])[None, None].repeat(batch, 1, 1, 1)
    trans = torch.zeros(batch, 1, 3)
    k = torch.eye(4)[None, None].repeat(batch, 1, 1, 1)
    k[..., 0, 0] = k[..., 1, 1] = 16
    k[..., 0, 2] = k[..., 1, 2] = 16
    return [rot, trans, k, torch.eye(3)[None, None].repeat(batch, 1, 1, 1),
            torch.zeros(batch, 1, 3), torch.eye(4)[None].repeat(batch, 1, 1)]


def test_geometry():
    from _foundation_test.voxel.geometry import unproject, project, disparity_to_depth, VoxelGeometry
    cam = cameras()
    p = torch.tensor([[[[16., 16., 4.]]]])
    torch.testing.assert_close(unproject(p, cam), torch.tensor([[[[4., 0., 0.]]]]))
    #* 手算非中心像素：向图像右/下分别对应 LiDAR -Y/-Z，验证轴方向。
    torch.testing.assert_close(unproject(torch.tensor([[[[20., 20., 4.]]]]), cam), torch.tensor([[[[4., -1., -1.]]]]))
    cam[3][..., 0, 0] = cam[3][..., 1, 1] = 2
    cam[4][..., :2] = torch.tensor([-3., -5.])
    cam[1][..., :] = torch.tensor([.2, -.1, .3])
    cam[5][..., :3, 3] = torch.tensor([1., 2., 3.])
    # K 第四列非零的旧 KITTI 约定也应正确往返。
    cam[2][..., :3, 3] = torch.tensor([.1, .2, .01])
    torch.testing.assert_close(project(unproject(p, cam), cam), p, atol=1e-5, rtol=1e-5)
    # 原焦距16，缩放2，基线.5，增强视差4 → 深度4米。
    torch.testing.assert_close(disparity_to_depth(torch.full((1, 1, 2, 2), 4.), torch.tensor([.5]), cam), torch.full((1, 1, 2, 2), 4.))
    geom = VoxelGeometry((0, -4, -2, 8, 4, 2), (8, 8, 4), (1, 9, 1), (32, 32))
    _, valid = geom.indices(torch.tensor([[-.01, 0., 0.], [0., -4., -2.], [8., 0., 0.]]))
    assert valid.tolist() == [False, True, False]
    assert geom.proposal(torch.zeros(2, 1, 32, 32), cameras(2)).shape == (2, 1, 8, 8, 4)
    assert geom.proposal(torch.zeros(2, 1, 32, 32), cameras(2)).sum() == 0
    context = torch.randn(1, 1, 2, 4, 4, requires_grad=True)
    prob = torch.randn(1, 8, 4, 4).softmax(1).requires_grad_()
    a, _ = geom.lift_splat(context, prob, cameras())
    geom.pool_chunk = 1
    b, _ = geom.lift_splat(context, prob, cameras())
    torch.testing.assert_close(a, b)
    a.square().sum().backward()
    assert context.grad.abs().sum() > 0 and prob.grad.abs().sum() > 0
    #* 独立手算单射线的 splat 位置和数值，排除 X/Y/Z 交换后仍能 roundtrip 的假通过。
    single = VoxelGeometry((0, -4, -2, 8, 4, 2), (8, 8, 4), (4, 5, 1), (1, 1), downsample=1)
    cam = cameras(); cam[2][..., :2, 2] = 0
    result, hits = single.lift_splat(torch.tensor([[[[[2.]], [[3.]]]]]), torch.tensor([[[[.75]]]]), cam)
    expected = torch.zeros_like(result)
    expected[0, :, 4, 4, 2] = torch.tensor([1.5, 2.25])
    torch.testing.assert_close(result, expected)
    assert hits.item() == 1
    print('通过：手算轴方向、缩放视差→米制深度、投影往返、边界/空候选、分块 LSS 梯度')


def test_sampling():
    from _foundation_test.voxel.refiner import sample_depth_weighted
    v = torch.randn(1, 2, 3, 3, 4, dtype=torch.double, requires_grad=True)
    d = torch.rand(1, 5, 3, 4, dtype=torch.double, requires_grad=True)
    loc = (torch.rand(1, 7, 2, 2, 3, dtype=torch.double) * 1.2 - .1).requires_grad_()
    actual = sample_depth_weighted(v, d, loc)
    volume = (v.unsqueeze(3) * d[:, None, None]).reshape(2, 3, 5, 3, 4)
    grid = loc.permute(0, 2, 1, 3, 4).reshape(2, 1, 7, 2, 3) * 2 - 1
    expected = F.grid_sample(volume, grid, align_corners=False).reshape(1, 2, 3, 7, 2).permute(0, 3, 1, 4, 2)
    torch.testing.assert_close(actual, expected)
    grad_a = torch.autograd.grad(actual.square().sum(), (v, d, loc), retain_graph=True)
    grad_b = torch.autograd.grad(expected.square().sum(), (v, d, loc))
    for a, b in zip(grad_a, grad_b):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-7)
    print('通过：隐式 DFA3D 插值与显式 3D grid_sample 的数值/梯度对照')


def test_encoder():
    from _foundation_test.voxel.encoder import FoundationVoxelEncoder
    model = FoundationVoxelEncoder(
        point_cloud_range=(0, -4, -2, 8, 4, 2), voxel_shape=(8, 8, 4), input_size=(32, 32),
        depth_bound=(1, 9, 1), input_channels=32, channels=32, disparity_channels=8,
        depth_cfg=dict(dformer_layers=1, mixer_layers=1), fusion_groups=4,
        refiner_cfg=dict(cross_layers=1, self_layers=1, heads=8, points=2,
                         ffn_channels=64, self_layout=(16, 16), query_chunk=64, dropout=0.))
    for batch in (1, 2):
        model.train(); model.zero_grad(set_to_none=True)
        feat = torch.randn(batch, 1, 32, 4, 4, requires_grad=True)
        cam = cameras(batch)
        inputs = [torch.empty(batch, 2, 3, 32, 32)] + [x.repeat(1, 2, *([1] * (x.ndim - 2))) for x in cam[:5]] + [cam[5]]
        disp = [torch.rand(batch, 8, 8, 8).softmax(1), torch.full((batch, 1, 32, 32), 2.)]
        out = model(dict(img_feats=feat, disparity=disp), inputs, dict(baseline=torch.full((batch,), .5)))
        assert out['voxel_feats'].shape == (batch, 32, 8, 8, 4)
        assert all(torch.isfinite(x).all() for x in out.values())
        assert out['proposal'].sum() > 0 and (out['lifted_valid_points'] > 0).all()
        torch.testing.assert_close(out['depth_prob'].sum(1), torch.ones(batch, 4, 4))
        out['voxel_feats'].square().mean().backward()
        assert feat.grad is not None and torch.isfinite(feat.grad).all() and feat.grad.abs().sum() > 0
        for name in ('depth_net', 'refiner', 'fusion'):
            grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(x).all() for x in grads), name
            assert any(x.abs().sum() > 0 for x in grads), name
        print(f'通过：batch={batch} 深度/LSS/候选/注意力/融合完整前向及反传')
    #* 全无效视差必须稳健返回空 proposal，不能走原代码的随机点兜底。
    model.eval()
    with torch.no_grad():
        disp[1].zero_()
        out = model(dict(img_feats=feat, disparity=disp), inputs, dict(baseline=torch.full((batch,), .5)))
    assert out['proposal'].sum() == 0 and torch.isfinite(out['voxel_feats']).all()
    print('通过：全无效视差/空候选仍能前向，未制造伪候选')


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.manual_seed(0)
    load_local_package()
    test_geometry()
    test_sampling()
    test_encoder()
