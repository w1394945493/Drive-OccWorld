#!/usr/bin/env python3
"""FoundationSSC 分阶段完整验证入口。

当前已实现：data 数据检查；images 图像特征；voxels 三维体素特征与梯度检查。
后续扩展：占据预测、训练损失和评估。
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


def check_batch(batch):
    images, rots, trans, intrins, post_rots, post_trans, bda, c2l = batch['img_inputs']
    assert images.ndim == 5 and images.shape[:3] == (1, 2, 3)
    assert intrins.shape == (1, 2, 4, 4) and c2l.shape == (1, 2, 4, 4)
    assert bda.shape == (1, 4, 4)
    assert batch['gt_occ'].dtype == torch.int64 and batch['gt_occ'].ndim == 4
    for tensor in batch['img_inputs']:
        assert torch.isfinite(tensor).all()
    meta = batch['img_metas']
    assert (meta['baseline'] > 0).all()
    torch.testing.assert_close(c2l[..., :3, :3], rots)
    torch.testing.assert_close(c2l[..., :3, 3], trans)
    #* 检查投影：K4 @ lidar2camera 与 PKL 的 lidar2img 必须一致。
    projected = (intrins @ torch.linalg.inv(c2l))[..., :3, :]
    torch.testing.assert_close(projected, meta['lidar2img'], atol=1e-3, rtol=1e-5)
    for cam in range(2):
        raw = meta['raw_img'][cam]
        assert raw.dtype == torch.uint8 and raw.shape == (1, images.shape[-2], images.shape[-1], 3)
        rgb = raw.permute(0, 3, 1, 2).float() / 255
        expected = (rgb - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
        torch.testing.assert_close(images[:, cam], expected)
    #* 检查 post transform 正反变换；不改变相机内参以免增强被重复应用。
    point = torch.tensor([120., 90., 1.]).expand(1, 2, 3)
    transformed = (post_rots @ point[..., None]).squeeze(-1) + post_trans
    recovered = (torch.linalg.inv(post_rots) @ (transformed - post_trans)[..., None]).squeeze(-1)
    torch.testing.assert_close(point, recovered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/configs/foundationssc/foundationssc_semantic_kitti.py')
    parser.add_argument('--stage', choices=['data', 'images', 'voxels'], default='data')
    parser.add_argument('--stereo-checkpoint', help='覆盖 FoundationStereo 权重路径')
    parser.add_argument('--stereo-config', help='覆盖 FoundationStereo YAML 路径')
    parser.add_argument('--check-grad', action='store_true', help='使用特征探针检查反传，不是占据训练损失')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ops-backend', choices=['cuda', 'pytorch', 'auto'], help='覆盖第三阶段算子后端')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--ann-file', help='覆盖 PKL 路径；图像/标签路径仍取 PKL')
    parser.add_argument('--indices', nargs='+', type=int, default=[0])
    args = parser.parse_args()
    if args.check_grad and args.stage == 'data':
        parser.error('--check-grad 需要 --stage images 或 voxels')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin  # noqa: F401，注册 Dataset 与 pipeline
    cfg = Config.fromfile(args.config)
    dataset_cfg = cfg.data[args.split].copy()
    if args.ann_file:
        dataset_cfg['ann_file'] = args.ann_file
    dataset = build_dataset(dataset_cfg)
    for index in args.indices:
        if not 0 <= index < len(dataset):
            raise IndexError(f'索引 {index} 越界，样本数 {len(dataset)}')
    loader = DataLoader(Subset(dataset, args.indices), batch_size=1, num_workers=0, shuffle=False)
    model = None
    if args.stage in ('images', 'voxels'):
        from projects.mmdet3d_plugin.foundationssc import FoundationSSCImageModel
        options = dict(cfg.model)
        options.pop('type')
        if args.stage == 'images':
            options.pop('voxel_encoder', None)  #* 第二阶段不额外分配三维模块。
        elif args.ops_backend is not None:
            options['voxel_encoder'] = dict(options['voxel_encoder'], ops_backend=args.ops_backend)
        for key in ('stereo_checkpoint', 'stereo_config'):
            value = getattr(args, key)
            if value is not None:
                options[key] = value
        device = torch.device(args.device)
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('真实 FoundationStereo 图像阶段请使用 CUDA 环境')
        torch.cuda.set_device(device)
        model = FoundationSSCImageModel(**options).to(device)
        model.train(args.check_grad)
        print(f'权重检查：{model.checkpoint_report}')
        print(f'冻结骨干参数量：{sum(p.numel() for p in model.img_backbone.parameters()):,}')
        print(f'可训练 FPN 参数量：{sum(p.numel() for p in model.image_pyramid.parameters()):,}')
        if model.voxel_encoder is not None:
            print(f'可训练体素前端参数量：{sum(p.numel() for p in model.voxel_encoder.parameters()):,}')
            print(f'第三阶段算子后端：{model.voxel_encoder.ops_backend}；暂不包含 occupancy 分类头。')
    for index, batch in zip(args.indices, loader):
        check_batch(batch)
        print(f"样本 {index}，场景 {batch['img_metas']['scene_name'][0]}，帧 {batch['img_metas']['token'][0]}")
        names = ('归一化图像', 'camera→lidar旋转', 'camera→lidar平移', '4x4内参', '图像变换旋转', '图像变换平移', 'BDA单位阵', 'camera→lidar矩阵')
        for name, tensor in zip(names, batch['img_inputs']):
            print(f'  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}')
        print(f"  occupancy: {tuple(batch['gt_occ'].shape)}；基线(m): {batch['img_metas']['baseline'].item():.6f}")
        print('  投影、归一化、标定和 batch 检查通过。')
        if model is not None:
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            with torch.set_grad_enabled(args.check_grad):
                output = model(return_loss=False, stage=args.stage, **batch)
                feature = output['img_feats']
                assert feature.shape[:3] == (1, 1, 640), feature.shape
                assert feature.shape[-2:] == tuple(v // 8 for v in batch['img_inputs'][0].shape[-2:]), feature.shape
                tensors = [feature, *output['disparity'], *output['pyramid']]
                tensors += [value for pair in output['dino_features'] for value in pair]
                assert all(torch.isfinite(value).all() for value in tensors)
                assert not model.img_backbone.training
                if args.stage == 'voxels':
                    from projects.mmdet3d_plugin.foundationssc.voxel.geometry import project, unproject
                    geom = model.voxel_encoder.geometry
                    expected = (1, cfg.model.voxel_encoder.channels, *geom.voxel_shape)
                    assert output['voxel_feats'].shape == expected
                    for key in ('voxel_feats', 'coarse_voxel', 'refined_voxel', 'context', 'depth_prob', 'stereo_depth', 'proposal'):
                        assert torch.isfinite(output[key]).all(), key
                        print(f'  {key}: {tuple(output[key].shape)}')
                    torch.testing.assert_close(output['depth_prob'].sum(1), torch.ones_like(output['depth_prob'][:, 0]), atol=1e-5, rtol=1e-5)
                    assert (output['lifted_valid_points'] > 0).all(), '所有 LSS 射线都在范围外，请检查标定/坐标轴'
                    assert (output['visible_voxels'] > 0).all(), '没有可见体素，请检查相机外参'
                    print(f"  有效 LSS 点: {output['lifted_valid_points'].tolist()}；可见体素: {output['visible_voxels'].tolist()}；候选数: {int(output['proposal'].sum())}")
                    #* 用实际样本标定验证正反投影；不是单纯检查张量形状。
                    cam = [x[:, :1].to(device).float() for x in batch['img_inputs'][1:6]] + [batch['img_inputs'][6].to(device).float()]
                    uvd = torch.tensor([[[[100., 100., 10.], [500., 200., 20.]]]], device=device)
                    torch.testing.assert_close(project(unproject(uvd, cam), cam), uvd, atol=1e-3, rtol=1e-4)
                    torch.testing.assert_close(geom.lower.cpu(), batch['img_metas']['pc_range'][0, :3])
                    upper = geom.lower + geom.voxel_size * geom.voxel_size.new_tensor(geom.voxel_shape)
                    torch.testing.assert_close(upper.cpu(), batch['img_metas']['pc_range'][0, 3:])
                    print('  深度概率、实际标定正反投影和三维空间覆盖检查通过。')
                if args.check_grad:
                    #* 只验证梯度连通性，不代表已接入 SSC loss 或 optimizer。
                    probe = (output['voxel_feats'].square().mean() if args.stage == 'voxels' else feature.square().mean())
                    probe.backward()
                    grads = [p.grad for p in model.image_pyramid.parameters() if p.requires_grad]
                    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
                    assert any(g.abs().sum() > 0 for g in grads)
                    assert all(not p.requires_grad and p.grad is None for p in model.img_backbone.parameters())
                    print('  FPN 梯度有效；FoundationStereo 无梯度。')
                    if args.stage == 'voxels':
                        for name in ('depth_net', 'refiner', 'fusion'):
                            module = getattr(model.voxel_encoder, name)
                            values = [p.grad for p in module.parameters() if p.grad is not None]
                            assert values and all(torch.isfinite(x).all() for x in values), name
                            assert any(torch.count_nonzero(x) for x in values), name
                            print(f'  {name} 反向梯度检查通过。')
                        del values, module
                    del probe
            torch.cuda.synchronize(device)
            print(f'  融合图像特征: {tuple(feature.shape)}, dtype={feature.dtype}')
            print(f"  DINO各层: {[tuple(pair[0].shape) for pair in output['dino_features']]}")
            print(f"  视差概率/视差图: {[tuple(x.shape) for x in output['disparity']]}")
            print(f'  耗时: {time.perf_counter()-start:.3f}s；显存峰值: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GiB')
            del output, feature, tensors
            if args.check_grad:
                del grads
            model.zero_grad(set_to_none=True)
    print(f'{args.stage} 阶段检查完成；尚未接入占据分类头及训练损失。')


if __name__ == '__main__':
    main()
