#!/usr/bin/env python3
"""FoundationSSC 分阶段完整验证入口。

当前已实现：data 数据检查；images 图像特征前向及可选 FPN 梯度检查。
后续扩展：体素特征、占据预测、训练损失和评估。
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
    parser.add_argument('--stage', choices=['data', 'images'], default='data')
    parser.add_argument('--stereo-checkpoint', help='覆盖 FoundationStereo 权重路径')
    parser.add_argument('--stereo-config', help='覆盖 FoundationStereo YAML 路径')
    parser.add_argument('--check-grad', action='store_true', help='images 阶段使用特征平方均值检查 FPN 反传，不是占据训练损失')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--ann-file', help='覆盖 PKL 路径；图像/标签路径仍取 PKL')
    parser.add_argument('--indices', nargs='+', type=int, default=[0])
    args = parser.parse_args()
    if args.check_grad and args.stage != 'images':
        parser.error('--check-grad 需要 --stage images')
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
    if args.stage == 'images':
        from projects.mmdet3d_plugin.foundationssc import FoundationSSCImageModel
        options = dict(cfg.model)
        options.pop('type')
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
                output = model(return_loss=False, **batch)
                feature = output['img_feats']
                assert feature.shape[:3] == (1, 1, 640), feature.shape
                assert feature.shape[-2:] == tuple(v // 8 for v in batch['img_inputs'][0].shape[-2:]), feature.shape
                tensors = [feature, *output['disparity'], *output['pyramid']]
                tensors += [value for pair in output['dino_features'] for value in pair]
                assert all(torch.isfinite(value).all() for value in tensors)
                assert not model.img_backbone.training
                if args.check_grad:
                    #* 只验证梯度连通性，不代表已接入 SSC loss 或 optimizer。
                    probe = feature.square().mean()
                    probe.backward()
                    grads = [p.grad for p in model.image_pyramid.parameters() if p.requires_grad]
                    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
                    assert any(g.abs().sum() > 0 for g in grads)
                    assert all(not p.requires_grad and p.grad is None for p in model.img_backbone.parameters())
                    print('  FPN 梯度有效；FoundationStereo 无梯度。')
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
    print(f'{args.stage} 阶段检查完成；尚未接入体素预测及占据损失。')


if __name__ == '__main__':
    main()
