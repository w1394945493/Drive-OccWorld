#!/usr/bin/env python3
"""FoundationSSC 分阶段完整验证入口。

当前已实现：单帧双目数据接口及 batch_size=1 检查。
后续扩展：图像/体素特征、占据预测、训练损失和评估；目前不加载模型或权重。
"""
import argparse
import sys
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
    parser.add_argument('--config', default='projects/configs/foundationssc/foundationssc_semantic_kitti_data.py')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--ann-file', help='覆盖 PKL 路径；图像/标签路径仍取 PKL')
    parser.add_argument('--indices', nargs='+', type=int, default=[0])
    args = parser.parse_args()
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
    for index, batch in zip(args.indices, loader):
        check_batch(batch)
        print(f"样本 {index}，场景 {batch['img_metas']['scene_name'][0]}，帧 {batch['img_metas']['token'][0]}")
        names = ('归一化图像', 'camera→lidar旋转', 'camera→lidar平移', '4x4内参', '图像变换旋转', '图像变换平移', 'BDA单位阵', 'camera→lidar矩阵')
        for name, tensor in zip(names, batch['img_inputs']):
            print(f'  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}')
        print(f"  occupancy: {tuple(batch['gt_occ'].shape)}；基线(m): {batch['img_metas']['baseline'].item():.6f}")
        print('  投影、归一化、标定和 batch 检查通过。')
    print('第一阶段数据检查完成；尚未执行模型前向。')


if __name__ == '__main__':
    main()
