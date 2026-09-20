#!/usr/bin/env python3
"""FoundationSSC 当前帧无界面可视化；仅保存 pred/GT 并列 PNG 与完整三维 NPZ。"""
import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np


#* 与原 Drive-OccWorld 可视化保持相同 SemanticKITTI 配色。
COLORS = np.array([
    [255, 255, 255], [100, 150, 245], [100, 230, 245], [30, 60, 150],
    [80, 30, 180], [100, 80, 250], [255, 30, 30], [255, 40, 200],
    [150, 30, 90], [255, 0, 255], [255, 150, 255], [75, 0, 75],
    [175, 0, 75], [255, 200, 0], [255, 120, 50], [0, 175, 0],
    [135, 60, 0], [150, 240, 80], [255, 240, 150], [255, 0, 0],
], dtype=np.uint8)


def voxel_to_bev(voxel, empty_idx=0, ignore_idx=255):
    """[X,Y,Z] 沿高度取最高非空、非 ignore 类；全 ignore 柱保留 ignore。"""
    valid = (voxel != empty_idx) & (voxel != ignore_idx)
    top = voxel.shape[-1] - 1 - valid[..., ::-1].argmax(-1)
    bev = np.take_along_axis(voxel, top[..., None], axis=-1)[..., 0].copy()
    bev[~valid.any(-1)] = empty_idx
    bev[(voxel == ignore_idx).all(-1)] = ignore_idx
    return bev


def save_pair(pred, gt, path, empty_idx=0, ignore_idx=255):
    #* Agg 无需桌面/显示器；PNG 为顶视投影，不代表完整三维结构。
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, voxel, title in zip(axes, (pred, gt), ('Pred current', 'GT current')):
        bev = voxel_to_bev(voxel, empty_idx, ignore_idx)
        rgb = np.zeros((*bev.shape, 3), dtype=np.uint8)
        valid = (bev >= 0) & (bev < len(COLORS))
        rgb[valid] = COLORS[bev[valid]]
        rgb[bev == empty_idx] = [255, 255, 255]
        # 全 ignore 柱显示黑色；预测不按 GT mask 裁剪，避免掩盖预测差异。
        rgb[bev == ignore_idx] = [0, 0, 0]
        ax.imshow(rgb.transpose(1, 0, 2), origin='lower')
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout(pad=.5)
    fig.savefig(path, dpi=160, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/configs/foundationssc/foundationssc_semantic_kitti.py')
    parser.add_argument('--checkpoint', required=True, help='训练后的完整模型权重，不能仅提供 FoundationStereo 权重')
    parser.add_argument('--indices', nargs='+', type=int, default=[0], help='过滤后 Dataset 索引，例如 0 10 100')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--ann-file', help='覆盖指定 split 的 PKL')
    parser.add_argument('--out-dir', default='out/foundationssc_vis')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true', help='允许覆盖已有同名 PNG/NPZ')
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from torch.utils.data import DataLoader, Subset
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from projects.mmdet3d_plugin.foundationssc import FoundationSSCImageModel

    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('当前 FoundationSSC CUDA 算子要求使用可用的 CUDA 设备')
    torch.cuda.set_device(device)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    cfg = Config.fromfile(args.config)
    dataset_cfg = copy.deepcopy(cfg.data[args.split])
    dataset_cfg.pop('samples_per_gpu', None)
    dataset_cfg['test_mode'] = True
    if args.ann_file:
        dataset_cfg['ann_file'] = args.ann_file
    #* 单样本普通 DataLoader，不经过 train.py 的 MMCV scatter。
    for step in dataset_cfg['pipeline']:
        if step['type'] == 'PackFoundationSSCInputs':
            step['runner_format'] = False
    dataset = build_dataset(dataset_cfg)
    indices = list(dict.fromkeys(args.indices))
    paths = []
    for index in indices:
        if not 0 <= index < len(dataset):
            raise IndexError(f'索引 {index} 越界，Dataset 长度为 {len(dataset)}')
        info = dataset.data_infos[dataset.valid_indices[index]]
        token = str(info['token'])
        scene = str(info.get('scene_name') or info['scene_token'])
        frame = token.rsplit('_', 1)[-1]
        for name in (scene, frame):
            if name in ('', '.', '..') or '/' in name or '\\' in name:
                raise ValueError(f'非法目录名：{name!r}')
        folder = Path(args.out_dir) / scene / frame
        if not args.overwrite and any((folder / name).exists() for name in ('00_current_pred_gt.png', 'occupancy.npz')):
            raise FileExistsError(f'{folder} 已有结果；换 out-dir 或加 --overwrite')
        paths.append((folder, token, scene))

    options = copy.deepcopy(dict(cfg.model))
    if options.pop('type') != 'FoundationSSCImageModel':
        raise ValueError('此脚本仅支持 FoundationSSCImageModel')
    model = FoundationSSCImageModel(**options)
    #* 严格加载完整 checkpoint，防止漏掉占据头而用随机权重生成可视化。
    # 构造模型仍需配置中的 stereo YAML/预训练路径；随后全部由训练权重覆盖。
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.to(device).eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    for index, batch, (folder, token, scene) in zip(indices, loader, paths):
        with torch.no_grad():
            output = model(return_loss=False, return_outputs=True, **batch)
            pred = output['pred'][0].cpu().numpy().astype(np.uint8)
        gt = batch['gt_occ'][0].numpy().astype(np.uint8)
        if pred.shape != gt.shape:
            raise ValueError(f'预测/GT shape 不一致：{pred.shape} / {gt.shape}')
        del output  # 释放当前帧 GPU logits/特征，再进入下一帧。
        folder.mkdir(parents=True, exist_ok=True)
        #* NPZ 保留原始 [X,Y,Z] 类别数组，GT 255 不变；不是二维投影。
        np.savez_compressed(folder / 'occupancy.npz', pred_occ=pred, gt_occ=gt,
                            token=np.asarray(token), scene_name=np.asarray(scene),
                            dataset_index=np.asarray(index), checkpoint=np.asarray(str(Path(args.checkpoint).resolve())))
        save_pair(pred, gt, folder / '00_current_pred_gt.png',
                  model.pts_bbox_head.empty_idx, model.pts_bbox_head.ignore_index)
        print(f'样本 {index}，{token}，三维 shape={pred.shape}；已保存至 {folder}')
    print(f'可视化完成，共 {len(indices)} 个当前帧。白色=空，GT 全 ignore 柱=黑色。')


if __name__ == '__main__':
    main()
