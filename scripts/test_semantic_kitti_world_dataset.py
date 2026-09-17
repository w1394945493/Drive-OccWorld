#!/usr/bin/env python3
"""根据配置实例化 SemanticKITTIWorldDataset，并打印一帧样本信息。

示例：
    python scripts/test_semantic_kitti_world_dataset.py \
        --config projects/configs/kitti/semantic_kitti_world_dataset.py \
        --split train \
        --index 0

    python scripts/test_semantic_kitti_world_dataset.py \
        --config projects/configs/kitti/semantic_kitti_world_dataset.py \
        --split val \
        --index 0
"""

import argparse
import os
import os.path as osp
import sys

import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='检查 SemanticKITTIWorldDataset 是否能正常读取样本。')
    parser.add_argument(
        '--config',
        default='projects/configs/kitti/semantic_kitti_world_dataset.py',
        help='数据集配置文件路径。')
    parser.add_argument(
        '--split',
        choices=['train', 'val'],
        default='train',
        help='实例化 cfg.data 中的哪个 split。')
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='过滤无效边界样本后的 dataset index。')
    return parser.parse_args()


def setup_repo_imports(config_path):
    """Add repo root to sys.path and import plugin modules for registration."""
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    cfg = Config.fromfile(config_path)
    if getattr(cfg, 'plugin', False):
        plugin_dir = getattr(cfg, 'plugin_dir', None)
        if plugin_dir is not None:
            plugin_path = osp.abspath(osp.join(repo_root, plugin_dir))
            if plugin_path not in sys.path:
                sys.path.insert(0, plugin_path)
        # 导入插件包会触发 DATASETS.register_module()。
        import projects.mmdet3d_plugin  # noqa: F401
    return cfg


def summarize_frame_input(frame_input, prefix):
    print(f'\n{prefix}')
    print(f"  当前帧 token: {frame_input.get('token')}")
    print(f"  sample_idx: {frame_input.get('sample_idx')}")
    print(f"  所属 sequence/scene_token: {frame_input.get('scene_token')}")
    print(f"  上一帧 prev_idx: {frame_input.get('prev_idx')}")
    print(f"  下一帧 next_idx: {frame_input.get('next_idx')}")
    print(f"  occupancy 标注路径 occ_path: {frame_input.get('occ_path')}")
    print(f"  图像数量 image num: {len(frame_input.get('img_filename', []))}")
    for i, img_path in enumerate(frame_input.get('img_filename', [])):
        print(f"    图像 image[{i}]: {img_path}")
    lidar2img = frame_input.get('lidar2img', [])
    print(f"  lidar2img 数量: {len(lidar2img)}")
    if lidar2img:
        print(f"    lidar2img[0] shape: {np.asarray(lidar2img[0]).shape}")


def main():
    args = parse_args()
    cfg = setup_repo_imports(args.config)

    dataset_cfg = cfg.data[args.split]
    dataset = build_dataset(dataset_cfg)

    print('=' * 100)
    print('SemanticKITTIWorldDataset 读取检查')
    print('=' * 100)
    print(f'配置文件 config: {osp.abspath(args.config)}')
    print(f'数据划分 split: {args.split}')
    print(f'标注 pkl ann_file: {dataset.ann_file}')
    print(f'相机模式 use_camera: {dataset.use_camera}')
    print(f'实际使用相机 camera_names: {dataset.camera_names}')
    print(f'历史帧数量 history_queue_length: {dataset.history_queue_length}')
    print(f'未来帧数量 future_queue_length: {dataset.future_queue_length}')
    print(f'图像 pad_shape: {dataset.pad_shape}')
    print(f'图像 size_divisor: {dataset.size_divisor}')
    print(f'图像归一化 img_norm_cfg: {dataset.img_norm_cfg}')
    print(f'pkl 原始样本数 raw infos length: {len(dataset.data_infos)}')
    print(f'过滤边界后的有效样本数 valid dataset length: {len(dataset)}')

    if not (0 <= args.index < len(dataset)):
        raise IndexError(
            f'index={args.index} out of range [0, {len(dataset) - 1}]')

    data = dataset[args.index]
    print('\n' + '-' * 100)
    print('时序窗口信息 Window summary')
    print('-' * 100)
    print(f"当前参考帧 current_token: {data['current_token']}")
    print(f"窗口内所有 token window_tokens: {data['window_tokens']}")
    print(f"窗口帧数量 num frame_inputs: {len(data['frame_inputs'])}")

    img = data.get('img', None)
    if img is None:
        print('图像队列 img: None')
    else:
        print(
            f'图像队列 img: shape={img.shape}, dtype={img.dtype} '
            '[T_input, N_cam, C, H, W]')
        print(
            f'  图像数值范围: min={float(img.min()):.3f}, '
            f'max={float(img.max()):.3f}, mean={float(img.mean()):.3f}')

    img_metas = data.get('img_metas', None)
    if img_metas is None:
        print('图像 meta 队列 img_metas: None')
    else:
        print(f'图像 meta 队列 img_metas: len={len(img_metas)}')
        if img_metas:
            print(f"  第一帧 meta token: {img_metas[0].get('token')}")
            print(f"  当前帧 meta token: {img_metas[-1].get('token')}")
            print(f"  当前帧 ori_shape: {img_metas[-1].get('ori_shape')}")
            print(f"  当前帧 img_shape: {img_metas[-1].get('img_shape')}")
            print(f"  当前帧 pad_shape: {img_metas[-1].get('pad_shape')}")
            print(
                '  当前帧 ref_lidar_to_cur_lidar shape: '
                f"{np.asarray(img_metas[-1].get('ref_lidar_to_cur_lidar')).shape}")

    segmentation = data.get('segmentation', None)
    if segmentation is None:
        print('occupancy 序列 segmentation: None')
    else:
        print(
            f'occupancy 序列 segmentation: shape={segmentation.shape}, '
            f'dtype={segmentation.dtype}, '
            f'min={segmentation.min()}, max={segmentation.max()}')

    summarize_frame_input(data['frame_inputs'][0], '最早历史帧 First history frame')
    summarize_frame_input(
        data['frame_inputs'][dataset.history_queue_length], '当前参考帧 Current frame')
    summarize_frame_input(data['frame_inputs'][-1], '最后未来帧 Last future frame')

    print('\n检查完成。')


if __name__ == '__main__':
    main()
