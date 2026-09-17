#!/usr/bin/env python3
"""最小化测试 SemanticKITTI Drive-OccWorld forward_train 链路。

这个脚本不启动 runner，也不做优化器更新，只做：

1. 读取 ``projects/configs/kitti/semantic_kitti_drive_occworld.py``；
2. build_dataset / build_model；
3. 从 Dataset 取 batch_size=1 的一个 batch；
4. numpy -> torch；
5. 调用 ``model.forward_train(...)``。

用途：
    用最小成本暴露字段缺失、shape 不匹配、单目相机输入兼容性、
    occupancy 尺寸/类别不匹配等问题。
"""

import argparse
import importlib
import os.path as osp
import sys
import traceback

import numpy as np
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description='测试 SemanticKITTI Drive-OccWorld forward_train。')
    parser.add_argument(
        '--config',
        default='projects/configs/kitti/semantic_kitti_drive_occworld.py',
        help='SemanticKITTI Drive-OccWorld 配置文件。')
    parser.add_argument(
        '--split',
        choices=['train', 'val'],
        default='train',
        help='使用 cfg.data 中的哪个 split。')
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='调试用样本 index；脚本会从该位置附近取一个 batch。')
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='forward 使用的设备。默认 cuda；无 GPU 时可用 cpu 先检查构图。')
    parser.add_argument(
        '--no-init-weights',
        action='store_true',
        help='跳过 model.init_weights()，更快暴露输入/结构问题。')
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
            # 与 tools/train.py 保持一致：plugin_dir='projects/mmdet3d_plugin/'
            # 实际 import 的是 projects.mmdet3d_plugin。
            module_dir = osp.dirname(plugin_dir).split('/')
            module_path = module_dir[0]
            for part in module_dir[1:]:
                module_path = module_path + '.' + part
            importlib.import_module(module_path)
        else:
            import projects.mmdet3d_plugin  # noqa: F401
    return cfg


def debug_collate(batch):
    """把 Dataset 输出整理成 forward_train 更容易使用的 batch。

    当前只测试 batch_size=1；复杂字段如 img_metas 保留为 list。
    """
    collated = {}
    first = batch[0]
    for key in first.keys():
        values = [sample[key] for sample in batch]
        if key in (
            'img',
            'segmentation',
            'sdc_planning',
            'sdc_planning_mask',
            'command',
            'vel_steering',
        ) and values[0] is not None:
            collated[key] = np.stack(values, axis=0)
        else:
            collated[key] = values
    return collated


def to_tensor(value, device, dtype=None):
    tensor = torch.from_numpy(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device)


def print_batch_summary(batch):
    print('\n' + '-' * 100)
    print('Batch 输入摘要')
    print('-' * 100)
    for key in ('img', 'segmentation', 'sdc_planning',
                'sdc_planning_mask', 'command', 'vel_steering'):
        value = batch.get(key, None)
        if value is None:
            print(f'{key}: None')
        else:
            print(f'{key}: shape={value.shape}, dtype={value.dtype}')
    img_metas = batch.get('img_metas', None)
    if img_metas is not None:
        print(f'img_metas: batch={len(img_metas)}, T_input={len(img_metas[0])}')
        print(f"  当前帧 token: {img_metas[0][-1].get('token')}")
        print(f"  当前帧 lidar2img num: {len(img_metas[0][-1].get('lidar2img'))}")
        print(
            '  当前帧 lidar2global_rotation shape: '
            f"{np.asarray(img_metas[0][-1].get('lidar2global_rotation')).shape}")
        print(
            '  当前帧 can_bus[-1] 类型/值: '
            f"{type(img_metas[0][-1].get('can_bus')[-1]).__name__}/"
            f"{img_metas[0][-1].get('can_bus')[-1]}")
        print(
            '  当前帧 future2ref_lidar_transform shape: '
            f"{np.asarray(img_metas[0][-1].get('future2ref_lidar_transform')).shape}")
    print(f"current_token: {batch.get('current_token')}")
    print(f"window_tokens[0]: {batch.get('window_tokens', [''])[0]}")


def main():
    args = parse_args()
    cfg = setup_repo_imports(args.config)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('未检测到 CUDA，自动切换到 CPU。')
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    dataset_cfg = cfg.data[args.split]
    dataset = build_dataset(dataset_cfg)
    if not (0 <= args.index < len(dataset)):
        raise IndexError(
            f'index={args.index} out of range [0, {len(dataset) - 1}]')

    # 用 Subset 避免 DataLoader 从 0 开始时无法指定调试 index。
    subset = torch.utils.data.Subset(dataset, [args.index])
    dataloader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=debug_collate)
    batch = next(iter(dataloader))
    print_batch_summary(batch)

    print('\n' + '-' * 100)
    print('构建模型')
    print('-' * 100)
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if not args.no_init_weights:
        model.init_weights()
    model.to(device)
    model.train()
    if hasattr(model, 'set_epoch'):
        # drive_occworld.py 中 training_epoch 会影响 sem_gt_train / planner 分支。
        model.set_epoch(0)
    print(f'模型类型: {model.__class__.__name__}')
    print(f'设备: {device}')

    #* ================== numpy batch -> torch forward_train 输入 ==================
    img = to_tensor(batch['img'], device, dtype=torch.float32)
    segmentation = to_tensor(batch['segmentation'], device, dtype=torch.long)
    sdc_planning = to_tensor(batch['sdc_planning'], device, dtype=torch.float32)
    sdc_planning_mask = to_tensor(
        batch['sdc_planning_mask'], device, dtype=torch.float32)
    command = to_tensor(batch['command'], device, dtype=torch.long)
    vel_steering = to_tensor(batch['vel_steering'], device, dtype=torch.float32)

    # Dataset/DataLoader 保持结构为 img_metas[batch_idx][time_idx]，
    # 正好和 Drive_OccWorld.forward_train 中的访问方式一致：
    #   img_metas = [each[num_frames-1] for each in img_metas]
    img_metas = batch['img_metas']

    print('\n' + '-' * 100)
    print('开始 forward_train')
    print('-' * 100)
    try:
        with torch.set_grad_enabled(True):
            losses = model.forward_train(
                img_metas=img_metas,
                img=img,
                segmentation=[segmentation],
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command,
                vel_steering=vel_steering,
                # 第一阶段关闭 turn_on_plan/turn_on_flow，下列字段不应被使用。
                sample_traj=None,
                gt_future_boxes=None,
                flow=None,
                instance=None,
            )
        print('\nforward_train 成功。losses:')
        for key, value in losses.items():
            if torch.is_tensor(value):
                print(f'  {key}: shape={tuple(value.shape)}, '
                      f'value={float(value.detach().cpu()):.6f}')
            else:
                print(f'  {key}: {value}')
    except Exception:
        print('\nforward_train 失败，完整 traceback 如下：')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
