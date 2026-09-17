#!/usr/bin/env python3
"""最小化测试 SemanticKITTI Drive-OccWorld forward_train / train step 链路。

默认不启动 runner，也不做优化器更新，只做：

1. 读取 ``projects/configs/kitti/semantic_kitti_drive_occworld.py``；
2. build_dataset / build_model；
3. 从 Dataset 取 batch_size=1 的一个 batch；
4. numpy -> torch；
5. 调用 ``model.forward_train(...)``。

如果传入 ``--train-iters N`` 且 N > 0，则会额外执行 N 次最小训练 step：

    forward_train -> loss 求和 -> backward -> optimizer.step

用于验证反向传播、参数更新和显存占用是否能跑通。

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
    parser.add_argument(
        '--train-iters',
        type=int,
        default=0,
        help='执行多少个最小训练 step。默认 0 表示只做一次 forward_train。')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='--train-iters > 0 时使用的 AdamW 学习率。')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='--train-iters > 0 时使用的 AdamW weight decay。')
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
        # 兼容 Dataset format_for_train=True 时返回的 MMCV DataContainer。
        # DataContainer.data 才是真正的 tensor/list[dict]。
        if hasattr(values[0], 'data'):
            values = [value.data for value in values]
        if key in (
            'img',
            'segmentation',
            'sdc_planning',
            'sdc_planning_mask',
            'command',
            'vel_steering',
        ) and values[0] is not None:
            if torch.is_tensor(values[0]):
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = np.stack(values, axis=0)
        else:
            collated[key] = values
    return collated


def to_tensor(value, device, dtype=None):
    if torch.is_tensor(value):
        tensor = value
    else:
        tensor = torch.from_numpy(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device)


def batch_to_model_inputs(batch, device):
    """将 debug_collate 后的 numpy batch 转成 Drive_OccWorld 输入。

    #! 当前阶段只支持 batch_size=1。
    Drive-OccWorld 原始 compute_occ_loss 默认 ``segmentation[0]`` 是
    [T,H,W,D] 的时序 occupancy，而不是 [B,T,H,W,D]。因此这里把
    DataLoader 得到的 [1,T,H,W,D] 转成 list([T,H,W,D])，用于跑通
    SemanticKITTI 第一阶段链路。
    """
    img = to_tensor(batch['img'], device, dtype=torch.float32)
    segmentation = to_tensor(batch['segmentation'], device, dtype=torch.long)
    sdc_planning = to_tensor(batch['sdc_planning'], device, dtype=torch.float32)
    sdc_planning_mask = to_tensor(
        batch['sdc_planning_mask'], device, dtype=torch.float32)
    command = to_tensor(batch['command'], device, dtype=torch.long)
    vel_steering = to_tensor(batch['vel_steering'], device, dtype=torch.float32)

    if segmentation.shape[0] != 1:
        raise AssertionError(
            '当前 forward/train 调试脚本只支持 batch_size=1；'
            f'got segmentation batch={segmentation.shape[0]}')

    return dict(
        img_metas=batch['img_metas'],
        img=img,
        segmentation=[segmentation[0]],
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


def summarize_losses(losses):
    """把 loss dict 中的 tensor loss 求和，并返回便于打印的标量字典。"""
    total_loss = None
    loss_scalars = {}
    for key, value in losses.items():
        if not torch.is_tensor(value):
            continue
        loss_value = value.mean()
        total_loss = loss_value if total_loss is None else total_loss + loss_value
        loss_scalars[key] = float(loss_value.detach().cpu())

    if total_loss is None:
        raise RuntimeError('forward_train 没有返回可反传的 tensor loss。')

    loss_scalars['loss_total'] = float(total_loss.detach().cpu())
    return total_loss, loss_scalars


def print_cuda_memory(prefix, device):
    """打印 CUDA 显存摘要；CPU 模式下自动跳过。"""
    if device.type != 'cuda':
        return
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    print(
        f'{prefix} CUDA 显存: '
        f'allocated={allocated:.2f} GB, '
        f'reserved={reserved:.2f} GB, '
        f'max_allocated={max_allocated:.2f} GB')


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
    #* train-iters=0 时只取一个样本做 forward；
    #* train-iters>0 时取从 index 开始的连续若干样本，验证
    #* forward -> backward -> optimizer.step 的最小训练闭环。
    if args.train_iters > 0:
        end_index = min(args.index + args.train_iters, len(dataset))
        subset_indices = list(range(args.index, end_index))
        if len(subset_indices) < args.train_iters:
            print(
                f'警告：从 index={args.index} 到数据集末尾只剩 '
                f'{len(subset_indices)} 个样本，少于 train-iters={args.train_iters}。')
    else:
        subset_indices = [args.index]
    subset = torch.utils.data.Subset(dataset, subset_indices)
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

    if args.train_iters <= 0:
        print('\n' + '-' * 100)
        print('开始 forward_train')
        print('-' * 100)
        try:
            with torch.set_grad_enabled(True):
                model_inputs = batch_to_model_inputs(batch, device)
                losses = model.forward_train(**model_inputs)
            print('\nforward_train 成功。losses:')
            for key, value in losses.items():
                if torch.is_tensor(value):
                    print(f'  {key}: shape={tuple(value.shape)}, '
                          f'value={float(value.detach().cpu()):.6f}')
                else:
                    print(f'  {key}: {value}')
            print_cuda_memory('forward_train 后', device)
        except Exception:
            print('\nforward_train 失败，完整 traceback 如下：')
            traceback.print_exc()
            raise
        return

    print('\n' + '-' * 100)
    print(f'开始最小训练闭环测试：train-iters={args.train_iters}')
    print('-' * 100)
    #! 这里不是正式训练配置，只是为了验证 backward/optimizer.step 能否跑通。
    #! 正式训练仍应使用 tools/train.py 和完整 optimizer/lr_config。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
    print(f'调试优化器: AdamW(lr={args.lr}, weight_decay={args.weight_decay})')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for iter_idx, train_batch in enumerate(dataloader, start=1):
            optimizer.zero_grad(set_to_none=True)
            model_inputs = batch_to_model_inputs(train_batch, device)
            losses = model.forward_train(**model_inputs)
            total_loss, loss_scalars = summarize_losses(losses)

            if not torch.isfinite(total_loss):
                raise RuntimeError(
                    f'iter {iter_idx}: total_loss 非有限值: '
                    f'{float(total_loss.detach().cpu())}')

            total_loss.backward()
            optimizer.step()

            loss_text = ', '.join(
                f'{key}={value:.4f}' for key, value in loss_scalars.items())
            print(f'iter {iter_idx:03d}/{len(dataloader):03d}: {loss_text}')
            print_cuda_memory(f'iter {iter_idx:03d} 后', device)

        print('\n最小训练闭环测试成功：forward/backward/optimizer.step 均已跑通。')
    except Exception:
        print('\n最小训练闭环测试失败，完整 traceback 如下：')
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
