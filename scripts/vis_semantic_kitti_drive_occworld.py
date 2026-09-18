#!/usr/bin/env python3
"""SemanticKITTI Drive-OccWorld 推理与无界面 occupancy 可视化脚本。

用途：
    1. 读取 SemanticKITTI Drive-OccWorld 配置；
    2. 构建 Dataset / Model；
    3. 可选加载 checkpoint；
    4. 对指定样本执行一次 occupancy forecasting 推理；
    5. 保存预测/GT occupancy 的 BEV top-down 并列对比 PNG 和原始 .npy。

说明：
    - 本脚本不弹出 GUI，matplotlib 使用 Agg 后端，适合服务器环境。
    - 默认仅保存各时间步的 pred-gt 并列对比图，避免输出过多单独图片。
    - 为了拿到 raw occupancy prediction，本脚本复现 drive_occworld.py 中
      forward_test() 的主要 occupancy 前向链路，并在 evaluate_occ() 之前截取
      next_bev_preds。
    - 当前主要服务 SemanticKITTI 第一阶段：turn_on_plan=False,
      turn_on_flow=False。
"""

import argparse
import importlib
import os
import os.path as osp
import sys

# 无界面服务器上 /home 可能不可写；提前指定 matplotlib cache 目录，避免 warning。
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
try:
    from mmcv import Config  # noqa: E402
except ImportError:
    # 兼容少数 mmcv 安装方式；项目训练环境通常走上面的导入。
    from mmcv.utils import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


SEMANTIC_KITTI_COLORS = np.array([
    [255, 255, 255],  # 0 empty
    [100, 150, 245],  # 1 car
    [100, 230, 245],  # 2 bicycle
    [30, 60, 150],    # 3 motorcycle
    [80, 30, 180],    # 4 truck
    [100, 80, 250],   # 5 other-vehicle
    [255, 30, 30],    # 6 person
    [255, 40, 200],   # 7 bicyclist
    [150, 30, 90],    # 8 motorcyclist
    [255, 0, 255],    # 9 road
    [255, 150, 255],  # 10 parking
    [75, 0, 75],      # 11 sidewalk
    [175, 0, 75],     # 12 other-ground
    [255, 200, 0],    # 13 building
    [255, 120, 50],   # 14 fence
    [0, 175, 0],      # 15 vegetation
    [135, 60, 0],     # 16 trunk
    [150, 240, 80],   # 17 terrain
    [255, 240, 150],  # 18 pole
    [255, 0, 0],      # 19 traffic-sign
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='SemanticKITTI Drive-OccWorld 推理与无界面可视化。')
    parser.add_argument(
        '--config',
        default='projects/configs/kitti/semantic_kitti_drive_occworld.py',
        help='配置文件路径。')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='可选 checkpoint 路径；不提供则使用随机/预训练初始化权重。')
    parser.add_argument(
        '--split',
        choices=['train', 'val'],
        default='val',
        help='使用 cfg.data 中的哪个 split。')
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='需要可视化的数据集 index。')
    parser.add_argument(
        '--out-dir',
        default='out/semantic_kitti_drive_occworld_vis',
        help='可视化结果保存目录。')
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='推理设备。')
    parser.add_argument(
        '--save-npy',
        action='store_true',
        help='额外保存 pred_occ.npy / gt_occ.npy。')
    parser.add_argument(
        '--empty-idx',
        type=int,
        default=0,
        help='empty/free 类别 id。SemanticKITTI 默认为 0。')
    return parser.parse_args()


def setup_repo_imports(config_path):
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    cfg = Config.fromfile(config_path)
    if getattr(cfg, 'plugin', False):
        plugin_dir = getattr(cfg, 'plugin_dir', None)
        if plugin_dir is not None:
            module_dir = osp.dirname(plugin_dir).split('/')
            module_path = module_dir[0]
            for part in module_dir[1:]:
                module_path = module_path + '.' + part
            importlib.import_module(module_path)
        else:
            import projects.mmdet3d_plugin  # noqa: F401
    return cfg


def debug_collate(batch):
    """batch_size=1 调试 collate，兼容 MMCV DataContainer。"""
    collated = {}
    first = batch[0]
    for key in first.keys():
        values = [sample[key] for sample in batch]
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
    """转换成 Drive_OccWorld 推理需要的输入格式。"""
    img = to_tensor(batch['img'], device, dtype=torch.float32)
    segmentation = to_tensor(batch['segmentation'], device, dtype=torch.long)
    sdc_planning = to_tensor(batch['sdc_planning'], device, dtype=torch.float32)
    sdc_planning_mask = to_tensor(
        batch['sdc_planning_mask'], device, dtype=torch.float32)
    command = to_tensor(batch['command'], device, dtype=torch.long)
    vel_steering = to_tensor(batch['vel_steering'], device, dtype=torch.float32)

    if segmentation.shape[0] != 1:
        raise AssertionError(
            '当前可视化脚本只支持 batch_size=1；'
            f'got segmentation batch={segmentation.shape[0]}')

    return dict(
        img_metas=batch['img_metas'],
        img=img,
        segmentation=[segmentation[0]],
        sdc_planning=sdc_planning,
        sdc_planning_mask=sdc_planning_mask,
        command=command,
        vel_steering=vel_steering,
        sample_traj=None,
        gt_future_boxes=None,
        flow=None,
        instance=None,
    )


@torch.no_grad()
def inference_occ_logits(model, model_inputs):
    """复现 Drive_OccWorld.forward_test() 的 occupancy 前向并返回 logits/GT。

    Returns:
        occ_logits: [T, num_classes, H_pred, W_pred, D_pred]，T=当前+未来帧。
        occ_gts:    [T, H_gt, W_gt, D_gt]。
        img_metas:  当前参考帧 meta list。
    """
    model.eval()

    img = model_inputs['img']
    img_metas = model_inputs['img_metas']
    segmentation = model_inputs['segmentation']
    sdc_planning = model_inputs['sdc_planning']
    command = model_inputs['command']
    vel_steering = model_inputs['vel_steering']
    sample_traj = model_inputs['sample_traj']

    num_frames = img.size(1)

    #* ================== 1. 历史图像 -> 历史 BEV ==================
    prev_img = img[:, :-1, ...]
    prev_img_metas = [list(each) for each in img_metas]
    prev_bev, prev_bev_list = model.obtain_history_bev(prev_img, prev_img_metas)

    #* ================== 2. 当前图像 -> 当前 ref_bev ==================
    img_cur = img[:, -1, ...]
    img_metas_cur = [each[num_frames - 1] for each in img_metas]
    if model.turn_on_plan:
        # 当前 SemanticKITTI 第一阶段默认 turn_on_plan=False。
        if sample_traj is None:
            raise NotImplementedError(
                'turn_on_plan=True 需要 sample_traj；当前可视化脚本主要用于 '
                'SemanticKITTI 第一阶段 turn_on_plan=False。')
        ref_sample_traj = sample_traj[:, :, 0]
        ref_command = command[:, 0]
        ref_bev, ref_pose_pred, _ = model.obtain_ref_bev_with_plan(
            img_cur, img_metas_cur, prev_bev, ref_sample_traj, None,
            ref_command)
    else:
        ref_bev = model.obtain_ref_bev(img_cur, img_metas_cur, prev_bev)
        ref_pose_pred = None

    #* ================== 3. 自回归未来 BEV/Occupancy 预测 ==================
    prev_bev_list = torch.stack(prev_bev_list, dim=1)
    prev_bev_list = torch.cat(
        [prev_bev_list, ref_bev.unsqueeze(1)], dim=1
    )[:, -model.memory_queue_len:, ...]

    cond_norm_dict = {'occ_gts': None}
    action_condition_dict = {
        'command': command,
        'vel_steering': vel_steering,
    }
    plan_dict = {
        'sem_occupancy': None,
        'sample_traj': sample_traj,
        'gt_traj': sdc_planning,
        'ref_pose_pred': ref_pose_pred,
    }

    next_bev_preds, _, _, _ = model.future_pred(
        prev_bev_list,
        action_condition_dict,
        cond_norm_dict,
        plan_dict,
        valid_frames=[],
        img_metas=img_metas_cur,
        prev_img_metas=prev_img_metas,
        num_frames=num_frames,
        occ_flow='occ')

    # next_bev_preds 原始 shape:
    # [Lout, inter_num, pred_frame_num, B, HW, D_pred, num_cls]
    # 与 evaluate_occ() 保持一致，转成：
    # [inter_num, T*B, num_cls, H_pred, W_pred, D_pred]
    occ_preds = next_bev_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
    inter_num, select_frames, bs, num_cls, hw, d = occ_preds.shape
    occ_preds = occ_preds.view(
        inter_num, select_frames * bs, num_cls, model.bev_w, model.bev_h, d
    ).transpose(3, 4)

    # 只取最后一层 WorldDecoder intermediate 作为可视化预测。
    occ_logits = occ_preds[-1]
    occ_gts = segmentation[0][model.future_pred_head.history_queue_length:]
    occ_gts = occ_gts.view(select_frames * bs, *occ_gts.shape[-3:])
    return occ_logits, occ_gts, img_metas_cur


def logits_to_prediction(occ_logits, gt_shape):
    """将低分辨率 logits 插值到 GT 尺寸并 argmax。"""
    _, h, w, d = gt_shape
    occ_logits = F.interpolate(
        occ_logits, size=(h, w, d), mode='trilinear',
        align_corners=False).contiguous()
    return torch.argmax(occ_logits, dim=1)


def voxel_to_bev(voxel, empty_idx=0):
    """3D voxel -> 2D BEV。

    对每个 BEV 网格柱，如果任意高度有非 empty/非 ignore 类别，则取最高处
    第一个非 empty 类别作为 BEV 类别；否则为 empty。
    """
    voxel = np.asarray(voxel)
    h, w, d = voxel.shape
    bev = np.full((h, w), empty_idx, dtype=np.int64)
    valid = (voxel != empty_idx) & (voxel != 255)
    has_occ = valid.any(axis=2)
    # 从高到低找第一个非空类别，视觉上更接近 top-down 表面。
    reversed_valid = valid[:, :, ::-1]
    top_from_reversed = reversed_valid.argmax(axis=2)
    top_z = d - 1 - top_from_reversed
    xs, ys = np.where(has_occ)
    bev[xs, ys] = voxel[xs, ys, top_z[xs, ys]]
    return bev


def colorize_bev(bev, empty_idx=0):
    colors = SEMANTIC_KITTI_COLORS
    out = np.zeros((*bev.shape, 3), dtype=np.uint8)
    out[:] = colors[empty_idx]
    valid = (bev >= 0) & (bev < len(colors))
    out[valid] = colors[bev[valid]]
    out[bev == 255] = np.array([0, 0, 0], dtype=np.uint8)
    return out


def save_pair_png(pred_bev, gt_bev, path, step_name, empty_idx=0):
    pred_rgb = colorize_bev(pred_bev, empty_idx=empty_idx)
    gt_rgb = colorize_bev(gt_bev, empty_idx=empty_idx)
    os.makedirs(osp.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.transpose(pred_rgb, (1, 0, 2)), origin='lower')
    axes[0].set_title(f'Pred {step_name}')
    axes[0].axis('off')
    axes[1].imshow(np.transpose(gt_rgb, (1, 0, 2)), origin='lower')
    axes[1].set_title(f'GT {step_name}')
    axes[1].axis('off')
    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=160, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


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

    subset = torch.utils.data.Subset(dataset, [args.index])
    dataloader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=debug_collate)
    batch = next(iter(dataloader))
    model_inputs = batch_to_model_inputs(batch, device)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    if args.checkpoint is not None:
        print(f'加载 checkpoint: {args.checkpoint}')
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.to(device)
    model.eval()
    if hasattr(model, 'set_epoch'):
        model.set_epoch(0)

    token = batch.get('current_token', [f'index_{args.index}'])[0]
    out_dir = osp.join(args.out_dir, str(token))
    os.makedirs(out_dir, exist_ok=True)

    print(f'开始推理 index={args.index}, token={token}, device={device}')
    occ_logits, occ_gts, _ = inference_occ_logits(model, model_inputs)
    pred_occ = logits_to_prediction(occ_logits, occ_gts.shape)

    pred_np = pred_occ.detach().cpu().numpy().astype(np.uint8)
    gt_np = occ_gts.detach().cpu().numpy().astype(np.uint8)

    if args.save_npy:
        np.save(osp.join(out_dir, 'pred_occ.npy'), pred_np)
        np.save(osp.join(out_dir, 'gt_occ.npy'), gt_np)

    num_steps = pred_np.shape[0]
    for step_idx in range(num_steps):
        step_name = 'current' if step_idx == 0 else f'future_{step_idx}'
        pred_bev = voxel_to_bev(pred_np[step_idx], empty_idx=args.empty_idx)
        gt_bev = voxel_to_bev(gt_np[step_idx], empty_idx=args.empty_idx)

        #* 仅保存 pred-gt 并列对比图。
        #* 单独 pred.png / gt.png 容易造成输出文件过多，当前调试主要看对比效果。
        save_pair_png(
            pred_bev,
            gt_bev,
            osp.join(out_dir, f'{step_idx:02d}_{step_name}_pred_gt.png'),
            step_name=step_name,
            empty_idx=args.empty_idx)

    print('可视化完成。输出目录:')
    print(f'  {out_dir}')
    print('主要文件:')
    print('  00_current_pred_gt.png')
    if num_steps > 1:
        print(f'  01_future_1_pred_gt.png ... {num_steps - 1:02d}_future_{num_steps - 1}_pred_gt.png')


if __name__ == '__main__':
    main()
