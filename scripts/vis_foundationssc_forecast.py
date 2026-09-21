#!/usr/bin/env python3
"""位姿条件四步预测无界面可视化：逐样本保存各步 pred/GT 并列 PNG 和三维 NPZ。"""
import argparse
import copy
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'projects/configs/foundationssc_forcesting/foundationssc_forecast.py'))
    parser.add_argument('--checkpoint', help='优先于配置 load_from；可用单帧权重检查接口，正式预测应加载训练后的 forecasting 权重')
    parser.add_argument('--indices', nargs='+', type=int, default=[0], help='过滤后 Dataset 索引，例如 0 10 100')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--ann-file', help='覆盖指定 split 的 PKL 路径')
    parser.add_argument('--out-dir', default='out/foundationssc_forecast_vis')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true', help='允许覆盖已有同名 PNG/NPZ')
    args = parser.parse_args()

    import torch
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet.models import build_detector
    #! 复用单帧绘图：同一 SemanticKITTI 配色，GT 全 ignore 柱显示黑色。
    from vis_foundationssc import save_pair

    cfg = Config.fromfile(args.config)
    checkpoint_path = args.checkpoint or cfg.get('load_from')
    if not checkpoint_path:
        parser.error('请设置配置 load_from 或传入 --checkpoint')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('当前 FoundationSSC 算子要求可用的 CUDA 设备')
    torch.cuda.set_device(device)
    device_id = torch.cuda.current_device()
    importlib.import_module('projects.mmdet3d_plugin')
    for module in cfg.custom_imports.imports:
        importlib.import_module(module)

    dataset_cfg = copy.deepcopy(cfg.data[args.split])
    dataset_cfg.pop('samples_per_gpu', None)
    dataset_cfg['test_mode'] = True
    if args.ann_file:
        dataset_cfg['ann_file'] = args.ann_file
    dataset = build_dataset(dataset_cfg)
    indices = list(dict.fromkeys(args.indices))
    steps = int(cfg.model.get('future_steps', 4))
    names = ['current'] + [f'future_{step}' for step in range(1, steps + 1)]
    targets = []
    #! 提前检查全部索引和输出路径，默认不覆盖已有可视化结果。
    for index in indices:
        if not 0 <= index < len(dataset):
            raise IndexError(f'索引 {index} 超出 [0,{len(dataset)-1}]')
        info = dataset.data_infos[dataset.valid_indices[index]]
        token = str(info['token'])
        scene = str(info.get('scene_name') or info['scene_token'])
        frame = token.rsplit('_', 1)[-1]
        if any(name in ('', '.', '..') or '/' in name or '\\' in name for name in (scene, frame)):
            raise ValueError(f'非法场景/帧目录：{scene}/{frame}')
        out = Path(args.out_dir) / scene / frame
        paths = [out / 'occupancy.npz'] + [out / f'{step:02d}_{name}_pred_gt.png' for step, name in enumerate(names)]
        if not args.overwrite and any(path.exists() for path in paths):
            raise FileExistsError(f'{out} 已有结果；换输出目录或使用 --overwrite')
        targets.append((index, token, scene, out))

    model = build_detector(cfg.model).to(device)
    print(f'加载完整模型权重：{checkpoint_path}')
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu', strict=False)
    weights = checkpoint.get('state_dict', checkpoint)
    weights = {key.removeprefix('module.'): value for key, value in weights.items()}
    missing = []
    for key, value in model.state_dict().items():
        if key not in weights:
            missing.append(key)
        elif weights[key].shape != value.shape:
            raise ValueError(f'权重尺寸不匹配：{key}；请检查 Small/Large 及预测器配置')
    if any(not key.startswith('dynamics.') for key in missing):
        raise RuntimeError(f'感知模型权重不完整：{missing}')
    if missing:
        print('注意：预测器权重不完整，缺失参数使用初始化值；结果仅用于接口检查。')
    del checkpoint, weights
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    for count, (index, token, scene, out) in enumerate(targets, 1):
        print(f'[{count}/{len(targets)}] index={index}，场景={scene}，参考帧={token}')
        #! 与已验证的测试脚本一致，标准 MMCV collate/scatter 保留模型输入协议。
        batch = scatter(collate([dataset[index]], samples_per_gpu=1), [device_id])[0]
        with torch.no_grad():
            output = model(return_loss=False, return_outputs=True, **batch)
        pred = output['pred'][0].cpu().numpy().astype(np.uint8)
        gt = batch['gt_occ'][0].cpu().numpy().astype(np.uint8)
        if pred.shape != gt.shape or pred.shape[0] != steps + 1:
            raise ValueError(f'预测/标签时间空间尺寸不匹配：{pred.shape}/{gt.shape}')
        del output, batch
        out.mkdir(parents=True, exist_ok=True)
        #! 完整三维数组 [当前+未来,X,Y,Z]；各步在各自 LiDAR 坐标系，不是共同参考系。
        np.savez_compressed(out / 'occupancy.npz', pred_occ=pred, gt_occ=gt,
                            token=np.asarray(token), scene_name=np.asarray(scene),
                            dataset_index=np.asarray(index), step_names=np.asarray(names),
                            coordinate=np.asarray('per_frame_lidar'),
                            pose_condition=np.asarray('ground_truth_future_pose'))
        for step, name in enumerate(names):
            save_pair(pred[step], gt[step], out / f'{step:02d}_{name}_pred_gt.png', step_name=name)
        print(f'  已保存并列对比图和 occupancy.npz：{out}')
    print(f'完成，共 {len(targets)} 个样本；显存峰值 {torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB')


if __name__ == '__main__':
    main()
