"""四步预测验证：真实数据/权重 → 前向/评估，可选反向；无需 GUI。"""
import argparse
import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'projects/configs/foundationssc_forecasting/foundationssc_forecast.py'))
    parser.add_argument('--checkpoint', default=None,
                        help='完整 FoundationSSC checkpoint；优先于配置 load_from，未传时使用 load_from')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--check-grad', action='store_true')
    args = parser.parse_args()
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet.models import build_detector
    importlib.import_module('projects.mmdet3d_plugin')
    cfg = Config.fromfile(args.config)
    #! 与正式训练共享 load_from；命令行仅用于临时覆盖，不必重复填写权重路径。
    checkpoint_path = args.checkpoint or cfg.get('load_from')
    if not checkpoint_path:
        parser.error('请在配置中设置 load_from，或通过 --checkpoint 指定完整 FoundationSSC 检查点')
    print(f'加载权重（{"命令行 --checkpoint" if args.checkpoint else "配置 load_from"}）：{checkpoint_path}')
    for name in cfg.custom_imports.imports:
        importlib.import_module(name)
    dataset = build_dataset(cfg.data.val)
    batch = scatter(collate([dataset[args.index]], samples_per_gpu=1), [0])[0]
    model = build_detector(cfg.model).cuda()
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu', strict=False)
    #! 单帧模型允许仅缺 dynamics；其他缺失/尺寸错误不能视为成功加载基线。
    state = checkpoint.get('state_dict', checkpoint)
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    missing = [key for key in model.state_dict() if key not in state and not key.startswith('dynamics.')]
    if missing:
        raise RuntimeError(f'单帧基线权重缺失：{missing}')
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        #! 已切换为真实未来位姿条件注意力；预测和 GT 均处于各帧自己的坐标系。
        result = model(return_loss=False, **batch)
        assert len(result) == 1 and len(result[0]['hist_for_iou_per_frame']) == 5
        print('四步位姿条件预测（未训练 dynamics 时仅验证接口）')
        dataset.evaluate(result)
    if args.check_grad:
        model.train()
        losses = model(return_loss=True, **batch)
        total = sum(losses.values())
        if not torch.isfinite(total):
            raise AssertionError('损失非有限值')
        total.backward()
        grads = [p.grad for p in model.dynamics.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
        #! 跟随解冻配置检查：冻结参数无梯度，各解冻模块应有有效反向梯度。
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        for name in ('image_pyramid', 'voxel_encoder', 'occ_encoder_backbone', 'occ_encoder_neck', 'pts_bbox_head'):
            params = [p for p in getattr(model, name).parameters() if p.requires_grad]
            if params:
                active = [p.grad for p in params if p.grad is not None]
                assert active and all(torch.isfinite(g).all() for g in active), f'{name} 梯度异常'
                assert any(g.abs().sum() > 0 for g in active), f'{name} 无有效梯度'
        print(f'四步损失={total.item():.6f}；梯度检查通过；freeze_frontend={model.freeze_frontend}，freeze_decoder={model.freeze_decoder}')
    print(f'显存峰值：{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')


if __name__ == '__main__':
    main()
