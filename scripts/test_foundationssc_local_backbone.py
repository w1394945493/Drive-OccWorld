#!/usr/bin/env python3
"""CPU 冒烟测试：验证本地骨干自包含，不需要数据集、MMCV 或预训练权重。"""
import importlib.util
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf


def main():
    #* 单独加载本仓库 stereo 包，避免插件入口额外要求 MMCV/CUDA 扩展。
    root = Path(__file__).resolve().parents[1] / 'projects/mmdet3d_plugin/foundationssc/stereo'
    spec = importlib.util.spec_from_file_location(
        '_local_stereo_test', root / '__init__.py', submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from _local_stereo_test.core.foundation_stereo import FoundationStereo

    torch.set_num_threads(2)
    torch.manual_seed(0)
    #* 小模型/小图只检查接口；不是正式配置，不用于评价准确率。
    args = OmegaConf.create(dict(
        hidden_dims=[128, 128, 128], n_downsample=2, n_gru_layers=3,
        corr_radius=4, corr_levels=2, max_disp=32,
        vit_size='vits', mixed_precision=False))
    model = FoundationStereo(args).eval()
    with torch.no_grad():
        disparity, features = model(
            torch.rand(1, 3, 64, 128) * 255,
            torch.rand(1, 3, 64, 128) * 255, iters=1, test_mode=True)
    assert disparity[1].shape == (1, 1, 64, 128)
    assert len(features) == 4
    for feature, cls_token in features:
        assert feature.shape == (2, 384, 4, 8)  # 左右图像沿 batch 拼接。
        assert cls_token.shape == (2, 384)
    tensors = [*disparity, *(x for pair in features for x in pair)]
    assert all(torch.isfinite(x).all() for x in tensors)
    for name, module in list(sys.modules.items()):
        if name.startswith('_local_stereo_test') and getattr(module, '__file__', None):
            assert Path(module.__file__).resolve().is_relative_to(root)
    print('本地骨干 CPU 前向通过：未使用 FoundationSSC 仓库、数据集或权重。')
    print('视差输出：', [tuple(x.shape) for x in disparity])
    print('四层 DINO 特征：', [tuple(x[0].shape) for x in features])


if __name__ == '__main__':
    main()
