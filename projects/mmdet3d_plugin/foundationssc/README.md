# FoundationSSC 分阶段移植

目前实现图像前端，不是完整 SSC 模型，不能用于正式 train.py 训练。

- 数据：当前帧左右 RGB、标定、occupancy，全部通过现有 PKL 构造。
- 冻结骨干：本地 `stereo/` 包实现 FoundationStereo、DepthAnything 和 DINOv2。
  返回 `([prob, disp_up], dinov2_features)`，不需要 FoundationSSC 仓库。
- 可训练适配：等价实现 SimpleFPN 默认 `layers=[4]` 和 SECONDFPN，输出 640 通道。
- `forward_test` 返回特征用于检查；`forward_train` 暂明确报未实现，避免把调试目标当训练损失。

从 Drive-OccWorld 根目录运行：

```bash
python scripts/test_foundationssc.py --stage images --indices 0
python scripts/test_foundationssc.py --stage images --indices 0 --check-grad
```

无需数据/权重的本地骨干 CPU 冒烟测试（随机初始化 ViT-S 小输入）：
`python scripts/test_foundationssc_local_backbone.py`。

数据和模型配置已合并到 `projects/configs/foundationssc/foundationssc_semantic_kitti.py`。
路径可通过 `--stereo-checkpoint`、`--stereo-config`、`--ann-file` 覆盖。
仍需 PyTorch、torchvision、timm（支持 edgenext_small）、OmegaConf、einops、SciPy
等第三方库，以及完整 FoundationStereo 权重和其 YAML 配置；不自动下载依赖或权重。
构造网络不再读取单独的 EdgeNeXt/DINO 权重；外层 checkpoint 必须覆盖全部骨干参数。
原 FoundationSSC 仓库可以不在运行机器上，且没有修改原仓库。

本地化仅引入前向所需的 core、DepthAnything/DPT、DINOv2 模型和 layers，
没有引入原仓库训练/评估/演示代码。移除了 sys.path 注入和 Utils 的全局 logging 重置。
原始版权头与可用许可证保留在 stereo 目录；不同组件的许可分别适用，未统一改许可。

测试要求 CUDA，默认不反传；`--check-grad` 对特征平方均值反传，仅检查 FPN
梯度和骨干冻结。FP32 FPN 接收转换后的 DINO 特征，骨干混合精度沿用 YAML。
单独的 FoundationStereo checkpoint 不包含新 FPN，FPN 此阶段随机初始化。
