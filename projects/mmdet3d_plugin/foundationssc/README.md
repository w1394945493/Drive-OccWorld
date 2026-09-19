# FoundationSSC 分阶段移植

目前实现当前帧的图像前端、体素特征、占据预测与三项占据损失。
尚未接入原深度/2D 语义辅助监督、评估以及正式 train.py 的 runner/DataContainer 接口。

- 数据：当前帧左右 RGB、标定、occupancy，全部通过现有 PKL 构造。
- 冻结骨干：本地 `stereo/` 包实现 FoundationStereo、DepthAnything 和 DINOv2。
  返回 `([prob, disp_up], dinov2_features)`，不需要 FoundationSSC 仓库。
- 可训练适配：等价实现 SimpleFPN 默认 `layers=[4]` 和 SECONDFPN，输出 640 通道。
- 三维前端：DSGP 深度/context → LSS 粗体素与 proposal/VoxFormer 细化 → 三平面门控融合。
- 占据分支：原 CustomResNet3D → GeneralizedLSSFPN → OccHead，输出 20 类 logits 和类别预测。
- `forward_test` 返回特征、`output_voxels` 和 `pred`；`forward_train` 默认返回三项真实占据损失。
  调试时 `return_outputs=True` 额外返回同一次前向的特征与预测，不重复运行骨干。

从 Drive-OccWorld 根目录运行：

```bash
python scripts/test_foundationssc.py --indices 0
python scripts/test_foundationssc.py --indices 0 --check-grad
```

数据和模型配置已合并到 `projects/configs/foundationssc/foundationssc_semantic_kitti.py`。
路径可通过 `--stereo-checkpoint`、`--stereo-config`、`--ann-file` 覆盖。
仍需 PyTorch、torchvision、timm（支持 edgenext_small）、OmegaConf、einops、SciPy
等第三方库，以及完整 FoundationStereo 权重和其 YAML 配置；不自动下载依赖或权重。
构造网络不再读取单独的 EdgeNeXt/DINO 权重；外层 checkpoint 必须覆盖全部骨干参数。
原 FoundationSSC 仓库可以不在运行机器上，且没有修改原仓库。

本地化仅引入前向所需的 core、DepthAnything/DPT、DINOv2 模型和 layers，
没有引入原仓库训练/评估/演示代码。移除了 sys.path 注入和 Utils 的全局 logging 重置。
原始版权头与可用许可证保留在 stereo 目录；不同组件的许可分别适用，未统一改许可。

测试固定执行数据、图像/体素特征、占据预测和 GT 损失计算，要求 CUDA，默认不反传；
`--check-grad` 对真实 CE + semantic scaling + geometric scaling 总损失反传，检查包括占据分支
在内的可训练模块梯度和骨干冻结，不更新优化器。骨干混合精度沿用 YAML。
单独的 FoundationStereo checkpoint 不包含新 FPN/三维前端/占据分支，它们当前随机初始化。
Small 配置需要 11-33-40 的权重/YAML；占据分支规模没有随骨干缩小。

占据分支结构、标签和损失约定见 [occupancy/README.md](occupancy/README.md)。

第三阶段实现边界、坐标约定与原版差异见 [voxel/README.md](voxel/README.md)。

体素阶段配置默认使用本地 CUDA 汇聚/注意力，运行前需就地编译；命令及数值/梯度
验证见 [ops/README.md](ops/README.md)。可用 `--ops-backend pytorch` 切回对照后端。
