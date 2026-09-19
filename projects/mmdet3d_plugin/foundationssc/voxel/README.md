# 第三阶段：图像到体素

`encoder.py` 串联如下流程，未使用当前/未来 occupancy GT 生成特征：

1. `depth.py`：标定条件 ContextNet、深度引导 DFormer、视差概率通道 Mixer/UNet。
2. `geometry.py`：LSS 按深度概率 Lift-Splat；立体深度反投影生成二值 proposal。
3. `refiner.py`：proposal query 的深度加权交叉注意力、非 proposal 的 MLP prior、全体素自注意力扩散。
4. `fusion.py`：沿三个平面计算门控，融合 LSS 与细化体素特征。

默认输出 `[B,128,128,128,16]`，轴为 `[B,C,X,Y,Z]`，Z 最快变化。
空间为 `[0,-25.6,-2,51.2,25.6,4.4]` 米，边长 0.4 米；不是最终 20 类占据输出。
最终 occupancy GT 是 `[B,256,256,32]`，此阶段不对 GT 降采样、不计算 loss。

## 几何约定

- 使用 PKL 中左相机列向量 camera→LiDAR 外参；K4 第四列为零，平移仅在外参中计入。
- 像素先反解 resize/crop，再用相机 Z 深度反投影，最后 camera→LiDAR→BDA。
- 深度 `Z=增强后焦距*基线/增强后视差`；不是 LiDAR 距离。只支持无旋转、无翻转的水平整流双目。
- 原生 LSS 采用 linspace(0,尺寸-1) 生成像素网格；注意力采样采用 align_corners=False。
- voxel 索引用 floor 后过滤越界，不能把负数 long 截断到第 0 格。

## 与参考实现的关系和明确差异

参考 FoundationSSC 的 `DSGP_Net.py`、`LSSViewTransformer.py`、`VoxelProposalLayer.py`、
`VoxFormerHead.py`、transformer_utils 与 `DualFeatFusion.py`。所有执行代码都在本仓库。
Context/DFormer/Mixer/融合模块由参考实现移植；几何用 PyTorch 重写，汇聚与注意力提供本地 CUDA 和 PyTorch 后端。
没有用普通全局 attention 替代 deformable attention，也没有跳过 proposal/细化分支。

- 不依赖 `bev_pool`、`spconv`、`dfa3D` 外部扩展；配置默认使用本地 `ops/` CUDA 实现。
  PyTorch 对照后端使用 index_add 汇聚、八邻域三线性采样。
  DFA3D 已与显式 depth×feature 三维体的 grid_sample 比较数值/梯度；未与原 CUDA 扩展逐位比对。
  编译及 CUDA 前向/梯度对照命令见 [../ops/README.md](../ops/README.md)，正式训练前应实测性能。
- 默认保留原 self-attention 的 512×512 展平布局；这是 128×128×16 个 token 的二维布局，
  不是 512×512 米制 BEV。修改 voxel_shape 时须同步修改 self_layout，保证乘积一致。
- batch 中逐样本处理候选，支持小尺寸 batch=1/2 检查；原候选代码仅支持 batch=1。
- ContextNet 的 BN1d 在单样本时用 running stats；affine 参数仍有梯度。
- 视差通道来自 YAML max_disp//4；零/负/非有限视差置无效深度，空候选走 prior，
  不沿用原随机点或全体素 proposal 兜底。全无效深度允许前向但应在实际数据中排查。
- 单尺度单左相机 context，默认三层 cross/two-layer self。移除原融合中未用的 fuse 参数。
- 标定 MLP 数值基于本项目相机坐标，而非原 P2/P3 共用 cam0 坐标；几何等价，
  但其可学习条件数值不同。三维模块重新训练，不能直接加载原完整 FoundationSSC 权重。

## 验证

```bash
python scripts/test_foundationssc_local_voxel.py
python scripts/test_foundationssc.py --stage voxels --indices 0
python scripts/test_foundationssc.py --stage voxels --indices 0 --check-grad
```

第一条为无数据/无 checkpoint 的 CPU 回归测试，包含手算几何、边界、空候选、采样对照、
小尺寸 batch=1/2 完整三维前向及梯度检查。后两条使用真实数据和 CUDA，打印中间张量、
射线/候选/可见体素数量，检查深度概率和真实标定的投影往返。
通过结构/几何测试不代表已恢复原论文精度；占据预测头、监督与评估属于下一阶段。

移植模块的来源保留于此；许可参考上级 stereo/FOUNDATIONSSC_LICENSE 与原组件声明。
注意力设计参考 VoxFormer（NVIDIA Source Code License-NC）与 OpenMMLab；不统一重新许可。
