# 当前帧占据预测与监督

参照 FoundationSSC-small-SemanticKITTI.py 及下列原文件移植，运行无需原仓库：

- `models/backbones/resnet3d.py`：仅移植 BasicBlock3D、CustomResNet3D，保留 BN3d 残差结构。
- `models/necks/generalizedfpn.py`：保留 top-down 插值、拼接、1×1/3×3 卷积。
- `models/dense_heads/occ_head.py`：保留两层分类卷积及 logits 三线性插值。
- `utils/semkitti.py`：保留原 CE、semantic/geometric scaling 公式及类别频率。

不注册全局同名类，由本地模型直接构造。Neck 的 BaseModule 改用 nn.Module，
默认配置不涉及 init_cfg；网络层与前向结构保留。原仓库许可见上级
`ops/FOUNDATIONSSC_LICENSE`，不修改原仓库。

## 默认数据流

```text
融合体素 [B,128,128,128,16]
  → 3D ResNet 三个尺度：128×128×16、64×64×8、32×32×4
  → 3D FPN（原实现返回前两个融合尺度）
  → 取第一个 [B,128,128,128,16]
  → Conv3d 128→64 + GN + ReLU + Conv3d 64→20
  → 低分辨率 logits [B,20,128,128,16]
  → trilinear(align_corners=False) → [B,20,256,256,32]
  → argmax 类别预测 [B,256,256,32]
```

始终保持 `[B,C,X,Y,Z]` 轴顺序，不对整数 GT 插值；0=empty，255=ignore。
GT 必须已经映射为 0..19 或 255，非法值显式报错，不静默重映射。

## 损失与适配

- `loss_voxel_ce`：按原 `1/log(class_frequency+0.001)` 类别权重计算交叉熵。
- `loss_voxel_sem_scal`：对出现的语义类别约束 precision/recall/specificity，沿用原公式。
- `loss_voxel_geo_scal`：将非 empty 类合并，约束空/非空几何，沿用原公式。

三项默认权重均为 1。原 `non_empty_idx` 参数实际用于索引 empty 概率，传配置的 `empty_idx`。
必要适配：class_weights 注册为 buffer；取消未加权时硬编码 17 类；支持配置 ignore_index；
全 ignore 输入返回可反传零损失；损失强制 FP32/关闭 autocast，避免 scaling 的 BCE 混合精度错误。
默认有效标签条件下与原损失一致；没有额外加入 Lovasz 或更改原损失权重。

未接入原 depth loss 和图像语义辅助 loss，当前 pipeline 没有提供 gt_depths/gt_semantics。
仅接占据损失不等同于已复现原论文全部训练目标，也尚未接 IoU/mIoU 评估及正式训练 runner。

## 验证

仍使用一个主脚本，不新增分阶段入口：

```bash
python scripts/test_foundationssc.py
python scripts/test_foundationssc.py --check-grad
```

前者验证完整预测与损失数值，后者对真实 GT 损失反传。新增占据 3D 网络与高分辨率
logits 会增加显存，之前体素特征检查的 13.23 GiB 不能作为完整 SSC 的显存上限。
