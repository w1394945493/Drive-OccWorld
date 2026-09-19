# FoundationSSC 原始扩展本地移植

本目录替换此前自写的 `_C` 融合内核。原仓库仅作移植来源，运行和编译不再依赖它。

| 功能 | 来源 | 本地执行 |
|---|---|---|
| LSS 汇聚 | `packages/bev_pool/bev_pool/` | `bev_pool/bev_pool_ext*.so` |
| DFA3D | `packages/DFA3D/dfa3D/` | `dfa3D/_ext*.so` |
| DFA3D 前后向组合 | `models/img2bev/transformer_utils/multi_scale_3ddeformable_attn_function.py` | 原封装，仅调整相对导入 |
| 2D 自注意力 | `models/img2bev/transformer_utils/multi_scale_deformable_attn_function.py` | 原封装调用 `mmcv._ext`，与原工作一致 |

所有复制的 C++/CUDA/头文件逐字节保持原样，不再重写插值、汇聚或梯度公式。
`source_manifest.json` 记录原路径和 SHA256；26 个文件中仅 `dfa3D/ext_loader.py`
和 3D autograd 封装调整了本地包导入，其余文件与参考仓库完全一致。
新增 `functional.py` 仅处理当前前端的输入布局、设备检查和空输入，不定义新 autograd。

## 编译

在实际训练 conda 环境中执行（CUDA toolkit 应匹配 PyTorch CUDA 版本）：

```bash
cd projects/mmdet3d_plugin/foundationssc/ops
# A100 对应 8.0，其他卡需设置相应架构。
TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=2 python setup.py build_ext --inplace
```

同时编译 bev_pool 与 DFA3D，两份 so 输出到上述子目录。无需全局安装同名包。
MMCV 2D attention 使用环境中已有的 mmcv-full CUDA 扩展，本命令不会重新编译 MMCV。
旧 `_C*.so` 即使残留也不会被加载；更换 Python/PyTorch/CUDA 环境应重新编译。

## 调用约定与保留边界

- LSS 恢复原 Lift → 排序 → 分段求和路径，不再宣称融合 Lift 内核的内存优势。
  原汇聚输出 `[B,C,Z,X,Y]`，适配后转为前端需要的 `[B,C,X,Y,Z]`。
- DFA3D 按原实现先采样四邻域的 depth_score，再做加权注意力；每个 head 共用深度分布。
- 自注意力两队列合入 batch，分别调用 MMCV 后再平均，与原调用方式一致。
- 适配入口目前使用 FP32；原 fp16/fp32 封装均保留。坐标输入应有限。
- 原 bev_pool 使用默认 CUDA 流，适配层拒绝非默认流，不修改其内核。
- 原 bev_pool 不支持空集合、注意力不支持零 query，外围返回带零梯度链的空/零结果。
- 原 DFA3D 组合封装不支持给 depth_score 单独附加损失；本前端只使用 output。
- 保留原 atomic 归约及其浮点误差，不承诺逐位确定性或比旧实现更快。

## 验证（仓库根目录）

```bash
# 无 GPU 也可检查源码；reference-root 可选，模型运行不需要原仓库。
python scripts/test_foundationssc_cuda_ops.py --source-only
python scripts/test_foundationssc_cuda_ops.py --source-only --reference-root ../FoundationSSC
# 检查实际加载的 so 路径，包括 MMCV。
python scripts/test_foundationssc_cuda_ops.py --import-only
# GPU：汇聚数值、适配层对原封装的前后向、完整体素模块梯度。
python scripts/test_foundationssc_cuda_ops.py
python scripts/test_foundationssc.py --check-grad
```

完整模块检查有限梯度和关键偏移参数非零梯度；算子适配层与原封装仍严格比较前后向。
不再将之前自写 PyTorch/grid_sample 在插值折点的导数作为原 CUDA 实现的判定基准。
PyTorch 后端仍可用于 CPU 调试，并不据此宣称完整模型已复现原论文精度。

## 来源与许可

保留全部源文件版权头与仓库 `FOUNDATIONSSC_LICENSE`。DFA3D 文件另有 IDEA License
声明及 OpenMMLab 来源声明，应遵守各组件许可；参考仓库未附独立 IDEA 许可正文，
本移植不将这些组件统一重新许可。没有修改 FoundationSSC 原仓库。
