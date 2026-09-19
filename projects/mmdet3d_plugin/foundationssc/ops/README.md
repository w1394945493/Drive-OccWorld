# 本地 CUDA 汇聚与注意力

不依赖外部 FoundationSSC、bev_pool 或 dfa3D 扩展，也不在运行时自动 JIT 编译。

- LSS：融合 depth probability × context 与 voxel scatter-add，避免保存完整 lifted 特征。
- DFA3D：对隐式 depth × context 特征体做八邻域三线性采样，并融合注意力加权求和。
- 2D deformable self-attention：融合四邻域双线性采样和权重归约，保留原双队列平均语义。
- 三者均实现一阶 backward，包括注意力采样坐标、权重、特征和深度概率梯度。

## 就地编译

在实际训练的 conda 环境中执行，CUDA toolkit 应与 PyTorch 的 CUDA 版本匹配：

```bash
cd projects/mmdet3d_plugin/foundationssc/ops
# 8.0 对应 A100；其他 GPU 请设置相应架构。
TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=2 python setup.py build_ext --inplace
```

生成的 `_C*.so` 位于本目录，通过相对包路径加载，无需 pip install。
切换 Python/PyTorch/CUDA 环境后应重新编译，不要直接复制另一环境的 so。
需要强制重编译时追加 `--force`。

配置中 `voxel_encoder.ops_backend='cuda'` 已启用新实现；`pytorch` 保留对照路径。
`auto` 按输入设备选择后端；CUDA 输入缺少扩展时明确报错，不静默回退。
目前接口针对本项目单尺度特征，不是原 DFA3D 全接口的直接替代品。
只支持 FP32 和一阶梯度；坐标应为有限值。CUDA atomicAdd 的归约次序可能带来微小
数值差异，不保证逐位确定性。实际提速与峰值显存需在目标 GPU 上测量。

## 验证（回到仓库根目录）

```bash
python scripts/test_foundationssc_cuda_ops.py --import-only
python scripts/test_foundationssc_cuda_ops.py
python scripts/test_foundationssc_cuda_ops.py --benchmark
python scripts/test_foundationssc.py --stage voxels --check-grad
# 对照后端：
python scripts/test_foundationssc.py --stage voxels --ops-backend pytorch --check-grad
```

算子测试无需数据或权重，覆盖 batch=1/2、边界/越界、空候选、所有输入梯度，
并比较小尺寸完整体素模块的前向和反向。真实数据测试仍需要 PKL 与立体骨干权重。
`--import-only` 仅验证扩展加载，不代表 CUDA 数值/梯度测试通过。
