"""继承未冻结训练配置，仅替换未来体素状态更新模块。"""
_base_ = './foundationssc_forecast_no_freeze.py'

#* 仅修改 dynamics；_delete_ 清除原 PoseVoxelAttention 的 heads/points 等参数。
model = dict(dynamics=dict(
    _delete_=True, type='DecoupledVoxelDynamics', channels=128,
    pc_range=(0, -25.6, -2, 51.2, 25.6, 4.4),
    hidden_channels=64, refinement_layers=2))
# 其余设置原样继承，包括 work_dir/auto_resume；运行时请用 --work-dir 指定新目录，避免恢复旧预测器。
