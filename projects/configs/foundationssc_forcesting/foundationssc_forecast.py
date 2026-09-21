_base_ = '../foundationssc/foundationssc_semantic_kitti.py'

#! 必须指定已训练的完整单帧 checkpoint（不是 FoundationStereo 权重）。
load_from = None
auto_resume = False
work_dir = 'out/foundationssc_forecast'
custom_imports = dict(imports=['projects.mmdet3d_plugin.foundationssc_forcesting'],
                      allow_failed_imports=False)
#! 各步查询上一时刻三维 memory，给定真实未来自车位姿；不再使用卷积残差基线。
model = dict(type='FoundationSSCForecastModel', future_steps=4,
             attention_heads=4, sampling_points=4,
             use_depth_loss=False, use_semantic_loss=False)
#! 仅当前双目图像，未来四帧标签用于监督；原始帧间隔5，按10Hz名义频率对应0.5秒。
forecast_pipeline = [
    dict(type='LoadFoundationSSCStereo', input_size=(384, 1280)),
    dict(type='LoadFoundationForecastOccupancy', future_steps=4, frame_stride=5),
    dict(type='PackFoundationSSCInputs', runner_format=True),
]
data = dict(samples_per_gpu=1,
    train=dict(history_queue_length=0, future_queue_length=4, pipeline=forecast_pipeline),
    val=dict(history_queue_length=0, future_queue_length=4, pipeline=forecast_pipeline))
#! 先按当前+未来平均选最优；日志/评估沿用已有逐样本混淆矩阵入口。
evaluation = dict(save_best='avg_mIoU', rule='greater')
