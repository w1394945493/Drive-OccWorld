"""联合训练原可训练感知模块与未来预测器；FoundationStereo 仍冻结。"""
_base_ = './foundationssc_forecast.py'

freeze_frontend = False  #* 修改1：解冻图像金字塔与 voxel_encoder；FoundationStereo 仍冻结。
freeze_decoder = False  #* 修改2：解冻 3D ResNet、FPN 和占据头，与 dynamics 联合优化。
model = dict(freeze_frontend=freeze_frontend, freeze_decoder=freeze_decoder)  #* 显式覆盖 model，顶层变量不会自动更新父配置字典。
work_dir = 'out/foundationssc_forecast_no_freeze'  #* 修改3：独立保存日志及 checkpoint。

#* 修改4：训练加载当前帧 LiDAR 并投影到左图，恢复辅助深度/二维语义监督；验证保持不读取 LiDAR。
train_pipeline = [
    dict(type='LoadFoundationSSCStereo', input_size=(384, 1280)),
    dict(type='LoadFoundationForecastOccupancy', future_steps=4, frame_stride=5),
    dict(type='LoadSemanticKITTIPointsAndLabels', pts_label_root=None),
    dict(type='ProjectFoundationSSCLidar', camera_indices=(0,)),
    dict(type='PackFoundationSSCInputs', runner_format=True),
]
data = dict(train=dict(pipeline=train_pipeline))
# 其余继承父配置；未来四步占据损失 + 当前辅助损失，无当前帧占据损失。解冻增加显存。
