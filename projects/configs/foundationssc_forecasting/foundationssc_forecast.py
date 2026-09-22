_base_ = '../foundationssc/foundationssc_semantic_kitti.py'

#! 冻结开关：True=冻结，False=解冻；默认只训练未来预测器 dynamics。
freeze_frontend = True  # 图像金字塔 + voxel_encoder；不包含始终冻结的 FoundationStereo。
freeze_decoder = True  # 3D ResNet + 3D FPN + 占据分类头。
# 解冻前端时模型自动启用当前帧辅助深度/语义损失，需搭配 no_freeze 的训练 pipeline。
# 前端或解码器解冻时启用当前帧占据损失；仅修改 freeze_frontend 不会自动增加点云投影 pipeline。
# 解冻会增加显存；CLI 覆盖时使用 model.freeze_frontend=False / model.freeze_decoder=False。
#! 必须指定已训练的完整单帧 checkpoint（不是 FoundationStereo 权重）。
load_from = "/c20250502/wangyushen/Outputs/drive_occworld/foundationssc/train2/best_current_mIoU_epoch_15.pth"
# auto_resume = False
auto_resume = True
work_dir = 'out/foundationssc_forecast'
custom_imports = dict(imports=['projects.mmdet3d_plugin.foundationssc_forecasting'],
                      allow_failed_imports=False)
#! 各步查询上一时刻三维 memory，给定真实未来自车位姿；不再使用卷积残差基线。
model = dict(type='FoundationSSCForecastModel', future_steps=4,
             freeze_frontend=freeze_frontend, freeze_decoder=freeze_decoder,
             #! 可替换的未来状态更新模块；显式指定全部构建参数。
             #* channels/pc_range 必须与基础配置 voxel_encoder 的通道数/空间范围一致。
             dynamics=dict(type='PoseVoxelAttention', channels=128,
                           pc_range=(0, -25.6, -2, 51.2, 25.6, 4.4), heads=4, points=4),
             loss_depth_weight=1., loss_seg_weight=1.)  # 辅助损失启停由 freeze_frontend 决定。
#* no_freeze 配置继承同一 dynamics；替换结构时用 dynamics=dict(_delete_=True, type='新注册类', ...)。
# 新类用 @NECKS.register_module() 注册，并通过 custom_imports 导入；切换结构请更换 work_dir，避免自动恢复旧预测器。
#! 仅当前双目图像，未来四帧标签用于监督；原始帧间隔5，按10Hz名义频率对应0.5秒。
forecast_pipeline = [
    dict(type='LoadFoundationSSCStereo', input_size=(384, 1280)),
    #! future_steps=4：当前帧之后预测4步，标签共5帧（当前+未来4帧）；不是 feature_steps。
    # frame_stride=5：相邻关键帧的原始帧编号相差5；本参数仅校验窗口间隔，不负责抽帧。
    # 窗口由 Dataset 沿 PKL 的 prev/next 链构建，需与模型 future_steps、Dataset future_queue_length 一致。
    # 按原始序列名义10Hz计算：每步5/10=0.5s，四步对应0.5/1.0/1.5/2.0s。
    # 若当前原始帧号为10，则目标帧号为15/20/25/30；最远跨度=future_steps*frame_stride=20帧。
    dict(type='LoadFoundationForecastOccupancy', future_steps=4, frame_stride=5),
    dict(type='PackFoundationSSCInputs', runner_format=True),
]
data = dict(samples_per_gpu=1,
    train=dict(history_queue_length=0, future_queue_length=4, pipeline=forecast_pipeline),
    val=dict(history_queue_length=0, future_queue_length=4, pipeline=forecast_pipeline))
#! 先按当前+未来平均选最优；日志/评估沿用已有逐样本混淆矩阵入口。
evaluation = dict(save_best='avg_mIoU', rule='greater')
