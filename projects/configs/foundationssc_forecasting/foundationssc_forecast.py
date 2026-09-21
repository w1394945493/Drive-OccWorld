_base_ = '../foundationssc/foundationssc_semantic_kitti.py'

#! 冻结开关：True=冻结，False=解冻；默认只训练未来预测器 dynamics。
freeze_frontend = True  # 图像金字塔 + voxel_encoder；不包含始终冻结的 FoundationStereo。
freeze_decoder = True  # 3D ResNet + 3D FPN + 占据分类头。
# 解冻后仅由未来占据损失联合优化；当前帧损失及辅助深度/语义监督尚未加入。
# 解冻会增加显存；CLI 覆盖时使用 model.freeze_frontend=False / model.freeze_decoder=False。
#! 必须指定已训练的完整单帧 checkpoint（不是 FoundationStereo 权重）。
load_from = "/c20250502/wangyushen/Outputs/drive_occworld/foundationssc/train2/best_current_mIoU_epoch_15.pth"
auto_resume = False
work_dir = 'out/foundationssc_forecast'
custom_imports = dict(imports=['projects.mmdet3d_plugin.foundationssc_forecasting'],
                      allow_failed_imports=False)
#! 各步查询上一时刻三维 memory，给定真实未来自车位姿；不再使用卷积残差基线。
model = dict(type='FoundationSSCForecastModel', future_steps=4,
             freeze_frontend=freeze_frontend, freeze_decoder=freeze_decoder,
             attention_heads=4, sampling_points=4,
             use_depth_loss=False, use_semantic_loss=False)
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
