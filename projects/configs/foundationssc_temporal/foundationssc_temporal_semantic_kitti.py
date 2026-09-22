"""时序接口验证：历史一帧＋当前双目，当前 SSC 标签；尚不支持模型训练。"""
_base_ = '../foundationssc/foundationssc_semantic_kitti.py'
custom_imports = dict(imports=['projects.mmdet3d_plugin.foundationssc_temporal'], allow_failed_imports=False)
model = dict(type='FoundationSSCTemporalModel')
# history_steps = 1
history_steps = 3
input_size = (384, 1280)
history_loader = dict(type='LoadFoundationSSCHistory', history_steps=history_steps,
                      frame_stride=5, frame_period=0.1, input_size=input_size)
#* 所有历史数据从 PKL 获取，不重新读取 calib.txt，不引入 PoseNet 或新 ResNet。
val_pipeline = [
    dict(type='LoadFoundationSSCStereo', input_size=input_size),
    history_loader,
    dict(type='LoadFoundationSSCOccupancy', occ_size=(256, 256, 32),
         point_cloud_range=(0, -25.6, -2, 51.2, 25.6, 4.4)),
    dict(type='PackFoundationSSCTemporalInputs', runner_format=True)]

train_pipeline = val_pipeline[:-1] + [
    dict(type='LoadSemanticKITTIPointsAndLabels', pts_label_root=None),
    dict(type='ProjectFoundationSSCLidar', camera_indices=(0,)), val_pipeline[-1]]
data = dict(
    train=dict(history_queue_length=history_steps, future_queue_length=0,
               filter_invalid=True, pad_history_with_current=True, pipeline=train_pipeline),
    val=dict(history_queue_length=history_steps, future_queue_length=0,
             filter_invalid=True, pad_history_with_current=True, pipeline=val_pipeline))
#! 历史不足先复制当前帧补齐，因此场景第一帧也保留；filter_invalid 仍检查其他窗口问题。
work_dir = 'out/foundationssc_temporal'
