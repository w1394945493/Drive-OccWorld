"""Debug config for SemanticKITTI Drive-OccWorld experiments.

用途：
    在 SemanticKITTIWorldDataset 已能读取历史/当前图像与未来 occupancy
    序列的基础上，进一步定义 Drive_OccWorld 模型配置。最终目标是跑通
    SemanticKITTI 上的 future occupancy forecasting 流程。
    本 debug 版本相比 semantic_kitti_drive_occworld.py 更轻量：

        - image backbone 从 ResNet101 + DCNv2 改为 ResNet50；
        - 使用 ResNet50/FPN 常见 COCO/nuImages 预训练权重初始化 backbone；
        - 保持图像几何预处理为 pad-only，不做 resize/crop，避免 lidar2img
          需要额外同步更新。

        历史/当前单目图像 -> BEV -> future occupancy forecasting

    第一阶段暂时关闭：
        - planning head；
        - flow / instance-level VPQ；
        - 真实 CAN bus action condition。

    但为了跑通原 Drive-OccWorld future_pred() 接口，Dataset 会构造
    pseudo sdc_planning / command / vel_steering 字段。

使用方式：
    python scripts/test_semantic_kitti_world_dataset.py \
        --config projects/configs/kitti/semantic_kitti_drive_occworld_debug.py \
        --split train \
        --index 0
"""

_base_ = [
    '../_base_/default_runtime.py',
]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

data_root = '/c20250502/wangyushen/Datasets/kitti/semantickitti/dataset'
ann_root = "/vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/data/semantic_kitti"

# *=======================================================
# * 训练参数
# 统一管理 train.py 常用训练控制项，方便命令行或改配置时快速定位。
# - samples_per_gpu: 每张 GPU 的 batch size；
# - workers_per_gpu: 每张 GPU 对应的 dataloader worker 数；
# - max_epochs: 总训练 epoch 数；
# - log_interval: 训练日志打印间隔；
# - eval_interval: 评估间隔；EpochBasedRunner 下表示每多少个 epoch 评估一次；
# - checkpoint_interval: 保存模型间隔；EpochBasedRunner 下表示每多少个 epoch 保存一次；
# - max_keep_ckpts: 最多保留多少个 checkpoint，避免 work_dir 无限增大。
# - learning_rate: AdamW 基础学习率；img_backbone 会在 optimizer.paramwise_cfg
#   中乘以 lr_mult=0.1。
samples_per_gpu = 1
workers_per_gpu = 4
max_epochs = 24
log_interval = 50
eval_interval = 1
checkpoint_interval = 1
max_keep_ckpts = 1
learning_rate = 2e-4

# * ================== 快速调试样本数开关 ==================
# 默认 None 表示使用完整 train/val split。
# 调试 EvalHook 是否能跑通时，可以命令行覆盖：
#   --cfg-options data.val.max_samples=20
# 如果想快速 overfit/debug 训练，也可以覆盖：
#   --cfg-options data.train.max_samples=100
train_max_samples = None
val_max_samples = None

# * ================== Stage-1 时序窗口配置 ==================
# 对齐 Drive-OccWorld 第一阶段：
# - history_queue_length: 输入历史帧数量；
# - future_queue_length : 需要加载的未来 occupancy 标注数量。
history_queue_length = 2
future_queue_length = 4

# * ================== 相机输入模式 ==================
# 可选：
# - left:   只使用 image_2 / CAM_FRONT_LEFT，推荐作为单目 baseline；
# - stereo: 使用 image_2 + image_3 双目前视图像。
use_camera = 'left'

# * ================== 图像预处理配置 ==================
# 当前 SemanticKITTI 原图常见大小为 376x1241。
# 第一阶段先不做 resize/crop/flip，只 pad 到 384x1248：
# - 384 和 1248 都能被 32 整除，适合 CNN/FPN；
# - 不改变像素坐标尺度和原点，因此 lidar2img / cam_intrinsic 不需要更新；
# - 若后续改成 resize/crop，需要同步更新 cam_intrinsic / lidar2img。
pad_shape = (384, 1248)
size_divisor = 32

#* ================== Debug 版图像归一化 ==================
# 参照 Uni-Occ/SuperOcc 的 ResNet50 预训练设置：
#   mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
#
# 原正式配置使用 ResNet101 + DCNv2 + Caffe 风格 BGR 归一化：
#   mean=[103.530, 116.280, 123.675], std=[1,1,1], to_rgb=False
#
# todo: 当前只改颜色归一化，不改 resize/crop。
# todo: 如果后续参考 Uni-Occ 把图像 resize 到 256x704，需要同步更新
# todo: cam_intrinsic / lidar2img，否则 BEVFormer 的图像-几何投影会错位。
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

# * ================== SemanticKITTI occupancy / BEV 配置 ==================
# SemanticKITTI dense occupancy 标签当前 shape 为 [256, 256, 32]，255 为 ignore。
# 第一阶段为了减少额外 resize/remap，先让 BEV 网格与 occupancy 的 H/W 对齐。
point_cloud_range = [0.0, -25.6, -2.0, 51.2, 25.6, 4.4]
voxel_size = [0.2, 0.2, 0.2]
occ_size = [256, 256, 32]
# bev_h_ = 256
# bev_w_ = 256
# pred_height = 32
bev_h_ = 128
bev_w_ = 128
pred_height = 16


# SemanticKITTI 常用 20 类编码为 0..19，255 为 ignore。
# 这里仅用于 num_classes / class_weights 长度；类别名主要帮助阅读配置。
semantic_kitti_class_names = [
    'empty',
    'car',
    'bicycle',
    'motorcycle',
    'truck',
    'other-vehicle',
    'person',
    'bicyclist',
    'motorcyclist',
    'road',
    'parking',
    'sidewalk',
    'other-ground',
    'building',
    'fence',
    'vegetation',
    'trunk',
    'terrain',
    'pole',
    'traffic-sign',
]
num_cls = len(semantic_kitti_class_names)
empty_idx = 0

# * ================== Drive-OccWorld 模型配置 ==================
# 目标是逐步跑通 SemanticKITTI 上的完整 Drive-OccWorld 流程。
# 当前阶段先采用较小模型参数，便于定位数据/forward/loss 链路问题；
# 后续确认链路稳定后，可继续扩大 backbone、embed_dims 和 transformer 层数。
turn_on_flow = False
turn_on_plan = False
only_generate_dataset = False
supervise_all_future = True

memory_queue_len = 1
future_pred_frame_num_train = future_queue_length
future_pred_frame_num_test = future_queue_length

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
future_decoder_layer_num = 3
bevformer_encoder_layer_num = 6

# WorldHeadV1.forward_head 会输出当前帧 + future_queue_length 帧的 occupancy。
frame_loss_weight = [[1] for _ in range(future_queue_length + 1)]
world_head_pred_history_frame_num = 0
world_head_pred_future_frame_num = 0
world_head_per_frame_loss_weight = (1.0,)

model = dict(
    type='Drive_OccWorld',
    turn_on_flow=turn_on_flow,
    turn_on_plan=turn_on_plan,
    memory_queue_len=memory_queue_len,
    use_grid_mask=True,
    video_test_mode=True,
    only_generate_dataset=only_generate_dataset,
    supervise_all_future=supervise_all_future,

    point_cloud_range=point_cloud_range,
    bev_h=bev_h_,
    bev_w=bev_w_,

    future_pred_frame_num=future_pred_frame_num_train,
    test_future_frame_num=future_pred_frame_num_test,

    #* ================== 图像 backbone：debug 版 ResNet50 ==================
    # 参照 Uni-Occ/SuperOcc 的轻量图像 backbone 设置，使用 ResNet50 + FPN。
    # 相比正式配置的 ResNet101 + DCNv2，该版本显存和计算开销更低，
    # 便于后续评估是否采用 ResNet50 作为 SemanticKITTI baseline。
    img_backbone=dict(
        # todo: 这里使用 Uni-Occ/SuperOcc 中的 nuImages/COCO 风格 ResNet50
        # todo: 预训练权重，只初始化 backbone。由于该权重 key 通常带
        # todo: "backbone." 前缀，因此设置 prefix='backbone.'。
        # todo: 如果实际环境没有该文件，可启动时覆盖：
        # todo:   --cfg-options model.img_backbone.init_cfg=None
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/c20250502/wangyushen/Weights/pretrained/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth',
            prefix='backbone.'),
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),

    future_pred_head=dict(
        type='WorldHeadV1',
        num_classes=num_cls,
        history_queue_length=history_queue_length,
        memory_queue_len=memory_queue_len,
        soft_weight=False,
        turn_on_flow=False,
        obj_motion_norm=False,
        pred_history_frame_num=world_head_pred_history_frame_num,
        pred_future_frame_num=world_head_pred_future_frame_num,
        per_frame_loss_weight=world_head_per_frame_loss_weight,
        num_pred_fcs=1,
        num_pred_height=pred_height,
        #! occupancy 中 empty/free space 的类别 id。
        #! 原 Drive-OccWorld 代码在 loss 中硬编码 empty_idx=0；
        #! 这里改为配置项，后续如果数据集类别定义变化，只需要修改 empty_idx。
        empty_idx=empty_idx,

        #* ================== 第一阶段动作条件设置 ==================
        # 不使用 nuScenes CAN bus，因此 use_can_bus=False，避免读取
        # img_meta['future_can_bus']。
        # 仍保留 use_plan_traj=True + use_command=True：
        # - plan_traj 来自 Dataset 构造的 pseudo sdc_planning；
        # - command 来自 Dataset 构造的 pseudo command；
        # 这样既能跑通原 WorldDecoder 的 action-condition 接口，也不依赖真实 CAN bus。
        use_can_bus=False,
        use_plan_traj=True,
        use_command=True,
        use_vel_steering=False,
        use_vel=False,
        use_steering=False,
        use_fourier=False,
        condition_ca_add='ca',
        can_bus_norm=True,
        can_bus_dims=(0, 1, 2, 17),

        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        loss_weight=frame_loss_weight,
        #! 原 Drive-OccWorld 的 WorldHeadV1.loss_voxel 默认按
        #! (256, 256, 20) 对齐监督，适配 nuScenes/OpenOccupancy
        #! 512x512x40 -> 256x256x20 的压缩逻辑。
        #! SemanticKITTI 当前 occupancy 标签为 (256, 256, 32)，
        #! 因此这里显式覆盖，避免高度维被错误插值到 20。
        loss_voxel_align_size=occ_size,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        prev_render_neck=dict(
            type='ConditionalNorm',
            occ_flow='occ',
            embed_dims=_dim_,
            sem_norm=True,
            sem_gt_train=False,
            ego_motion_ln=True,
            obj_motion_ln=False,
            pred_height=pred_height,
            num_cls=num_cls,
            num_pred_fcs=0,
        ),
        transformer=dict(
            type='PredictionTransformer',
            embed_dims=_dim_,
            decoder=dict(
                type='WorldDecoder',
                num_layers=future_decoder_layer_num,
                return_intermediate=True,
                transformerlayers=dict(
                    type='PredictionTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='PredictionMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=memory_queue_len),
                        dict(
                            type='PredictionMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=memory_queue_len),
                        dict(
                            type='GroupMultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'cross_attn_action', 'norm', 'ffn', 'norm')))),
    ),

    pts_bbox_head=dict(
        type='WorldBEVFormerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=num_cls,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer',
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            encoder=dict(
                type='CustomBEVFormerEncoder',
                num_layers=bevformer_encoder_layer_num,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayerV2',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            # Drive_OccWorld.__init__ 会删除 detection decoder/box head；
            # 这里保留占位配置只是为了兼容 WorldBEVFormerHead 构造。
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=1,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-10.0, -35.6, -5.0, 61.2, 35.6, 8.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_cls),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[256, 256, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
            pc_range=point_cloud_range))))

# 这里先不设置 pipeline，让 Dataset 自己返回 frame_inputs + img + img_metas + segmentation，
# 便于检查第一阶段模型输入：
#   img: [history + current, num_cam, C, H, W]
#   segmentation: [history + current + future, H_occ, W_occ, D_occ]
semantic_kitti_train_dataset = dict(
    type='SemanticKITTIWorldDataset',
    ann_file=f'{ann_root}/semantickitti_infos_train.pkl',
    data_root=data_root,
    pipeline=None,
    use_camera=use_camera,
    history_queue_length=history_queue_length,
    future_queue_length=future_queue_length,
    filter_invalid=True,
    load_occ=True,
    load_img=True,
    to_float32=True,
    img_norm_cfg=img_norm_cfg,
    pad_shape=pad_shape,
    size_divisor=size_divisor,
    empty_idx=empty_idx,
    max_samples=train_max_samples,
    #* 正式接入 tools/train.py 时打开 DataContainer 格式：
    #* - 只返回 Drive_OccWorld.forward_train 接收的字段；
    #* - 保持 img_metas / segmentation 的特殊 list 结构，避免默认 collate 破坏。
    format_for_train=True,
    test_mode=False,
)

semantic_kitti_val_dataset = dict(
    type='SemanticKITTIWorldDataset',
    ann_file=f'{ann_root}/semantickitti_infos_val.pkl',
    data_root=data_root,
    pipeline=None,
    use_camera=use_camera,
    history_queue_length=history_queue_length,
    future_queue_length=future_queue_length,
    filter_invalid=True,
    load_occ=True,
    load_img=True,
    to_float32=True,
    img_norm_cfg=img_norm_cfg,
    pad_shape=pad_shape,
    size_divisor=size_divisor,
    empty_idx=empty_idx,
    max_samples=val_max_samples,
    format_for_train=True,
    test_mode=True,
)

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    #! custom_train_detector 会从 cfg.data 中读取这两个 sampler 配置。
    #! 单卡 launcher=none 时主要使用 GroupSampler；分布式时使用这里的配置。
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
    train=semantic_kitti_train_dataset,
    val=semantic_kitti_val_dataset,
)

# * ================== 正式 train.py 运行配置 ==================
# 当前仍是 SemanticKITTI 第一阶段 default 配置：
# - 单卡 samples_per_gpu=1；
# - 内部 BEV 分辨率 128x128x16；
# - 默认按 epoch 训练和评估，更接近常规训练流程。
#
# 推荐首次运行：
#   python tools/train.py projects/configs/kitti/semantic_kitti_drive_occworld.py \
#       --work-dir work_dirs/semantic_kitti_drive_occworld_debug \
#       --cfg-options data.val.max_samples=20
#
# 若只想快速 iter 调试，也可临时覆盖回 IterBasedRunner：
#   --cfg-options runner.type=IterBasedRunner runner.max_iters=20 \
#       evaluation.interval=10 checkpoint_config.interval=10 \
#       checkpoint_config.by_epoch=False lr_config.by_epoch=False
optimizer = dict(
    type='AdamW',
    lr=learning_rate,
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }))

optimizer_config = dict(
    # 先保守加梯度裁剪，避免小 batch / 随机初始化早期偶发梯度尖峰。
    grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    by_epoch=True)

total_epochs = max_epochs
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

checkpoint_config = dict(
    interval=checkpoint_interval,
    by_epoch=True,
    max_keep_ckpts=max_keep_ckpts)
log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])

# 每个 epoch 结束后评估一次。
# 调试时建议用 --cfg-options data.val.max_samples=20 缩短评估时间。
evaluation = dict(interval=eval_interval)

workflow = [('train', 1)]
find_unused_parameters = False
cudnn_benchmark = True
# Debug 版 backbone 已通过 model.img_backbone.init_cfg 单独加载 ResNet50
# 预训练权重；这里不要再使用正式配置的 ResNet101 + DCNv2 全局 load_from，
# 否则会出现 backbone 结构/key 不匹配。
load_from = None
resume_from = None
