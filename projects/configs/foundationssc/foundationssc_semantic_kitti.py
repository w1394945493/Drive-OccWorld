"""FoundationSSC Small：双目 → 体素 → 当前帧占据预测与三项占据损失。"""
#* ================== 训练参数 ==================
samples_per_gpu = 1
workers_per_gpu = 4
total_epochs = 24
learning_rate = 3e-4
log_interval = 50
eval_interval = 1
checkpoint_interval = 1
max_keep_ckpts = 1
train_max_samples = None
val_max_samples = None

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
#* 按配置注册新模型，不让原 Drive-OccWorld 配置额外依赖 FoundationSSC。
custom_imports = dict(imports=['projects.mmdet3d_plugin.foundationssc'], allow_failed_imports=False)
ann_root = 'data/semantic_kitti'
input_size = (384, 1280)
occ_size = (256, 256, 32)
point_cloud_range = (0, -25.6, -2, 51.2, 25.6, 4.4)
#* 原 SemanticKITTI 20 类统计，0=empty，255=ignore；用于 1/log(freq+0.001) 的 CE 权重。
num_classes = 20
empty_idx = 0
ignore_index = 255
semantic_kitti_class_frequencies = [
    5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05,
    6.46799000e05, 8.21951000e05, 2.62978000e05, 2.83696000e05,
    2.04750000e05, 6.16887030e07, 4.50296100e06, 4.48836500e07,
    2.26992300e06, 5.68402180e07, 1.57196520e07, 1.58442623e08,
    2.06162300e06, 3.69705220e07, 1.15198800e06, 3.34146000e05,
]
pipeline = [
    dict(type='LoadFoundationSSCStereo', input_size=input_size),
    dict(type='LoadFoundationSSCOccupancy', occ_size=occ_size,
         point_cloud_range=point_cloud_range),
    dict(type='PackFoundationSSCInputs', runner_format=True),
]
dataset_common = dict(
    type='SemanticKITTIWorldDataset', pipeline=pipeline, use_camera='stereo',
    history_queue_length=0, future_queue_length=0, filter_invalid=True,
    format_for_train=False, load_img=True, load_occ=True)
data = dict(
    samples_per_gpu=samples_per_gpu, workers_per_gpu=workers_per_gpu,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
    train=dict(**dataset_common, ann_file=f'{ann_root}/semantickitti_infos_train.pkl',
               max_samples=train_max_samples, test_mode=False),
    val=dict(**dataset_common, ann_file=f'{ann_root}/semantickitti_infos_val.pkl',
             max_samples=val_max_samples, test_mode=True))

#* 数据与模型统一配置；FoundationStereo/DINOv2 均使用本仓库实现。
#* 对齐原 FoundationSSC-small-SemanticKITTI.py：使用 11-33-40 的 Small 权重/YAML，
# DINOv2 为 ViT-S/14，输出 384 通道；不是只把 Large 模型通道数强行缩小。
# FoundationImagePyramid 自动生成 [96,192,384,384] 四尺度，融合后仍为 640 通道。
# Small 不改变图像尺寸、体素网格和三维模块规模。
model = dict(
    type='FoundationSSCImageModel',
    stereo_checkpoint='/c20250502/wangyushen/Weights/foundationssc/11-33-40/model_best_bp2.pth',
    stereo_config='/c20250502/wangyushen/Weights/foundationssc/11-33-40/cfg.yaml',
    gru_iters=12, backbone_channels=384, out_channels=160,
    strict_load=True,
    #* 第三阶段：输出 [B,128,128,128,16] = [B,C,X,Y,Z]，覆盖完整 point_cloud_range。
    # 与 GT [256,256,32] 不同，体素边长为 0.4m；占据头对 logits 上采样到目标尺寸。
    voxel_encoder=dict(
        #* ops/ 就地编译原 bev_pool/DFA3D；自注意力沿用 MMCV，pytorch 仅用于调试。
        ops_backend='cuda',
        point_cloud_range=point_cloud_range, voxel_shape=(128, 128, 16),
        input_size=input_size, depth_bound=(2., 58., .5), channels=128,
        downsample=8, pool_chunk=8,
        # disparity_channels 自动取 stereo YAML 的 max_disp//4，不再硬编码 104。
        depth_cfg=dict(dformer_layers=4, mixer_layers=8),
        refiner_cfg=dict(cross_layers=3, self_layers=2, heads=8, points=8,
                         ffn_channels=256, dropout=.1, self_layout=(512, 512),
                         query_chunk=2048),
        fusion_groups=16),
    #* 原占据编码路径：三组 3D 残差块，输出 1x / 1/2x / 1/4x 空间分辨率。
    occ_encoder_backbone=dict(
        type='CustomResNet3D', numC_input=128, num_layer=[2, 2, 2],
        num_channels=[128, 128, 128], stride=[1, 2, 2], with_cp=False),
    #* 原 GeneralizedLSSFPN 实际返回两个融合尺度；预测头使用最高分辨率的第一个。
    occ_encoder_neck=dict(
        type='GeneralizedLSSFPN', in_channels=[128, 128, 128],
        out_channels=128, start_level=0, num_outs=3,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        conv_cfg=dict(type='Conv3d'), act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='trilinear', align_corners=False)),
    #* 128→64→20 通道分类，再将 logits 从 [128,128,16] 插值到 [256,256,32]。
    # 不插值类别标签、不改变 GT；未接入原 depth/2D segmentation 辅助监督。
    pts_bbox_head=dict(
        type='OccHead', in_channels=[128], out_channel=num_classes,
        num_level=1, with_cp=False, occ_size=occ_size,
        empty_idx=empty_idx, ignore_index=ignore_index,
        balance_cls_weight=True, class_frequencies=semantic_kitti_class_frequencies,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        loss_weight_cfg=dict(loss_voxel_ce_weight=1., loss_voxel_sem_scal_weight=1.,
                             loss_voxel_geo_scal_weight=1.)))
#* 完整 stereo checkpoint 必须包含 EdgeNeXt、DINO 和立体匹配网络权重。
# Small 权重需搭配其原版 cfg.yaml（vit_size='vits'），不要混用 23-51-11 的 Large YAML。
# 不再读取任何辅助骨干 checkpoint，也不自动下载权重。

#* ================== tools/train.py 正式运行配置 ==================
# 只训练当前帧 SSC 的三项占据损失，不等于原论文全部辅助监督的复现。
# 优化器参考原 Small 的 AdamW；此处采用本仓库 epoch runner 的余弦调度。
optimizer = dict(type='AdamW', lr=learning_rate, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='CosineAnnealing', by_epoch=True,
                 warmup='linear', warmup_iters=500, warmup_ratio=1. / 3,
                 min_lr_ratio=1e-3)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=checkpoint_interval, by_epoch=True,
                         max_keep_ckpts=max_keep_ckpts)
log_config = dict(interval=log_interval, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=eval_interval, save_best='current_mIoU', rule='greater')
workflow = [('train', 1)]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'out/foundationssc_semantic_kitti'
load_from = None  # 冻结骨干由 model.stereo_checkpoint 加载；不是全模型 checkpoint。
resume_from = None
auto_resume = True
# 动态可见体素/候选筛选可能使部分参数在某个 batch 未参与 loss，DDP 需允许未使用参数。
find_unused_parameters = True
cudnn_benchmark = False
#* 暂不启用 MMCV fp16：先验证原 CUDA 算子的 FP32 训练闭环。
