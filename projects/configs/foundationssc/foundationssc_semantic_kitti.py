"""FoundationSSC Small：双目数据、图像前端、三维体素特征。"""
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
ann_root = 'data/semantic_kitti'
input_size = (384, 1280)
occ_size = (256, 256, 32)
point_cloud_range = (0, -25.6, -2, 51.2, 25.6, 4.4)
pipeline = [
    dict(type='LoadFoundationSSCStereo', input_size=input_size),
    dict(type='LoadFoundationSSCOccupancy', occ_size=occ_size,
         point_cloud_range=point_cloud_range),
    dict(type='PackFoundationSSCInputs'),
]
dataset_common = dict(
    type='SemanticKITTIWorldDataset', pipeline=pipeline, use_camera='stereo',
    history_queue_length=0, future_queue_length=0, filter_invalid=True,
    format_for_train=False, load_img=True, load_occ=True)
data = dict(
    samples_per_gpu=1, workers_per_gpu=0,
    train=dict(**dataset_common, ann_file=f'{ann_root}/semantickitti_infos_train.pkl', test_mode=False),
    val=dict(**dataset_common, ann_file=f'{ann_root}/semantickitti_infos_val.pkl', test_mode=True))

#* 数据与模型统一配置；FoundationStereo/DINOv2 均使用本仓库实现。
#* 对齐原 FoundationSSC-small-SemanticKITTI.py：使用 11-33-40 的 Small 权重/YAML，
# DINOv2 为 ViT-S/14，输出 384 通道；不是只把 Large 模型通道数强行缩小。
# FoundationImagePyramid 自动生成 [96,192,384,384] 四尺度，融合后仍为 640 通道。
# Small 不改变图像尺寸、体素网格和三维模块规模；原占据预测头尚未接入。
model = dict(
    type='FoundationSSCImageModel',
    stereo_checkpoint='/c20250502/wangyushen/Weights/foundationssc/11-33-40/model_best_bp2.pth',
    stereo_config='/c20250502/wangyushen/Weights/foundationssc/11-33-40/cfg.yaml',
    gru_iters=12, backbone_channels=384, out_channels=160,
    strict_load=True,
    #* 第三阶段：输出 [B,128,128,128,16] = [B,C,X,Y,Z]，覆盖完整 point_cloud_range。
    # 与最终 GT [256,256,32] 不同，体素边长为 0.4m；后续占据头再恢复目标尺寸。
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
        fusion_groups=16))
#* 完整 stereo checkpoint 必须包含 EdgeNeXt、DINO 和立体匹配网络权重。
# Small 权重需搭配其原版 cfg.yaml（vit_size='vits'），不要混用 23-51-11 的 Large YAML。
# 不再读取任何辅助骨干 checkpoint，也不自动下载权重。
