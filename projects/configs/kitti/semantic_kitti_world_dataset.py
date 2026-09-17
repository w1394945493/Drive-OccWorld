"""Minimal config for SemanticKITTIWorldDataset sanity check.

用途：
    只用于验证新建的 SemanticKITTIWorldDataset 能否从 converter 生成的
    train/val pkl 中读取一帧数据；不包含模型配置。

使用方式：
    python scripts/test_semantic_kitti_world_dataset.py \
        --config projects/configs/kitti/semantic_kitti_world_dataset.py \
        --split train \
        --index 0
"""

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

data_root = '/c20250502/wangyushen/Datasets/kitti/semantickitti/dataset'
ann_root = 'out/semantic_kitti'

#* ================== Stage-1 时序窗口配置 ==================
# 对齐 Drive-OccWorld 第一阶段：
# - history_queue_length: 输入历史帧数量；
# - future_queue_length : 需要加载的未来 occupancy 标注数量。
history_queue_length = 2
future_queue_length = 4

#* ================== 相机输入模式 ==================
# 可选：
# - left:   只使用 image_2 / CAM_FRONT_LEFT，推荐作为单目 baseline；
# - stereo: 使用 image_2 + image_3 双目前视图像。
use_camera = 'left'

# 这里先不设置 pipeline，让 Dataset 返回原始 frame_inputs + segmentation，
# 同时由 Dataset 直接读取 history+current 图像，便于检查第一阶段模型输入：
#   img: [history + current, num_cam, H, W, C]
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
    test_mode=True,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=semantic_kitti_train_dataset,
    val=semantic_kitti_val_dataset,
)
