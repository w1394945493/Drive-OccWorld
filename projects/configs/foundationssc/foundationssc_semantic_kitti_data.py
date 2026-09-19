"""阶段一：当前帧双目数据接口。无模型、无随机增强、无深度/二维语义辅助监督。"""
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
