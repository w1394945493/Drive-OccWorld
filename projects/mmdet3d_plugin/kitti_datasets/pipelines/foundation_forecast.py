"""当前双目 + oracle 未来自车位姿；标签保留各未来帧原始坐标系。"""
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadFoundationForecastOccupancy:
    def __init__(self, future_steps=4, occ_size=(256, 256, 32),
                 point_cloud_range=(0, -25.6, -2, 51.2, 25.6, 4.4), frame_stride=5):
        self.steps, self.shape = future_steps, tuple(occ_size)
        self.pc_range, self.stride = point_cloud_range, frame_stride

    def __call__(self, results):
        from ..semantic_kitti_world_dataset import SemanticKITTIWorldDataset
        frames = results['frame_inputs']
        #! 第一版仅当前输入，不支持历史；严格校验窗口而非静默截取错位标签。
        if len(frames) != self.steps + 1 or frames[0]['token'] != results['current_token']:
            raise ValueError('预测 pipeline 要求 history=0，future=future_steps')
        previous_pose = SemanticKITTIWorldDataset._lidar_to_global(frames[0])
        targets = []
        transforms = []
        for step, frame in enumerate(frames):
            #! converter.timestamp 实际是原始帧编号，不是真实秒；检查固定关键帧步长。
            if frame['scene_token'] != frames[0]['scene_token'] or frame['timestamp'] - frames[0]['timestamp'] != step * self.stride:
                raise ValueError('未来窗口跨场景或关键帧间隔不匹配')
            label = np.load(frame['occ_path'])
            if label.shape != self.shape or not np.isin(label, list(range(20)) + [255]).all():
                raise ValueError('未来 occupancy 尺寸或语义标签不合法')
            pose = SemanticKITTIWorldDataset._lidar_to_global(frame)
            if step:
                #! T_previous_target=inv(T_global_previous) @ T_global_target。
                #! 使用真实未来位姿作为模型条件，属于 oracle-pose forecasting，不是无条件预测。
                transforms.append(np.linalg.inv(previous_pose) @ pose)
            previous_pose = pose
            #! 不再 warp GT；状态已经在各步目标 LiDAR 坐标系构建，保留原标签及255。
            targets.append(label.astype(np.int64))
        #! 沿时间维打包；沿用 PackFoundationSSCInputs 的 stack=True，无需改变现有单帧接口。
        results['gt_occ'] = torch.from_numpy(np.stack(targets))
        results['foundation_meta'].update(pc_range=torch.tensor(self.pc_range),
            occ_size=torch.tensor(self.shape), forecast_coordinate='per_frame_lidar',
            forecast_target_to_previous=torch.tensor(np.stack(transforms), dtype=torch.float32))
        return results
