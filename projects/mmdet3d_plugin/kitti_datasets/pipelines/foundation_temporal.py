"""当前帧 SSC 输入不变，额外提供历史双目、标定与列向量相对位姿。"""
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from .foundationssc import LoadFoundationSSCStereo, PackFoundationSSCInputs


@PIPELINES.register_module()
class LoadFoundationSSCHistory:
    def __init__(self, history_steps=1, frame_stride=5, frame_period=0.1, input_size=(384, 1280)):
        if history_steps < 1 or frame_stride < 1 or frame_period <= 0:
            raise ValueError('历史帧数、帧间隔和原始帧周期必须为正')
        self.steps, self.stride, self.period = history_steps, frame_stride, frame_period
        self.stereo = LoadFoundationSSCStereo(input_size)

    def __call__(self, results):
        from ..semantic_kitti_world_dataset import SemanticKITTIWorldDataset
        frames = results['input_frame_inputs']  # 历史槽位从旧到新，缺失槽位可填当前帧；最后为当前。
        current = results['current_input']
        if len(frames) != self.steps + 1 or frames[-1]['token'] != current['token']:
            raise ValueError('历史窗口长度/当前帧不一致，请检查 history_queue_length')
        pose_fn = SemanticKITTIWorldDataset._lidar_to_global
        current_pose = pose_fn(current)
        inputs, metas, poses, gaps, history_valid = [], [], [], [], []
        for slot, frame in enumerate(frames[:-1]):
            gap = current['timestamp'] - frame['timestamp']
            #! converter 的 timestamp 是原始帧编号；秒数只能按配置帧周期近似换算。
            padded = frame['token'] == current['token']
            expected_gap = 0 if padded else (self.steps - slot) * self.stride
            if frame['scene_token'] != current['scene_token'] or gap != expected_gap:
                raise ValueError('历史跨序列、非过去帧或关键帧间隔不匹配')
            loaded = self.stereo(dict(current_input=frame))  # 复用原 resize/crop/归一化和每帧双目标定。
            inputs.append(loaded['img_inputs'])
            metas.append(loaded['foundation_meta'])
            poses.append(pose_fn(frame))
            gaps.append(gap)
            history_valid.append(not padded)
        poses = np.stack(poses)
        current_to_history = np.linalg.inv(poses) @ current_pose
        #! 反向采样用 current→history；history→current 供检查/可视化，均为列向量矩阵。
        results['history_img_inputs'] = tuple(torch.stack([item[k] for item in inputs]) for k in range(8))
        results['history_img_metas'] = metas
        results['temporal_metas'] = dict(
            current_to_history=torch.tensor(current_to_history, dtype=torch.float32),
            history_to_current=torch.tensor(np.linalg.inv(current_to_history), dtype=torch.float32),
            history_lidar_to_global=torch.tensor(poses, dtype=torch.float64),
            current_lidar_to_global=torch.tensor(current_pose, dtype=torch.float64),
            history_valid=torch.tensor(history_valid, dtype=torch.bool),
            frame_gaps=torch.tensor(gaps, dtype=torch.long),
            time_offsets_seconds=-torch.tensor(gaps, dtype=torch.float32) * self.period,
            history_tokens=[f['token'] for f in frames[:-1]], current_token=current['token'],
            coordinate_convention='column_vector', time_source='frame_index_times_nominal_period')
        #! 补帧复制当前图像/标定，时间差为 0、相对位姿为单位阵；valid=False 区分真实历史。
        return results


@PIPELINES.register_module()
class PackFoundationSSCTemporalInputs(PackFoundationSSCInputs):
    def __call__(self, results):
        packed = super().__call__(results)  # 当前 GT/辅助监督保留原接口；不加载历史 occupancy。
        if self.runner_format:
            from mmcv.parallel import DataContainer as DC
            packed.update(
                history_img_inputs=tuple(DC(x, stack=True, pad_dims=None) for x in results['history_img_inputs']),
                history_img_metas=DC(results['history_img_metas'], cpu_only=True),
                temporal_metas=DC(results['temporal_metas'], cpu_only=True))
        else:
            packed.update({key: results[key] for key in ('history_img_inputs', 'history_img_metas', 'temporal_metas')})
        return packed
