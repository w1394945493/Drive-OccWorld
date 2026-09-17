import copy
import os.path as osp
import pickle

import mmcv
import numpy as np
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
from torch.utils.data import Dataset


@DATASETS.register_module()
class SemanticKITTIWorldDataset(Dataset):
    """Drive-OccWorld-light SemanticKITTI dataset for stage-1 forecasting.

    设计目标：
        第一阶段只验证「历史/当前图像 -> future occupancy forecasting」链路，
        暂时关闭 action condition / planning / flow / instance-level supervision。

    数据来源：
        ann_file 由 tools/semantickitti_converter.py 生成，里面每一帧 info 已包含：
          - token / prev / next：有 occupancy 标注的关键帧链路；
          - cams：CAM_FRONT_LEFT / CAM_FRONT_RIGHT；
          - occ_path：当前帧 dense occupancy .npy；
          - ego2global / lidar2ego / lidar2img 等基础几何字段。

    相机选择：
        use_camera='left'   : 只使用 image_2 / CAM_FRONT_LEFT；
        use_camera='stereo' : 使用左右双目前视图像。

    说明：
        KITTI/SemanticKITTI 中 image_2 通常是默认主相机；image_3 更多用于
        双目/立体匹配。第一阶段只保留 left 单目 baseline 和 stereo 双目输入，
        不单独提供 right-only 模式，避免无必要的配置分支。

    输出约定：
        - img: 若 pipeline 中包含 LoadMultiViewImageFromFiles，则由 pipeline 加载；
        - img_metas: 由 CustomCollect3D 收集；
        - segmentation: shape = [history + current + future, H, W, D]，
          直接由 occ_path 读取并 stack，供 Drive-OccWorld 第一阶段 occupancy loss 使用。

    注意：
        这是第一阶段轻量 Dataset，不直接复用 NuScenesWorldDatasetTemplate 的原因是：
          1. SemanticKITTI 没有 nuScenes SDK / CAN bus / sample_annotations；
          2. occupancy 路径已经逐帧写入 pkl，无需按 scene_token/lidar_token 拼路径；
          3. 第一阶段只需要图像和 occupancy 序列，规划相关字段先不引入。
    """

    CAMERA_GROUPS = {
        'left': ('CAM_FRONT_LEFT',),
        'stereo': ('CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'),
    }

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 use_camera='left',
                 history_queue_length=2,
                 future_queue_length=4,
                 filter_invalid=True,
                 load_occ=True,
                 test_mode=False):
        super().__init__()
        if use_camera not in self.CAMERA_GROUPS:
            raise ValueError(
                f'use_camera must be one of {list(self.CAMERA_GROUPS)}, '
                f'got {use_camera}')

        self.ann_file = ann_file
        self.data_root = data_root
        self.use_camera = use_camera
        self.camera_names = self.CAMERA_GROUPS[use_camera]
        self.history_queue_length = int(history_queue_length)
        self.future_queue_length = int(future_queue_length)
        self.filter_invalid = filter_invalid
        self.load_occ = load_occ
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline) if pipeline is not None else None

        data = self._load_pkl(ann_file)
        self.metadata = data.get('metadata', {})
        self.data_infos = list(data['infos'])
        self.token2idx = {
            info['token']: idx for idx, info in enumerate(self.data_infos)
        }

        #* ================== 有效样本过滤 ==================
        # 一个训练样本需要：
        #   history_queue_length 帧历史 + 当前帧 + future_queue_length 帧未来。
        # 如果 prev/next 链不足，说明靠近 sequence 边界，默认过滤掉。
        if filter_invalid:
            self.valid_indices = [
                idx for idx in range(len(self.data_infos))
                if self._get_window_indices(idx) is not None
            ]
        else:
            self.valid_indices = list(range(len(self.data_infos)))

    @staticmethod
    def _load_pkl(path):
        if not osp.isfile(path):
            raise FileNotFoundError(f'Missing ann_file: {path}')
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.valid_indices)

    def _trace_tokens(self, start_token, direction, steps):
        """Trace prev/next token chain for a fixed number of steps."""
        tokens = []
        cur_token = start_token
        for _ in range(steps):
            cur_info = self.data_infos[self.token2idx[cur_token]]
            next_token = cur_info[direction]
            if not next_token or next_token not in self.token2idx:
                return None
            tokens.append(next_token)
            cur_token = next_token
        return tokens

    def _get_window_indices(self, raw_index):
        """Return indices of [history..., current, future...] or None.

        prev/next 链是在 converter 中按“有 occupancy 标注的 token 列表”构建的，
        因此这里沿链取到的每一帧都应有 occ_path。
        """
        cur_info = self.data_infos[raw_index]
        cur_token = cur_info['token']

        prev_tokens = self._trace_tokens(
            cur_token, 'prev', self.history_queue_length)
        if prev_tokens is None:
            return None
        next_tokens = self._trace_tokens(
            cur_token, 'next', self.future_queue_length)
        if next_tokens is None:
            return None

        # _trace_tokens('prev') 得到的是 [t-1, t-2, ...]，这里反转成时间顺序。
        window_tokens = list(reversed(prev_tokens)) + [cur_token] + next_tokens
        return [self.token2idx[token] for token in window_tokens]

    def _select_camera_infos(self, info):
        cams = info['cams']
        selected = []
        for cam_name in self.camera_names:
            if cam_name not in cams:
                raise KeyError(f'{info["token"]} does not contain camera {cam_name}')
            selected.append(cams[cam_name])
        return selected

    def _build_single_frame_input(self, info, is_current=False):
        """Build pipeline input dict for one frame."""
        cam_infos = self._select_camera_infos(info)
        img_filename = [cam['data_path'] for cam in cam_infos]
        lidar2img = [np.asarray(cam['lidar2img'], dtype=np.float64)
                     for cam in cam_infos]
        cam_intrinsic = [np.asarray(cam['cam_intrinsic'], dtype=np.float64)
                         for cam in cam_infos]

        input_dict = dict(
            sample_idx=info['token'],
            token=info['token'],
            pts_filename=info.get('lidar_path', None),
            lidar_path=info.get('lidar_path', None),
            filename=img_filename,
            img_filename=img_filename,
            lidar2img=lidar2img,
            cam_intrinsic=cam_intrinsic,
            cams=copy.deepcopy(info['cams']),
            scene_token=info['scene_token'],
            scene_name=info.get('scene_name', info['scene_token']),
            lidar_token=info['token'],
            timestamp=info.get('timestamp', 0),
            prev_idx=info.get('prev', ''),
            next_idx=info.get('next', ''),
            can_bus=info.get('can_bus', np.zeros(18, dtype=np.float32)),
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            occ_path=info['occ_path'],
            # 与 BEVFormer/Drive-OccWorld meta 习惯对齐。
            prev_bev_exists=(not is_current),
        )
        return input_dict

    @staticmethod
    def _load_occ(occ_path):
        """Load dense occupancy label from converter-produced occ_path."""
        if not osp.isfile(occ_path):
            raise FileNotFoundError(f'Missing occupancy file: {occ_path}')
        occ = np.load(occ_path)
        # 保留原始 shape；后续 config/模型里再决定是否 resize / remap 类别。
        return occ.astype(np.int64, copy=False)

    def get_data_info(self, index):
        """Return raw window information before pipeline processing."""
        raw_index = self.valid_indices[index]
        window_indices = self._get_window_indices(raw_index)
        if window_indices is None:
            raise RuntimeError(
                f'Invalid window at raw index {raw_index}. '
                'Set filter_invalid=True to filter boundary samples.')

        current_pos = self.history_queue_length
        frame_inputs = []
        occ_seq = []
        for pos, frame_idx in enumerate(window_indices):
            info = self.data_infos[frame_idx]
            frame_inputs.append(self._build_single_frame_input(
                info, is_current=(pos == current_pos)))
            if self.load_occ:
                occ_seq.append(self._load_occ(info['occ_path']))

        current_info = self.data_infos[raw_index]
        return dict(
            frame_inputs=frame_inputs,
            current_input=frame_inputs[current_pos],
            segmentation=np.stack(occ_seq) if self.load_occ else None,
            window_tokens=[self.data_infos[i]['token'] for i in window_indices],
            current_token=current_info['token'],
        )

    def __getitem__(self, index):
        """Get one sample.

        若提供 pipeline：
            目前只对当前帧 current_input 执行 pipeline，并把 segmentation
            附加回输出。这适合先验证单帧/当前帧图像读取链路。

        若不提供 pipeline：
            直接返回包含 frame_inputs 和 segmentation 的原始 dict，便于调试
            历史/未来窗口是否正确。

        后续若要完全贴合 Drive-OccWorld 的 [history,current] 图像队列格式，
        可在此基础上增加 queue 合并逻辑。
        """
        data = self.get_data_info(index)
        if self.pipeline is None:
            return data

        results = self.pipeline(data['current_input'])
        results['segmentation'] = data['segmentation']
        results['window_tokens'] = data['window_tokens']
        results['current_token'] = data['current_token']
        return results

    def evaluate(self, results, **kwargs):
        """Placeholder for compatibility with MMDetection dataset API."""
        return {}
