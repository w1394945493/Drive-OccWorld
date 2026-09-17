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
        - img: shape = [history + current, num_cam, C, H, W]，
          Dataset 内部完成读取、normalize、pad 和 HWC->CHW；
        - img_metas: 历史帧 + 当前帧的几何与图像 meta；
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
                 load_img=True,
                 to_float32=True,
                 img_norm_cfg=None,
                 pad_shape=(384, 1248),
                 size_divisor=32,
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
        self.load_img = load_img
        self.to_float32 = to_float32
        #* ================== 图像预处理配置 ==================
        # 第一阶段先采用确定性预处理：
        #   1. mmcv.imread 读取 BGR 图像；
        #   2. 按 Drive-OccWorld / BEVFormer 常用 Caffe 风格做 normalize；
        #   3. 只在右侧和下侧 pad 到固定尺寸，不做 resize/crop/flip。
        # 因为没有改变像素坐标尺度和原点，所以 lidar2img / cam_intrinsic 不需要更新。
        # 如果后续改成 resize/crop，需要同步更新 cam_intrinsic / lidar2img。
        if img_norm_cfg is None:
            img_norm_cfg = dict(
                mean=[103.530, 116.280, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False)
        self.img_norm_cfg = img_norm_cfg
        self.pad_shape = tuple(pad_shape) if pad_shape is not None else None
        self.size_divisor = size_divisor
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

    @staticmethod
    def _quat_wxyz_to_rot(quat):
        """Convert wxyz quaternion to 3x3 rotation matrix."""
        w, x, y, z = np.asarray(quat, dtype=np.float64)
        n = np.sqrt(w * w + x * x + y * y + z * z) + 1e-12
        w, x, y, z = w / n, x / n, y / n, z / n
        return np.asarray([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    @classmethod
    def _pose_matrix(cls, translation, rotation):
        """Build 4x4 pose matrix from translation and wxyz quaternion."""
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = cls._quat_wxyz_to_rot(rotation)
        pose[:3, 3] = np.asarray(translation, dtype=np.float64)
        return pose

    @classmethod
    def _lidar_to_global(cls, info):
        """Build LiDAR -> global transform from info pose fields."""
        ego2global = cls._pose_matrix(
            info['ego2global_translation'], info['ego2global_rotation'])
        lidar2ego = cls._pose_matrix(
            info['lidar2ego_translation'], info['lidar2ego_rotation'])
        return ego2global @ lidar2ego

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

    def _pad_image(self, img):
        """Pad one image on right/bottom and return padded image + shapes."""
        ori_shape = img.shape
        h, w = img.shape[:2]

        if self.pad_shape is not None:
            target_h, target_w = self.pad_shape
            if h > target_h or w > target_w:
                raise ValueError(
                    f'Image shape {(h, w)} is larger than pad_shape '
                    f'{self.pad_shape}. Use a larger pad_shape or add resize.')
        elif self.size_divisor is not None:
            target_h = int(np.ceil(h / self.size_divisor) * self.size_divisor)
            target_w = int(np.ceil(w / self.size_divisor) * self.size_divisor)
        else:
            target_h, target_w = h, w

        pad_shape = (target_h, target_w, img.shape[2])
        if (target_h, target_w) == (h, w):
            return img, ori_shape, pad_shape

        padded = np.zeros(pad_shape, dtype=img.dtype)
        padded[:h, :w, :] = img
        return padded, ori_shape, pad_shape

    def _normalize_image(self, img):
        """Normalize BGR image according to img_norm_cfg."""
        mean = np.asarray(self.img_norm_cfg['mean'], dtype=np.float32)
        std = np.asarray(self.img_norm_cfg['std'], dtype=np.float32)
        if self.img_norm_cfg.get('to_rgb', False):
            img = img[..., ::-1]
        return (img - mean) / std

    def _load_images(self, frame_inputs):
        """Load and preprocess history/current images.

        Returns:
            img: np.ndarray, shape = [T, N, C, H, W]
                T = history_queue_length + 1，N = 1(left) 或 2(stereo)。
            shape_metas: list[list[dict]]
                每个输入时刻、每个相机的 ori/img/pad shape，用于写入 img_metas。
        """
        frame_imgs = []
        shape_metas = []
        for frame_input in frame_inputs:
            cam_imgs = []
            cam_shape_metas = []
            for img_path in frame_input['img_filename']:
                img = mmcv.imread(img_path, flag='color')
                if img is None:
                    raise FileNotFoundError(f'Cannot read image: {img_path}')
                if self.to_float32:
                    img = img.astype(np.float32)
                #* 对齐 Drive-OccWorld 原 pipeline 顺序：
                # NormalizeMultiviewImage -> PadMultiViewImage。
                # 即先对真实图像区域做 normalize，再把右侧/下侧 pad 为 0。
                img = self._normalize_image(img)
                img, ori_shape, pad_shape = self._pad_image(img)
                # HWC -> CHW，对齐 Drive-OccWorld/BEVFormer 模型输入习惯。
                img = img.transpose(2, 0, 1).copy()
                cam_imgs.append(img)
                cam_shape_metas.append(dict(
                    ori_shape=ori_shape,
                    # 当前阶段没有 resize/crop，normalize 不改变图像尺寸，
                    # 所以 img_shape 仍是 pad 前的真实图像大小。
                    img_shape=ori_shape,
                    pad_shape=pad_shape,
                ))
            frame_imgs.append(np.stack(cam_imgs, axis=0))
            shape_metas.append(cam_shape_metas)
        return np.stack(frame_imgs, axis=0), shape_metas

    def _build_img_metas(self, input_frame_inputs, current_input, shape_metas=None):
        """Build meta list for history + current image frames.

        Drive-OccWorld/BEVFormer 会使用 ref_lidar_to_cur_lidar /
        cur_lidar_to_ref_lidar 做历史 BEV 对齐。这里以当前参考帧为 ref，
        根据 converter 写入的 ego2global/lidar2ego 位姿计算相对变换。
        """
        ref_info = self.data_infos[self.token2idx[current_input['token']]]
        ref_lidar2global = self._lidar_to_global(ref_info)
        global2ref_lidar = np.linalg.inv(ref_lidar2global)

        metas = []
        for meta_idx, frame_input in enumerate(input_frame_inputs):
            info = self.data_infos[self.token2idx[frame_input['token']]]
            cur_lidar2global = self._lidar_to_global(info)
            cur_lidar_to_ref_lidar = global2ref_lidar @ cur_lidar2global
            ref_lidar_to_cur_lidar = np.linalg.inv(cur_lidar_to_ref_lidar)

            meta = copy.deepcopy(frame_input)
            if shape_metas is None:
                ori_shape = None
                img_shape = None
                pad_shape = None
            else:
                # 多相机场景下保持 list[tuple]，单目时也是长度为 1 的 list，
                # 这样和 img_filename / lidar2img 的多相机结构一致。
                ori_shape = [cam_meta['ori_shape']
                             for cam_meta in shape_metas[meta_idx]]
                img_shape = [cam_meta['img_shape']
                             for cam_meta in shape_metas[meta_idx]]
                pad_shape = [cam_meta['pad_shape']
                             for cam_meta in shape_metas[meta_idx]]
            meta.update(
                img_shape=img_shape,
                ori_shape=ori_shape,
                pad_shape=pad_shape,
                img_norm_cfg=copy.deepcopy(self.img_norm_cfg),
                cur_lidar_to_ref_lidar=cur_lidar_to_ref_lidar,
                ref_lidar_to_cur_lidar=ref_lidar_to_cur_lidar,
                total_cur2ref_lidar_transform=cur_lidar_to_ref_lidar,
                total_ref2cur_lidar_transform=ref_lidar_to_cur_lidar,
            )
            metas.append(meta)
        return metas

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
        input_frame_inputs = frame_inputs[:current_pos + 1]
        current_input = frame_inputs[current_pos]
        if self.load_img:
            img, shape_metas = self._load_images(input_frame_inputs)
        else:
            img, shape_metas = None, None
        img_metas = self._build_img_metas(
            input_frame_inputs, current_input, shape_metas=shape_metas)

        return dict(
            frame_inputs=frame_inputs,
            input_frame_inputs=input_frame_inputs,
            current_input=current_input,
            img=img,
            img_metas=img_metas,
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
            直接返回包含 img / img_metas / segmentation / frame_inputs 的 dict，
            便于调试历史/未来窗口和第一阶段模型输入是否正确。

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
