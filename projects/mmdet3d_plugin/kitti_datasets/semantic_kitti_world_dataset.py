import copy
import os.path as osp
import pickle

import mmcv
import numpy as np
from prettytable import PrettyTable
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
import torch
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
          时序 pipeline 完成读取、normalize、pad 和 HWC->CHW；
        - img_metas: 历史帧 + 当前帧的几何与图像 meta；
        - segmentation: shape = [history + current + future, H, W, D]，
          直接由 occ_path 读取并 stack，供 Drive-OccWorld 第一阶段 occupancy loss 使用。
        - sdc_planning / command / vel_steering:
          第一阶段为了跑通 Drive-OccWorld 的 future_pred() 接口构造的兼容字段。
          它们不是 nuScenes CAN bus/planner 的真实监督，只用于最小链路调试。

    注意：
        这是第一阶段轻量 Dataset，不直接复用 NuScenesWorldDatasetTemplate 的原因是：
          1. SemanticKITTI 没有 nuScenes SDK / CAN bus / sample_annotations；
          2. occupancy 路径已经逐帧写入 pkl，无需按 scene_token/lidar_token 拼路径；
          3. 第一阶段只需要图像和 occupancy 序列；规划/action 字段仅构造
             最小 dummy/pseudo 输入，用于跑通原 Drive-OccWorld forward。
    """

    CAMERA_GROUPS = {
        'left': ('CAM_FRONT_LEFT',),
        'stereo': ('CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'),
    }

    CLASSES = (
        'empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
        'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
        'other-ground', 'building', 'fence', 'vegetation', 'trunk',
        'terrain', 'pole', 'traffic-sign')
    PALETTE = None

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
                 empty_idx=0,
                 max_samples=None,
                 format_for_train=False,
                 test_mode=False,
                 pad_history_with_current=False):
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
        #* SemanticKITTI occupancy 关键帧按约 2Hz 组织，相邻 token 间隔约 0.5s。
        #* EvalHook 写 .log.json 时会用该字段把 step_1/step_2/... 转成
        #* 0.5s/1s/... 这样的更直观时间标签。
        self.forecast_time_interval = 0.5
        self.filter_invalid = filter_invalid
        #! 时序 SSC 可选：历史不足时用当前帧补齐；默认关闭，保留其他任务的过滤行为。
        self.pad_history_with_current = pad_history_with_current
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
        self.empty_idx = int(empty_idx)
        self.max_samples = max_samples
        #* format_for_train=True 时，Dataset 会把输出包装成 MMDetection/MMCV
        #* train.py 期望的 DataContainer 格式，只保留 Drive_OccWorld.forward_train
        #* 真正接收的字段；False 时保留完整调试字段，方便脚本直接检查样本内容。
        self.format_for_train = format_for_train
        self.test_mode = test_mode
        #* 默认提供等价 BEVFormer 时序处理链，兼容旧配置 pipeline=None。
        # 显式 pipeline 也接收完整窗口；format_for_train 不再提前绕过 pipeline。
        if pipeline is None:
            pipeline = [dict(type=name) for name in (
                'LoadTemporalKittiImages', 'NormalizeTemporalKittiImages',
                'PadTemporalKittiImages', 'LoadTemporalKittiOccupancy',
                'PackKittiWorldInputs')]
        self.pipeline = Compose(pipeline)
        #! SemanticKITTI 第一阶段评估已经在 evaluate() 内部打印 compact table。
        #! 如果继续让 MMCV TextLoggerHook 把 eval_results 作为普通训练日志
        #! 再打印一遍，会出现类似 Epoch [1][5/10]、time=0、data_time=0、
        #! memory 异常等误导信息。因此让自定义 EvalHook 在 evaluate() 后
        #! 清理 log_buffer，保留 compact table，跳过二次 TextLoggerHook 输出。
        self.suppress_eval_log_buffer = True

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
        #* ================== 快速调试/快速评估样本数截断 ==================
        # max_samples 只截断当前 Dataset 的有效样本列表，不改 pkl，也不影响
        # prev/next 时序窗口构造。常用于：
        #   - data.val.max_samples=20：快速测试 EvalHook / evaluate() 是否正常；
        #   - data.train.max_samples=100：快速 overfit/debug 一小段训练数据。
        # 默认 None 表示使用完整 split。
        if self.max_samples is not None:
            self.valid_indices = self.valid_indices[:int(self.max_samples)]
        #! tools/train.py 在非分布式训练时会使用 mmdet GroupSampler，
        #! 该 sampler 要求 dataset.flag 存在。SemanticKITTI 第一阶段不做
        #! aspect-ratio 分组，统一置 0 即可。
        self.flag = np.zeros(len(self.valid_indices), dtype=np.uint8)

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
            #! 历史不足：未开启补帧则返回无效窗口；开启后保留该样本，用当前帧填满缺失槽位。
            if not self.pad_history_with_current:
                return None
            #! 重新沿 prev 链收集已有历史；上面的 _trace_tokens 在不足时只返回 None，不保留部分结果。
            prev_tokens = []  # 暂按由近到远存放真实历史：[t-1, t-2, ...]。
            token = cur_token  # 从当前帧开始向前查找，不是从整个数据列表直接减索引。
            for _ in range(self.history_queue_length):
                previous = self.data_infos[self.token2idx[token]]['prev']  # 上一个关键帧的 token。
                if not previous or previous not in self.token2idx:
                    break  # 到达 prev 链边界或引用缺失时停止；依赖 PKL 的 prev 链不跨场景。
                prev_tokens.append(previous)  # 保留真实历史，不用当前帧覆盖已有历史。
                token = previous  # 继续查找更早的一帧。
            #! 只补缺失的历史槽位：重复当前 token，让后续 pipeline 加载当前图像/标定，不伪造历史位姿。
            # 此时补在列表末尾；下方 reversed 后位于窗口开头，即最早的缺失槽位。
            # 例：需要 3 帧历史、只有 t-1，补后为 [t-1,t,t]，反转后为 [t,t,t-1]。
            # 时序 pipeline 会将这些副本标记 history_valid=False，时间差为 0；未来 GT 不在这里补齐。
            prev_tokens += [cur_token] * (self.history_queue_length - len(prev_tokens))
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

    @staticmethod
    def _to_homogeneous_4x4(mat):
        """Convert KITTI-style 3x4 projection matrix to BEVFormer-style 4x4.

        #! 修复原因：
        #! SemanticKITTI/KITTI 标定中 lidar2img 通常是 P2 @ Tr_velo_to_cam，
        #! shape 为 3x4；但 Drive-OccWorld 复用的 BEVFormer encoder
        #! point_sampling() 会把 lidar2img reshape 成 (..., 4, 4)。
        #! 因此这里在 Dataset 层补齐最后一行 [0,0,0,1]，使其符合
        #! nuScenes/BEVFormer 的 4x4 齐次矩阵协议。
        """
        mat = np.asarray(mat, dtype=np.float64)
        if mat.shape == (4, 4):
            return mat
        if mat.shape == (3, 4):
            mat4 = np.eye(4, dtype=np.float64)
            mat4[:3, :4] = mat
            return mat4
        raise ValueError(f'Unsupported lidar2img shape: {mat.shape}')

    def _build_single_frame_input(self, info, is_current=False):
        """Build pipeline input dict for one frame."""
        cam_infos = self._select_camera_infos(info)
        img_filename = [cam['data_path'] for cam in cam_infos]
        lidar2img = [self._to_homogeneous_4x4(cam['lidar2img'])
                     for cam in cam_infos]
        cam_intrinsic = [np.asarray(cam['cam_intrinsic'], dtype=np.float64)
                         for cam in cam_infos]
        #! 修复原因：
        #! BEVFormer PerceptionTransformer 在 rotate_prev_bev=True 时会调用
        #! torchvision.transforms.functional.rotate，并传入
        #! img_meta['can_bus'][-1] 作为旋转角。torchvision 对 angle 类型
        #! 检查较严格，只接受 Python int/float，不接受 np.float32/np.float64。
        #! SemanticKITTI 第一阶段 can_bus 是 dummy 占位，因此这里统一转成
        #! Python float list，既保持原 18 维协议，又避免 rotate 类型报错。
        can_bus = np.asarray(
            info.get('can_bus', np.zeros(18, dtype=np.float32)),
            dtype=np.float32)
        can_bus = [float(x) for x in can_bus]

        input_dict = dict(
            sample_idx=info['token'],
            token=info['token'],
            pts_filename=info.get('lidar_path', None),
            lidar_path=info.get('lidar_path', None),
            #* （FoundationSSC 辅助深度&语义损失) 仅透传路径；读取、解码交给 pipeline。
            pts_label_path=info.get('pts_label_path', None),
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
            can_bus=can_bus,
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
        #* 旧实现仅保留用于等价性对照；正式 __getitem__ 已改用 pipeline。
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
        #* 旧实现仅用于对照；正式路径由 NormalizeTemporalKittiImages 处理。
        """Normalize BGR image according to img_norm_cfg."""
        mean = np.asarray(self.img_norm_cfg['mean'], dtype=np.float32)
        std = np.asarray(self.img_norm_cfg['std'], dtype=np.float32)
        if self.img_norm_cfg.get('to_rgb', False):
            img = img[..., ::-1]
        return (img - mean) / std

    def _load_images(self, frame_inputs):
        #* 旧实现仅用于对照；正式路径由时序图像 pipeline 处理。
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
            global2cur_lidar = np.linalg.inv(cur_lidar2global)

            #*(0918) SemanticKITTI converter 中保存的是标准 column-vector pose。
            #* 因此这里先按常规几何定义计算 column-vector 形式的相对变换：
            #*   cur_lidar_to_ref_lidar_col: 当前/历史帧 lidar -> ref/current lidar；
            #*   ref_lidar_to_cur_lidar_col: ref/current lidar -> 当前/历史帧 lidar。
            #*   p_global = T_global_lidar @ p_lidar。
            cur_lidar_to_ref_lidar_col = global2ref_lidar @ cur_lidar2global
            ref_lidar_to_cur_lidar_col = np.linalg.inv(
                cur_lidar_to_ref_lidar_col)

            #*(0918) 转成 Drive-OccWorld 使用的 row-vector 右乘矩阵格式。
            #* 原 nuScenes Dataset 也是转成 row-vector 形式保存，后续使用方式是：
            #*   aligned_bev_coords = aligned_bev_coords @ transform
            #* 如果不转置，平移项位于最后一列，但 row-vector 右乘期望平移在最后一行，
            #* 会导致历史 BEV / memory BEV 坐标对齐不正确。
            cur_lidar_to_ref_lidar = cur_lidar_to_ref_lidar_col.T
            ref_lidar_to_cur_lidar = ref_lidar_to_cur_lidar_col.T

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
                #! 修复原因：
                #! Drive-OccWorld 复用 BEVFormer 的 PerceptionTransformer，
                #! get_bev_features() 会读取 img_meta['lidar2global_rotation']
                #! 将 can_bus 中的 global 位移转换到当前 LiDAR 坐标系，
                #! 用于计算 BEV shift / rotate_prev_bev。nuScenes pkl 原本会
                #! 提供这些字段；SemanticKITTI converter 只保存了
                #! ego2global/lidar2ego，因此这里在 Dataset 中在线补齐。
                lidar2global_rotation=cur_lidar2global[:3, :3],
                lidar2global_translation=cur_lidar2global[:3, 3],
                global2lidar_rotation=global2cur_lidar[:3, :3],
                global2lidar_translation=global2cur_lidar[:3, 3],
                cur_lidar_to_ref_lidar=cur_lidar_to_ref_lidar,
                ref_lidar_to_cur_lidar=ref_lidar_to_cur_lidar,
                total_cur2ref_lidar_transform=cur_lidar_to_ref_lidar,
                total_ref2cur_lidar_transform=ref_lidar_to_cur_lidar,
            )
            metas.append(meta)
        return metas

    def _add_future_transforms_to_current_meta(
            self, img_metas, window_indices, current_pos):
        """Add future<->reference lidar transforms to current-frame meta.

        #! 修复原因：
        #! Drive-OccWorld.future_pred() 在预测未来 BEV 时会调用
        #! _align_bev_coordnates()，并从当前参考帧 img_meta 中读取：
        #!   - future2ref_lidar_transform[future_frame_index]
        #!   - ref2future_lidar_transform[future_frame_index]
        #! 原 nuScenes Dataset 会在 union2one/queue 合并阶段预先写入这些字段；
        #! SemanticKITTI 第一阶段 Dataset 目前只构造了历史/当前帧相对参考帧
        #! 的变换，所以这里基于完整时序窗口在线补齐未来帧到参考帧的变换。
        #!
        #! 注意 future_pred() 的 future_frame_index 从 1 开始，因此这里第 0
        #! 个元素放 identity，表示 current/ref -> current/ref；第 1..N 个
        #! 元素分别对应 t+1, t+2, ... 的未来帧。
        """
        ref_info = self.data_infos[window_indices[current_pos]]
        ref_lidar2global = self._lidar_to_global(ref_info)
        global2ref_lidar = np.linalg.inv(ref_lidar2global)

        future2ref_lidar_transform = [np.eye(4, dtype=np.float64)]
        ref2future_lidar_transform = [np.eye(4, dtype=np.float64)]

        for frame_idx in window_indices[current_pos + 1:]:
            future_info = self.data_infos[frame_idx]
            future_lidar2global = self._lidar_to_global(future_info)
            #*(0918) 先按标准 column-vector 约定计算未来帧和当前参考帧的相对位姿。
            #*   future2ref_col: future lidar -> ref/current lidar；
            #*   ref2future_col: ref/current lidar -> future lidar。
            future2ref_col = global2ref_lidar @ future_lidar2global
            ref2future_col = np.linalg.inv(future2ref_col)

            #*(0918) 转成 Drive-OccWorld 使用的 row-vector 右乘矩阵格式。
            #* _align_bev_coordnates() 中实际使用：
            #*   [x, y, z, 1] @ future2ref @ ref_to_history
            #* 因此这里和历史帧变换保持一致，把 column-vector 矩阵转置后保存。
            future2ref = future2ref_col.T
            ref2future = ref2future_col.T
            future2ref_lidar_transform.append(future2ref)
            ref2future_lidar_transform.append(ref2future)

        # 写到当前参考帧 meta 上。Drive-OccWorld.forward_train() 会在当前帧
        # img_metas = [each[num_frames-1] for each in img_metas] 后保留这个 meta。
        img_metas[-1]['future2ref_lidar_transform'] = np.stack(
            future2ref_lidar_transform, axis=0)
        img_metas[-1]['ref2future_lidar_transform'] = np.stack(
            ref2future_lidar_transform, axis=0)
        return img_metas

    @staticmethod
    def _load_occ(occ_path):
        #* 旧实现仅用于对照；正式路径由 LoadTemporalKittiOccupancy 处理。
        """Load dense occupancy label from converter-produced occ_path."""
        if not osp.isfile(occ_path):
            raise FileNotFoundError(f'Missing occupancy file: {occ_path}')
        occ = np.load(occ_path)
        # 保留原始 shape；后续 config/模型里再决定是否 resize / remap 类别。
        return occ.astype(np.int64, copy=False)

    def _build_stage1_drive_occworld_compat_fields(self, current_info):
        """Build pseudo fields required by Drive-OccWorld forward.

        #* ================== 第一阶段兼容字段：只为跑通链路 ==================
        Drive-OccWorld 原始 nuScenes 训练链路即使关闭 turn_on_plan，也会在
        future_pred() 中使用 sdc_planning 构造 plan_traj，用于未来 BEV query
        与历史 BEV memory 的坐标对齐：

            plan_traj = sdc_planning[:, :future_frame_index, :2]

        SemanticKITTI 第一阶段暂不验证真实 action-conditioned forecasting 和
        occupancy-based planning，因此这里仅根据 converter 写入的
        gt_ego_fut_trajs 构造 pseudo sdc_planning，并补齐 command /
        vel_steering 等接口字段。

        注意：
            - sdc_planning 不是网络预测，也不是 nuScenes CAN bus 真值；
            - vel_steering 全 0，只是占位 dummy action state；
            - sample_traj / gt_future_boxes / flow / instance 当前不构造，
              因为 turn_on_plan=False、turn_on_flow=False 时不会使用。
        """
        # Drive-OccWorld 中 command 常见 shape=(future_pred_frame_num+1,)。
        # 当前 converter 已按 nuScenes v2 风格写入 shape=(5,)。
        command = np.asarray(
            current_info.get(
                'command',
                np.full((self.future_queue_length + 1,), 2, dtype=np.int64)),
            dtype=np.int64)

        future_steps = int(command.shape[0])

        # gt_ego_fut_trajs: converter 中由 SemanticKITTI pose 差分得到，
        # shape 通常为 (6, 2)。这里取前 future_steps 步作为 pseudo 未来位移。
        gt_ego_fut_trajs = np.asarray(
            current_info.get('gt_ego_fut_trajs',
                             np.zeros((future_steps, 2), dtype=np.float32)),
            dtype=np.float32)

        # sdc_planning: Drive-OccWorld 期望每步至少有 [dx, dy, dyaw]。
        # 第一阶段先只填 dx/dy，dyaw 置 0，保证 future_pred() 可以使用
        # [:, :, :2] 完成未来 BEV 坐标对齐。
        sdc_planning = np.zeros((future_steps, 3), dtype=np.float32)
        copy_steps = min(future_steps, gt_ego_fut_trajs.shape[0])
        sdc_planning[:copy_steps, :2] = gt_ego_fut_trajs[:copy_steps, :2]

        # sdc_planning_mask: 第一阶段统一置 1，表示这些 pseudo step 可用。
        # 如果后续要严格处理序列尾部，可结合 fut_valid_flag 或窗口有效性细化。
        sdc_planning_mask = np.ones((future_steps,), dtype=np.float32)

        # vel_steering: Drive-OccWorld action condition 的细粒度自车状态字段。
        # nuScenes 中可包含速度/角速度/转向等；SemanticKITTI 没有 CAN bus，
        # 当前先用全 0 dummy 值占位，只为跑通原模型接口。
        vel_steering = np.zeros((future_steps, 4), dtype=np.float32)

        return dict(
            sdc_planning=sdc_planning,
            sdc_planning_mask=sdc_planning_mask,
            command=command,
            vel_steering=vel_steering,
        )

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
        for pos, frame_idx in enumerate(window_indices):
            info = self.data_infos[frame_idx]
            frame_inputs.append(self._build_single_frame_input(
                info, is_current=(pos == current_pos)))

        current_info = self.data_infos[raw_index]
        input_frame_inputs = frame_inputs[:current_pos + 1]
        current_input = frame_inputs[current_pos]
        img_metas = self._build_img_metas(
            input_frame_inputs, current_input)
        img_metas = self._add_future_transforms_to_current_meta(
            img_metas, window_indices, current_pos)
        compat_fields = self._build_stage1_drive_occworld_compat_fields(
            current_info)

        return dict(
            frame_inputs=frame_inputs,
            input_frame_inputs=input_frame_inputs,
            current_input=current_input,
            img=None,
            img_metas=img_metas,
            segmentation=None,
            occ_paths=[frame['occ_path'] for frame in frame_inputs],
            preprocess_cfg=dict(
                load_img=self.load_img, load_occ=self.load_occ,
                to_float32=self.to_float32, img_norm_cfg=copy.deepcopy(self.img_norm_cfg),
                pad_shape=self.pad_shape, size_divisor=self.size_divisor,
                format_for_train=self.format_for_train),
            #* 这些字段是为了兼容 Drive-OccWorld 原 forward 接口的
            #* pseudo/dummy 字段；第一阶段关闭 planning/action ablation 时，
            #* 它们只用于让 future_pred() 的对齐逻辑先跑通。
            **compat_fields,
            window_tokens=[self.data_infos[i]['token'] for i in window_indices],
            current_token=current_info['token'],
        )

    def __getitem__(self, index):
        """Dataset 整理完整窗口，pipeline 加载/预处理/打包；不读取未来图像。"""
        return self.pipeline(self.get_data_info(index))

    def _format_for_train(self, data):
        #* 旧实现仅用于对照；正式路径由 PackKittiWorldInputs 打包。
        """Format one sample for MMDetection/MMCV train.py.

        #! 修复/适配原因：
        #! 调试模式下返回的 frame_inputs/window_tokens/current_token 等字段
        #! 有助于人工检查，但 Drive_OccWorld.forward_train() 不接收这些参数。
        #! 如果直接交给 tools/train.py，它们会被作为多余 kwargs 传入模型。
        #! 因此正式训练模式只保留模型 forward_train 需要的字段。
        #
        #! img_metas 和 segmentation 不能被默认 collate 随意 stack：
        #! - img_metas 需要保持 list[dict]，放在 CPU；
        #! - segmentation 需要保持 list[tensor]，因为原 Drive-OccWorld
        #!   compute_occ_loss 使用 segmentation[0] 作为 [T,H,W,D]。
        """
        from mmcv.parallel import DataContainer as DC

        formatted = dict(
            img=DC(torch.from_numpy(data['img']).float(), stack=True),
            img_metas=DC(data['img_metas'], cpu_only=True),
            segmentation=DC(
                torch.from_numpy(data['segmentation']).long(),
                stack=False),
            #! MMCV DataContainer 在 stack=True 时默认 pad_dims=2，
            #! 适合图像类高维张量；但 sdc_planning/sdc_planning_mask/
            #! command/vel_steering 是二维或一维动作/轨迹张量。
            #! 若不显式设置 pad_dims=None，mmcv.parallel.collate 会检查
            #! ndim > pad_dims，导致 shape=[5,3] 或 [5] 的字段触发
            #! AssertionError。因此这些非图像张量只需要直接 stack，
            #! 不做按 H/W 维度 padding。
            sdc_planning=DC(
                torch.from_numpy(data['sdc_planning']).float(),
                stack=True,
                pad_dims=None),
            sdc_planning_mask=DC(
                torch.from_numpy(data['sdc_planning_mask']).float(),
                stack=True,
                pad_dims=None),
            command=DC(
                torch.from_numpy(data['command']).long(),
                stack=True,
                pad_dims=None),
            vel_steering=DC(
                torch.from_numpy(data['vel_steering']).float(),
                stack=True,
                pad_dims=None),
        )
        return formatted

    @staticmethod
    def _as_numpy_hist(hist):
        """Convert one confusion matrix-like result to numpy array."""
        if torch.is_tensor(hist):
            hist = hist.detach().cpu().numpy()
        return np.asarray(hist, dtype=np.float64)

    @classmethod
    def _collect_result_values(cls, results, key):
        """Collect ``key`` values from MMDet eval outputs.

        Drive_OccWorld.forward_test() returns a dict for each sample:
            {
              'hist_for_iou': ...,
              'hist_for_iou_current': ...,
              'hist_for_iou_future': ...,
              ...
            }

        Depending on the eval hook / test API, Dataset.evaluate() may receive:
          1. list[dict]: standard MMDet single_gpu_test style；
          2. dict[list]: some custom hooks may pre-aggregate by key。

        This helper normalizes both forms into list[value].
        """
        if results is None:
            return []
        if isinstance(results, dict):
            value = results.get(key, [])
            return value if isinstance(value, list) else [value]
        if isinstance(results, (list, tuple)):
            values = []
            for item in results:
                if isinstance(item, dict) and key in item:
                    values.append(item[key])
            return values
        return []

    @classmethod
    def _sum_histograms(cls, hist_values):
        """Sum confusion matrices while skipping invalid placeholders."""
        hist_sum = None
        for hist in hist_values:
            # Drive_OccWorld.forward_test() 的 only_generate_dataset 分支可能返回 0。
            if hist is None:
                continue
            if np.isscalar(hist):
                continue
            hist_np = cls._as_numpy_hist(hist)
            if hist_np.ndim != 2:
                continue
            hist_sum = hist_np if hist_sum is None else hist_sum + hist_np
        return hist_sum

    @staticmethod
    def _hist_to_ious(hist):
        """Convert confusion matrix to per-class IoU.

        hist[gt, pred] is produced by Drive_OccWorld.fast_hist().
        For class c:
            TP    = hist[c, c]
            gt    = hist[c, :].sum()
            pred  = hist[:, c].sum()
            union = pred + gt - TP

        #* 这里实现的是“每一类 IoU / per-class IoU”。
        对每个类别 c 单独计算：

            IoU_c = intersection_c / union_c
                  = TP_c / (Pred_c + GT_c - TP_c)

        返回的 ious 是长度为 num_classes 的数组，例如 SemanticKITTI 下：

            ious[0] -> empty 类 IoU
            ious[1] -> car 类 IoU
            ...
            ious[19] -> traffic-sign 类 IoU

        #! 注意：这里还没有计算 mIoU；mIoU 是在 _format_iou_table()
        #! 中对这些 per-class IoU 再求平均得到的。
        """
        num_classes = hist.shape[0]
        ious = np.full((num_classes,), np.nan, dtype=np.float64)
        for cls_idx in range(num_classes):
            # 当前类别预测正确的 voxel 数，即 intersection。
            tp = hist[cls_idx, cls_idx]
            # 当前类别在 GT 中真实存在的 voxel 总数。
            gt = hist[cls_idx, :].sum()
            # 当前类别被模型预测出来的 voxel 总数。
            pred = hist[:, cls_idx].sum()
            # IoU 分母：预测集合 ∪ GT 集合。
            union = pred + gt - tp
            if union > 0:
                #* per-class IoU 计算位置。
                ious[cls_idx] = tp / union
        return ious

    def _format_iou_table(self, ious, title):
        """Build pretty table and metric dict for SemanticKITTI IoU.

        #* 这里负责两件事：
        #* 1. 把 _hist_to_ious() 得到的每类 IoU 写入日志表格和 metric_dict；
        #* 2. 计算 mIoU，也就是对多个类别 IoU 求平均。

        当前 mIoU 的定义：

            mIoU = mean(IoU_c for c in non-empty classes with valid union)

        也就是默认不把 empty/free space 类计入 SemanticKITTI 语义 mIoU。
        这样可以避免 empty 类占比过大导致指标虚高。
        """
        table = PrettyTable()
        table.field_names = ['class_id', 'class', 'IoU']

        metric_dict = {}
        # valid_non_empty 收集所有“非 empty 且有效”的类别 IoU，
        # 后面会对它求平均得到 mIoU。
        valid_non_empty = []
        for cls_idx, cls_name in enumerate(self.CLASSES):
            if cls_idx >= len(ious):
                break
            iou = ious[cls_idx]
            iou_value = float(iou) if np.isfinite(iou) else np.nan
            #* 每类 IoU 日志表格输出位置。
            table.add_row([
                cls_idx,
                cls_name,
                'nan' if np.isnan(iou_value) else round(iou_value, 4)
            ])
            #* 每类 IoU 指标写入位置，例如 current/IoU_car。
            metric_dict[f'{title}/IoU_{cls_name}'] = iou_value
            # SemanticKITTI mIoU 通常不把 empty/free 类计入语义 mIoU。
            if cls_idx != self.empty_idx and np.isfinite(iou_value):
                valid_non_empty.append(iou_value)

        if valid_non_empty:
            #* mIoU 计算位置：对非 empty 类别的 per-class IoU 求平均。
            miou = float(np.mean(valid_non_empty))
        else:
            miou = np.nan
        #* mIoU 日志表格输出位置。
        table.add_row(['-', 'mIoU(non-empty)', 'nan' if np.isnan(miou) else round(miou, 4)])
        #* mIoU 指标写入位置，例如 current/mIoU、future/mIoU。
        metric_dict[f'{title}/mIoU'] = miou

        # empty 类 IoU 仍单独记录，方便观察 free-space/empty 预测是否异常；
        # 但它不参与上面的 mIoU。
        if len(ious) > self.empty_idx and np.isfinite(ious[self.empty_idx]):
            metric_dict[f'{title}/IoU_empty'] = float(ious[self.empty_idx])
        return table, metric_dict

    def _semantic_hist_to_binary_hist(self, hist):
        """Merge semantic confusion matrix into binary empty/occupied matrix.

        #* 这里实现“二值占据 IoU”所需的 confusion matrix 合并。
        SemanticKITTI 语义 hist 是 num_classes x num_classes，其中：
          - empty_idx 表示 empty/free space；
          - 其他所有类别都表示 occupied。

        合并后 binary_hist 的类别定义为：
          - 0: empty/free；
          - 1: occupied，即所有非 empty 语义类别的并集。

        注意 hist 的行列约定来自 Drive_OccWorld.fast_hist()：
            hist[gt, pred]

        因此：
          binary_hist[0, 0] = GT empty 且 Pred empty；
          binary_hist[0, 1] = GT empty 但 Pred occupied；
          binary_hist[1, 0] = GT occupied 但 Pred empty；
          binary_hist[1, 1] = GT occupied 且 Pred occupied。
        """
        num_classes = hist.shape[0]
        empty = int(self.empty_idx)
        occupied_indices = [
            cls_idx for cls_idx in range(num_classes)
            if cls_idx != empty
        ]

        binary_hist = np.zeros((2, 2), dtype=np.float64)
        binary_hist[0, 0] = hist[empty, empty]
        binary_hist[0, 1] = hist[empty, occupied_indices].sum()
        binary_hist[1, 0] = hist[occupied_indices, empty].sum()
        binary_hist[1, 1] = hist[np.ix_(occupied_indices, occupied_indices)].sum()
        return binary_hist

    @staticmethod
    def _format_binary_iou_table(binary_ious, title):
        """Build table and metric dict for binary empty/occupied IoU.

        #* 这里实现“空/非空占据预测”的指标输出。
        binary_ious[0] 是 empty/free IoU；
        binary_ious[1] 是 occupied IoU；
        binary_mIoU 是二者平均值。
        """
        table = PrettyTable()
        table.field_names = ['binary_class_id', 'binary_class', 'IoU']

        class_names = ['empty', 'occupied']
        metric_dict = {}
        valid_ious = []
        for cls_idx, cls_name in enumerate(class_names):
            iou = binary_ious[cls_idx]
            iou_value = float(iou) if np.isfinite(iou) else np.nan
            table.add_row([
                cls_idx,
                cls_name,
                'nan' if np.isnan(iou_value) else round(iou_value, 4)
            ])
            metric_dict[f'{title}/binary_IoU_{cls_name}'] = iou_value
            if np.isfinite(iou_value):
                valid_ious.append(iou_value)

        binary_miou = float(np.mean(valid_ious)) if valid_ious else np.nan
        table.add_row([
            '-',
            'binary_mIoU(empty+occupied)',
            'nan' if np.isnan(binary_miou) else round(binary_miou, 4)
        ])
        metric_dict[f'{title}/binary_mIoU'] = binary_miou
        return table, metric_dict

    def _semantic_miou_from_hist(self, hist):
        """Return non-empty semantic mIoU from one confusion matrix."""
        ious = self._hist_to_ious(hist)
        valid = [
            float(iou) for cls_idx, iou in enumerate(ious)
            if cls_idx != self.empty_idx and np.isfinite(iou)
        ]
        return float(np.mean(valid)) if valid else np.nan

    def _binary_occupied_iou_from_hist(self, hist):
        """Return occupied IoU after merging all non-empty classes."""
        binary_hist = self._semantic_hist_to_binary_hist(hist)
        binary_ious = self._hist_to_ious(binary_hist)
        return float(binary_ious[1]) if len(binary_ious) > 1 and np.isfinite(binary_ious[1]) else np.nan

    def _collect_per_frame_histograms(self, results):
        """Sum per-frame confusion matrices across samples.

        Drive_OccWorld.forward_test() returns:
            hist_for_iou_per_frame = [hist_t0, hist_t1, ...]

        EvalHook gathers these into list[dict]. This helper converts them into:
            [sum_hist_t0, sum_hist_t1, ...]
        """
        per_sample_values = self._collect_result_values(
            results, 'hist_for_iou_per_frame')
        if not per_sample_values:
            return []

        per_frame_sums = []
        for sample_values in per_sample_values:
            if sample_values is None or np.isscalar(sample_values):
                continue
            for frame_idx, hist in enumerate(sample_values):
                hist_np = self._as_numpy_hist(hist)
                if hist_np.ndim != 2:
                    continue
                while len(per_frame_sums) <= frame_idx:
                    per_frame_sums.append(None)
                per_frame_sums[frame_idx] = (
                    hist_np if per_frame_sums[frame_idx] is None
                    else per_frame_sums[frame_idx] + hist_np)

        return [hist for hist in per_frame_sums if hist is not None]

    @staticmethod
    def _pct(value):
        """Format ratio as percentage for compact paper-style tables."""
        if value is None or not np.isfinite(value):
            return 'nan'
        return round(float(value) * 100.0, 2)

    def _format_compact_forecast_table(self, per_frame_hists):
        """Build compact table: current/future-step mIoU and occupied IoU.

        #* 输出格式参考论文表格：
        #*   mIoU(%): 0, 1, 2, ..., Avg.
        #*   IoU(%):  0, 1, 2, ..., Avg.
        #
        其中：
          - 0 表示当前参考帧；
          - 1/2/3/... 表示第几个未来预测 step；
          - mIoU 是非 empty 语义 mIoU；
          - IoU 是二值 occupied IoU，即所有非 empty 类合并后的占据 IoU；
          - Avg. 是当前帧 + 所有未来步的平均值，包含 0-step。
        """
        table = PrettyTable()
        step_names = [str(i) for i in range(len(per_frame_hists))]
        #* PrettyTable 不允许初始化列数后再改变列数；单/多帧表头只设置一次。
        single_frame = len(per_frame_hists) == 1
        table.field_names = (['metric', 'current'] if single_frame
                             else ['metric'] + step_names + ['Avg.'])

        miou_values = [
            self._semantic_miou_from_hist(hist)
            for hist in per_frame_hists
        ]
        occ_iou_values = [
            self._binary_occupied_iou_from_hist(hist)
            for hist in per_frame_hists
        ]

        #* 单帧 SSC 仅显示 current；多帧 forecasting 的表格和指标键保持原样。
        if single_frame:
            table.add_row(['mIoU(%)', self._pct(miou_values[0])])
            table.add_row(['IoU(%)', self._pct(occ_iou_values[0])])
            return table, dict(current_mIoU=miou_values[0], current_IoU=occ_iou_values[0])

        #* Avg. 计算当前 + 未来所有 step 的平均值。
        # 也就是 step_0, step_1, ..., step_N 全部参与平均。
        valid_mious = [v for v in miou_values if np.isfinite(v)]
        valid_occ_ious = [v for v in occ_iou_values if np.isfinite(v)]
        avg_miou = float(np.mean(valid_mious)) if valid_mious else np.nan
        avg_occ_iou = float(np.mean(valid_occ_ious)) if valid_occ_ious else np.nan

        table.add_row(
            ['mIoU(%)'] + [self._pct(v) for v in miou_values] +
            [self._pct(avg_miou)])
        table.add_row(
            ['IoU(%)'] + [self._pct(v) for v in occ_iou_values] +
            [self._pct(avg_occ_iou)])

        metric_dict = {}
        for frame_idx, value in enumerate(miou_values):
            metric_dict[f'step_{frame_idx}_mIoU'] = value
        for frame_idx, value in enumerate(occ_iou_values):
            metric_dict[f'step_{frame_idx}_IoU'] = value
        metric_dict['avg_mIoU'] = avg_miou
        metric_dict['avg_IoU'] = avg_occ_iou
        return table, metric_dict

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate current/future occupancy IoU and mIoU.

        #* 对齐 nuScenesWorldDatasetTemplateOffline.evaluate() 的评估入口。
        Drive_OccWorld.forward_test() 已经在模型侧把预测 occupancy 与 GT
        occupancy 转成 confusion matrix：
          - hist_for_iou: 当前帧 + 未来帧整体；
          - hist_for_iou_current: 当前参考帧；
          - hist_for_iou_future: 未来帧；
          - hist_for_iou_future_time_weighting: 未来帧按时间衰减加权。

        这里负责跨样本累加 confusion matrix，并输出 SemanticKITTI 20 类
        per-class IoU 和不含 empty 类的 mIoU。

        #* 整体流程：
        #* 1. _collect_result_values(): 从 EvalHook 收集每个样本的 hist；
        #* 2. _sum_histograms(): 跨样本累加 confusion matrix；
        #* 3. _hist_to_ious(): 根据累计 confusion matrix 计算每一类 IoU；
        #* 4. _format_iou_table(): 打印每类 IoU，并计算/记录 mIoU。
        #* 5. _semantic_hist_to_binary_hist(): 把所有非 empty 类合并成
        #*    occupied，额外评估 empty / occupied 二值占据 IoU。
        """
        eval_results = {}

        #* ================== 精简 forecast 表格 ==================
        # 如果模型返回逐时间步 hist，则优先打印类似论文中的 compact table：
        #   0/current, 1-step, 2-step, ... Avg.
        # 这样日志不会被每类 IoU 长表刷屏，更适合训练中快速观察。
        per_frame_hists = self._collect_per_frame_histograms(results)
        if per_frame_hists:
            compact_table, compact_metrics = self._format_compact_forecast_table(
                per_frame_hists)
            eval_results.update(compact_metrics)
            title = ('SemanticKITTI 当前帧占据评估 (current):' if len(per_frame_hists) == 1 else
                     'SemanticKITTI compact forecasting metrics '
                     '(0=current, 1..N=future, Avg.=current+future average):')
            if logger is not None:
                logger.info(title)
                logger.info('\n' + compact_table.get_string())
            else:
                print('\n' + title)
                print(compact_table)

        eval_items = [
            ('hist_for_iou', 'current_future', '当前帧 + 未来帧'),
            ('hist_for_iou_current', 'current', '当前帧'),
            ('hist_for_iou_future', 'future', '未来帧'),
            ('hist_for_iou_future_time_weighting',
             'future_time_weighting', '未来帧 time-weighting'),
        ]

        for result_key, metric_prefix, readable_name in eval_items:
            hist_values = self._collect_result_values(results, result_key)
            hist_sum = self._sum_histograms(hist_values)
            if hist_sum is None:
                continue

            ious = self._hist_to_ious(hist_sum)
            table, metric_dict = self._format_iou_table(ious, metric_prefix)

            #* 额外计算二值占据 IoU：empty vs occupied。
            # 这回答的是“哪里被占据”是否预测正确，不关心 occupied
            # 体素具体属于 car/road/building 等哪个语义类别。
            binary_hist = self._semantic_hist_to_binary_hist(hist_sum)
            binary_ious = self._hist_to_ious(binary_hist)
            binary_table, binary_metric_dict = self._format_binary_iou_table(
                binary_ious, metric_prefix)
            #! 日志精简：
            #! metric_dict / binary_metric_dict 包含大量逐类 IoU，例如
            #! current_future/IoU_car、future/IoU_road 等。TextLoggerHook 会把
            #! eval_results 里的所有 key 全部展开打印，导致训练日志非常长。
            #! 因此默认不再把这些详细逐类指标写入 eval_results，只保留
            #! 上方 compact table 对应的 step_* / avg_* 精简指标。
            #!
            #! 如果后续需要 TensorBoard 记录逐类 IoU，可以临时打开：
            #! eval_results.update(metric_dict)
            #! eval_results.update(binary_metric_dict)

            if logger is not None:
                #* 为避免训练中日志过长，默认不打印每个类别的长表。
                # 如需逐类排查，metric_dict 中仍保留了每类 IoU，
                # 可在调试时临时打开下面两组 logger.info。
                # logger.info(
                #     f'SemanticKITTI occupancy IoU evaluation: {readable_name}')
                # logger.info('\n' + table.get_string())
                # logger.info(
                #     f'SemanticKITTI binary occupancy IoU evaluation: {readable_name}')
                # logger.info('\n' + binary_table.get_string())
                pass
            else:
                # 控制台模式同样默认只输出 compact table。
                pass

        #! 快速调试 workaround：
        #! 单卡 launcher=none + IterBasedRunner 下，MMDet 默认 EvalHook
        #! 在 after_train_iter 触发评估后，会继续交给 TextLoggerHook 打印日志。
        #! TextLoggerHook 的 iter 日志模板默认读取 time / data_time 字段；
        #! 但 Dataset.evaluate() 返回的是评估指标，不包含这两个训练耗时字段，
        #! 因而会触发 KeyError: 'data_time'。
        #!
        #! 这里补 0.0 只是为了让快速评估调试链路跑通，不表示真实评估耗时。
        #! 更干净的长期方案是实现单卡 CustomEvalHook 或改 logger hook。
        eval_results.setdefault('time', 0.0)
        eval_results.setdefault('data_time', 0.0)
        return eval_results
