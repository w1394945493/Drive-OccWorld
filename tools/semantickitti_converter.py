#!/usr/bin/env python3
"""Build a small Drive-OccWorld-light style PKL for SemanticKITTI.

第一阶段目标：
    基于 SemanticKITTI 构建「单目图像 -> BEV -> future occupancy forecasting」
    的轻量数据入口。该脚本只负责生成一个 sequence 中 N 帧的小样本 PKL，
    用来验证数据字段、相机标定、ego pose、occupancy 路径是否可用。

与 nuScenes Drive-OccWorld 原始数据的区别：
    - 使用 SemanticKITTI 前视双目相机 image_2 / image_3；
    - 不生成 CAN bus / action condition / planning / sample_traj；
    - 不依赖 nuScenes SDK；
    - 只保留 future occupancy forecasting 第一阶段必需字段。

输出 PKL 结构：
    {
        "infos": [frame_info_0, frame_info_1, ...],
        "metadata": {...}
    }

每个 frame_info 主要包含：
    - token / scene_token / frame_idx / timestamp；
    - cams["CAM_FRONT_LEFT"] / cams["CAM_FRONT_RIGHT"]：前视左右目图像路径、
      相机内参、lidar2img 等；
    - occ_path：FoundationSSC dense occupancy 标签路径；
    - ego2global / lidar2ego：自车位姿和简化外参；
    - prev / next：同一 sequence 内前后帧 token；
    - gt_ego_fut_trajs / command：和 Drive-OccWorld nuScenes v2 pkl
      形状对齐的轻量未来轨迹/指令字段，第一阶段模型可不使用；
    - fut_valid_flag：轻量有效性标记，便于后续 Dataset 过滤尾部未来帧不足样本。
"""

import argparse
import os
import os.path as osp
import pickle

import numpy as np
from tqdm import tqdm


DEFAULT_DATA_ROOT = '/c20250502/wangyushen/Datasets/kitti/semantickitti/dataset'


def resolve_ann_file(data_root):
    """由数据集根目录确定 FoundationSSC dense occupancy 根目录。

    当前第一阶段沿用 OccWorld / FoundationSSC 风格标签：
        <data_root>/labels/<sequence>/<frame_id>_1_1.npy

    这里不读取 SemanticKITTI 原始逐点 labels/*.label。
    """
    ann_file = osp.join(data_root, 'labels')
    if not osp.isdir(ann_file):
        raise FileNotFoundError(
            f'Cannot infer ann_file from data_root. Expected: {ann_file}')
    return ann_file


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build a small SemanticKITTI Drive-OccWorld style PKL.')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                        help='SemanticKITTI root, usually ending with /dataset')
    parser.add_argument('--sequence', default='00',
                        help='SemanticKITTI sequence id, e.g. 00')
    parser.add_argument('--frame-idx', type=int, default=0,
                        help='start index in the available occupancy token list')
    parser.add_argument('--num-frames', type=int, default=8,
                        help='number of consecutive available occupancy frames to dump')
    parser.add_argument('--all-frames', action='store_true',
                        help='dump all available occupancy frames in the sequence')
    parser.add_argument('--cmd-thresh', type=float, default=2.0,
                        help='lateral threshold for pseudo left/right/forward command')
    parser.add_argument('--out-pkl', required=True,
                        help='output PKL path')
    parser.add_argument('--check-files', action='store_true',
                        help='check image / occupancy files exist; may be slower on network FS')
    return parser.parse_args()


def read_calib(calib_path):
    """Read SemanticKITTI calib.txt.

    SemanticKITTI calib usually contains:
        P0, P1, P2, P3: camera projection matrices, shape 3x4;
        Tr: velodyne/LiDAR -> cam0, shape 3x4.

    We convert every 3x4 matrix to a 4x4 homogeneous matrix for convenience.
    """
    if not osp.isfile(calib_path):
        raise FileNotFoundError(f'Missing calib file: {calib_path}')
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            key, value = line.split(':', 1)
            values = np.fromstring(value, sep=' ', dtype=np.float64)
            if values.size == 12:
                mat = np.eye(4, dtype=np.float64)
                mat[:3, :4] = values.reshape(3, 4)
            elif values.size == 16:
                mat = values.reshape(4, 4)
            else:
                mat = values
            calib[key] = mat
    if 'Tr' not in calib:
        raise KeyError(f'{calib_path} does not contain Tr.')
    if 'P2' not in calib:
        raise KeyError(f'{calib_path} does not contain P2 for image_2.')
    if 'P3' not in calib:
        raise KeyError(f'{calib_path} does not contain P3 for image_3.')
    return calib


def read_poses(poses_path):
    """Read poses.txt as a list of 4x4 camera-to-global poses."""
    if not osp.isfile(poses_path):
        raise FileNotFoundError(f'Missing poses file: {poses_path}')
    poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            values = np.fromstring(line, sep=' ', dtype=np.float64)
            if values.size != 12:
                raise ValueError(f'Invalid pose line in {poses_path}: {line}')
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :4] = values.reshape(3, 4)
            poses.append(mat)
    return poses


def rotation_matrix_to_quaternion_wxyz(rot):
    """Convert a 3x3 rotation matrix to a wxyz quaternion without scipy."""
    rot = np.asarray(rot, dtype=np.float64)
    trace = np.trace(rot)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(rot)))
        if axis == 0:
            s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif axis == 1:
            s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-12
    return quat.tolist()


def list_frame_tokens(ann_file, sequence):
    """List available dense-occupancy frame tokens from labels/<seq>/*_1_1.npy."""
    occ_label_dir = osp.join(ann_file, sequence)
    if not osp.isdir(occ_label_dir):
        raise FileNotFoundError(
            f'Missing occupancy label directory: {occ_label_dir}')

    #* ================== 构建当前 sequence 的可用帧 token 列表 ==================
    # 第一阶段训练监督真正读取的是 dense occupancy 标签：
    #   labels/<sequence>/<token>_1_1.npy
    # 因此这里直接扫描 labels/<sequence>/*_1_1.npy 来构建 tokens。
    #
    # 这样可以保证：
    #   当前 token 以及基于 tokens 构建出的 prev/next token，
    #   都对应一个真实存在的 occ_path。
    #
    # 注意：这里不再扫描 sequences/<sequence>/voxels/*.bin。
    # 原因是 voxels/*.bin 与 labels/*_1_1.npy 通常同名但不是同一个监督文件；
    # 如果只用 voxels 决定 token，可能出现有 voxel bin 但没有 dense occ npy 的样本。
    #
    # sorted(...) 会按字符串顺序排序。由于 SemanticKITTI 帧号是 6 位补零格式，
    # 字符串顺序等价于时间顺序。
    # 因此后面 prev/next 的“相邻帧”含义就是：
    #   在排序后的“有 occupancy 标注”的 token 列表中索引相邻。
    # 它不是额外根据 timestamp / pose / image 最近邻搜索出来的。
    tokens = sorted(
        entry.name[:-len('_1_1.npy')]
        for entry in os.scandir(occ_label_dir)
        if entry.is_file() and entry.name.endswith('_1_1.npy'))
    if not tokens:
        raise RuntimeError(f'No *_1_1.npy files found in {occ_label_dir}')
    return tokens


def token_to_pose_index(token):
    """SemanticKITTI token name is the original pose row id."""
    return int(token)


def load_lidar_poses(data_root, sequence, calib):
    """Build LiDAR/ego -> global poses.

    SemanticKITTI poses.txt is usually cam0 -> global, and Tr is LiDAR -> cam0.
    Therefore:
        T_global_lidar = T_global_cam0 @ T_cam0_lidar

    For this light stage, LiDAR frame is treated as ego frame.
    """
    sequence_dir = osp.join(data_root, 'sequences', sequence)
    poses_cam0 = read_poses(osp.join(sequence_dir, 'poses.txt'))
    lidar_to_cam0 = calib['Tr']
    return [pose_cam0 @ lidar_to_cam0 for pose_cam0 in poses_cam0]


def relative_positions_in_current(poses_lidar, current_pose_idx, pose_indices):
    """Transform LiDAR origins of selected frames into current LiDAR frame."""
    global_to_current = np.linalg.inv(poses_lidar[current_pose_idx])
    origin = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    points = []
    for idx in pose_indices:
        point = global_to_current @ poses_lidar[idx] @ origin
        points.append(point[:3])
    return np.asarray(points, dtype=np.float64)


def build_future_trajs(poses_lidar, token_pose_indices, frame_idx, fut_ts):
    """Build fut_ts adjacent future ego displacements and valid mask."""
    max_token_idx = len(token_pose_indices) - 1
    token_indices = [min(max_token_idx, frame_idx + i)
                     for i in range(fut_ts + 1)]
    pose_indices = [token_pose_indices[i] for i in token_indices]
    masks = np.asarray(
        [1.0 if frame_idx + i <= max_token_idx else 0.0
         for i in range(1, fut_ts + 1)],
        dtype=np.float32)
    current_pose_idx = token_pose_indices[frame_idx]
    positions = relative_positions_in_current(
        poses_lidar, current_pose_idx, pose_indices)
    fut = (positions[1:] - positions[:-1])[:, :2].astype(np.float32)
    fut[masks == 0] = 0.0
    return fut, masks


def build_pseudo_command(fut_trajs, cmd_thresh):
    """Build integer pseudo command compatible with Drive-OccWorld convention.

    Drive-OccWorld planner uses:
        0: Right, 1: Left, 2: Forward.
    """
    final_xy = np.cumsum(fut_trajs[:, :2], axis=0)[-1]
    lateral = final_xy[1]
    if lateral <= -cmd_thresh:
        return 0
    if lateral >= cmd_thresh:
        return 1
    return 2


def build_pseudo_command_sequence(fut_trajs, cmd_thresh, command_steps=5):
    """Build Drive-OccWorld-style command sequence, shape (command_steps,).

    nuScenes v2 pkl 中 command 通常是 ``ndarray shape=(5,)``，用于多步
    planning/action condition。SemanticKITTI 第一阶段暂不使用 action condition，
    但这里仍按相同形式保存，便于后续 Dataset 适配。

    简化策略：
        根据当前帧未来累计轨迹得到一个 left/right/forward 伪指令，
        然后复制到 command_steps 个未来 step。
    """
    command = build_pseudo_command(fut_trajs, cmd_thresh)
    return np.full((command_steps,), command, dtype=np.int64)


def build_cam_info(sequence_dir, token, calib, image_dir, proj_key, cam_type):
    """Build nuScenes-style camera meta from one SemanticKITTI front camera.

    SemanticKITTI / KITTI 常用：
        - image_2 + P2: 左目彩色前视相机；
        - image_3 + P3: 右目彩色前视相机。

    这里显式命名为 CAM_FRONT_LEFT / CAM_FRONT_RIGHT，避免把 KITTI 的左目
    图像误认为 nuScenes 中几何意义上的居中 CAM_FRONT。
    """
    image_path = osp.join(sequence_dir, image_dir, f'{token}.png')

    # P2/P3 是从 rect cam0 坐标到对应图像平面的 3x4 投影矩阵。
    # 对 BEVFormer-style 图像投影，直接保存：
    #   lidar2img = P{2/3} @ Tr_lidar_to_cam0。
    # calib 中 P2/P3 已被扩展成 4x4，因此这里取前三行四列保留原始投影。
    proj = calib[proj_key][:3, :4].astype(np.float64)
    lidar_to_cam0 = calib['Tr'].astype(np.float64)
    lidar2img = proj @ lidar_to_cam0

    # 和 nuScenes pkl 对齐：cam_intrinsic 使用 3x3，而不是 4x4。
    cam_intrinsic = proj[:3, :3].astype(np.float64)

    # P2/P3 的第 4 列编码了相机相对 rect cam0 的平移。为了让
    # sensor2lidar_translation 对左右相机有所区分，这里近似恢复
    # cam0 -> 当前 camera 的平移，并组合到 LiDAR->camera 外参中。
    # 第一阶段主要使用 lidar2img；该外参字段更多是为了后续 Dataset 兼容。
    cam0_to_cam = np.eye(4, dtype=np.float64)
    cam0_to_cam[:3, 3] = np.linalg.inv(cam_intrinsic) @ proj[:3, 3]
    lidar_to_cam = cam0_to_cam @ lidar_to_cam0

    # sensor2lidar means camera -> LiDAR.
    sensor2lidar = np.linalg.inv(lidar_to_cam)
    sensor2lidar_quat = rotation_matrix_to_quaternion_wxyz(sensor2lidar[:3, :3])

    return {
        'data_path': image_path,
        'type': cam_type,
        'cam_intrinsic': cam_intrinsic.tolist(),
        'lidar2img': lidar2img.tolist(),
        'sensor2lidar_rotation': sensor2lidar[:3, :3].tolist(),
        'sensor2lidar_translation': sensor2lidar[:3, 3].tolist(),
        # Keep these aliases for easier adaptation to nuScenes-style loaders.
        # 和 nuScenes pkl 对齐：sensor2ego_rotation 使用 wxyz 四元数 len=4。
        # 第一阶段中 LiDAR frame 被当作 ego frame，因此 camera->ego 等同 camera->LiDAR。
        'sensor2ego_rotation': sensor2lidar_quat,
        'sensor2ego_translation': sensor2lidar[:3, 3].tolist(),
    }


def build_frame_info_from_cache(
        data_root,
        ann_file,
        sequence,
        tokens,
        token_pose_indices,
        poses_lidar,
        calib,
        frame_idx,
        cmd_thresh,
        check_files=False):
    """Build one Drive-OccWorld-light SemanticKITTI frame info."""
    sequence_dir = osp.join(data_root, 'sequences', sequence)
    token = tokens[frame_idx]
    pose_idx = token_pose_indices[frame_idx]
    if pose_idx >= len(poses_lidar):
        raise IndexError(
            f'token {token} maps to pose index {pose_idx}, '
            f'but poses.txt has only {len(poses_lidar)} lines.')

    #* prev/next 只表示排序后 tokens 列表里的前后相邻帧。
    # 例如 tokens = ['000000', '000001', '000002'] 时：
    #   frame_idx=1 -> prev='000000', next='000002'。
    # 边界帧没有前/后相邻帧时，用空字符串 '' 占位，和 nuScenes pkl 风格接近。
    prev_token = tokens[frame_idx - 1] if frame_idx > 0 else ''
    next_token = tokens[frame_idx + 1] if frame_idx + 1 < len(tokens) else ''
    scene_name = f'sequence-{sequence}'
    pose = poses_lidar[pose_idx]
    cam_front_left = build_cam_info(
        sequence_dir, token, calib,
        image_dir='image_2',
        proj_key='P2',
        cam_type='CAM_FRONT_LEFT')
    cam_front_right = build_cam_info(
        sequence_dir, token, calib,
        image_dir='image_3',
        proj_key='P3',
        cam_type='CAM_FRONT_RIGHT')
    cams = {
        'CAM_FRONT_LEFT': cam_front_left,
        'CAM_FRONT_RIGHT': cam_front_right,
    }

    occ_path = osp.join(ann_file, sequence, f'{token}_1_1.npy')

    if check_files:
        for cam_name, cam_info in cams.items():
            if not osp.isfile(cam_info['data_path']):
                raise FileNotFoundError(
                    f'Missing {cam_name} image file: {cam_info["data_path"]}')
        if not osp.isfile(occ_path):
            raise FileNotFoundError(f'Missing occupancy file: {occ_path}')

    #* ================== 5. Drive-OccWorld 第一阶段保留的未来轨迹/指令字段 ==================
    # 这里只保留 Drive-OccWorld nuScenes v2 pkl 中也存在、且后续适配可能用到的字段：
    # - gt_ego_fut_trajs: shape=(6, 2)，由 SemanticKITTI pose 差分得到的未来自车位移；
    # - command: shape=(5,)，由未来累计位移粗略离散出的伪驾驶指令序列。
    #
    # 注意：
    # - 这些不是第一阶段 image -> BEV -> future occupancy forecasting 的必要输入；
    # - 但保留它们可以降低后续扩展 action condition / planning 时的 Dataset 适配成本；
    # - 原先偏 OccWorld 风格的 gt_ego_his_trajs / gt_ego_fut_masks /
    #   gt_ego_fut_cmd 已删除，避免把 OccWorld 数据协议混进 Drive-OccWorld 第一阶段。
    #
    # nuScenes Drive-OccWorld v2 中常见设定是：
    # - gt_ego_fut_trajs 有 6 个未来轨迹 step；
    # - command 有 5 个未来动作条件 step。
    # 第一阶段关闭 action condition 和 planning，这些字段暂不作为模型输入，
    # 但保持形状一致可降低后续 Dataset 适配成本。
    gt_ego_fut_trajs, _ = build_future_trajs(
        poses_lidar, token_pose_indices, frame_idx, fut_ts=6)
    command = build_pseudo_command_sequence(
        gt_ego_fut_trajs, cmd_thresh, command_steps=5)

    # 第一阶段不使用 action condition，但保留一个轻量 can_bus 占位：
    # 后续若复用 BEVFormer 旧代码中读取 can_bus 的接口，可以先避免 KeyError。
    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[:3] = pose[:3, 3].astype(np.float32)

    return {
        #* ================== 1. 基础时序字段 ==================
        'token': token,                         # str，当前帧 id，例如 '000000'
        'scene_token': scene_name,              # str，场景/序列 id，这里为 'sequence-00' 等
        'scene_name': scene_name,               # str，场景/序列名；和 scene_token 保持一致
        'location': 'semantickitti',            # str，占位地图名；nuScenes 中对应 boston/singapore 等
        'frame_idx': int(frame_idx),            # int，当前帧在本次 tokens 列表中的索引
        'timestamp': int(pose_idx),             # int，SemanticKITTI 无真实时间戳，这里用 pose 行号占位
        'prev': prev_token,                     # str，上一帧 token；首帧为空字符串 ''
        'next': next_token,                     # str，下一帧 token；尾帧为空字符串 ''

        #* ================== 2. 前视双目图像与相机标定 ==================
        # cams: dict，包含两个相机：
        #   - CAM_FRONT_LEFT:  SemanticKITTI image_2 / P2；
        #   - CAM_FRONT_RIGHT: SemanticKITTI image_3 / P3。
        # 每个 cam_info 内部主要字段：
        #   data_path: str，图像路径；
        #   type: str，相机名；
        #   cam_intrinsic: list，shape=(3, 3)，相机内参；
        #   lidar2img: list，shape=(3, 4)，LiDAR 点投影到图像平面的矩阵；
        #   sensor2lidar_rotation: list，shape=(3, 3)，camera -> LiDAR 旋转；
        #   sensor2lidar_translation: list，shape=(3,)，camera -> LiDAR 平移；
        #   sensor2ego_rotation: list，len=4，wxyz 四元数；第一阶段 ego 近似等于 LiDAR；
        #   sensor2ego_translation: list，shape=(3,)。
        'cams': cams,
        'img_filename': [cam['data_path'] for cam in cams.values()],  # list[str]，长度=2，左右相机图像路径
        'lidar2img': [cam['lidar2img'] for cam in cams.values()],     # list，长度=2，每个元素 shape=(3, 4)

        #* ================== 3. occupancy 标签路径 ==================
        'occ_path': occ_path,                                         # str，dense occupancy 标签 .npy 路径
        'lidar_path': osp.join(sequence_dir, 'velodyne', f'{token}.bin'),  # str，原始 LiDAR bin 路径

        #* ================== 4. ego/LiDAR 位姿 ==================
        # 第一阶段把 LiDAR 坐标系直接作为 ego 坐标系。
        'lidar2ego_translation': [0.0, 0.0, 0.0],                     # list[float]，shape=(3,)，LiDAR->ego 平移
        'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],                   # list[float]，len=4，LiDAR->ego 单位四元数 wxyz
        'ego2global_translation': pose[:3, 3].astype(np.float64).tolist(),  # list[float]，shape=(3,)，当前帧全局位置
        'ego2global_rotation': rotation_matrix_to_quaternion_wxyz(pose[:3, :3]),  # list[float]，len=4，当前帧全局朝向 wxyz
        'can_bus': can_bus,                                           # np.ndarray，shape=(18,)，占位；前 3 维写入全局平移

        #* ================== 5. Drive-OccWorld 对齐字段，第一阶段可不使用 ==================
        'gt_ego_fut_trajs': gt_ego_fut_trajs,                         # np.ndarray，shape=(6, 2)，未来 6 步自车 xy 位移
        'command': command,                                           # np.ndarray，shape=(5,)，伪驾驶指令；0右/1左/2直行
        'fut_valid_flag': bool(frame_idx + 6 < len(tokens)),          # bool，未来 6 帧是否都在当前 sequence 内

        #* ================== 6. 占位字段：后续 dataset 可按需忽略 ==================
        'sweeps': [],                                                 # list，历史 LiDAR sweep 占位；第一阶段不用
        'gt_boxes': np.zeros((0, 7), dtype=np.float32),               # np.ndarray，shape=(0, 7)，3D box 占位
        'gt_names': np.asarray([], dtype=object),                     # np.ndarray，shape=(0,)，类别名占位
    }


def print_summary(infos):
    """Print compact summary for quick sanity check."""
    print(f'Built {len(infos)} frame infos.')
    if not infos:
        return
    info = infos[0]
    print('First frame summary:')
    print(f"  token: {info['token']}")
    print(f"  cameras: {list(info['cams'].keys())}")
    print(f"  left image: {info['cams']['CAM_FRONT_LEFT']['data_path']}")
    print(f"  right image: {info['cams']['CAM_FRONT_RIGHT']['data_path']}")
    print(f"  occ_path: {info['occ_path']}")
    print(f"  lidar2img shape: {np.asarray(info['lidar2img']).shape}")
    print(f"  left cam_intrinsic shape: "
          f"{np.asarray(info['cams']['CAM_FRONT_LEFT']['cam_intrinsic']).shape}")
    print(f"  ego2global_translation: {info['ego2global_translation']}")
    print(f"  gt_ego_fut_trajs shape: {info['gt_ego_fut_trajs'].shape}")
    print(f"  command shape: {np.asarray(info['command']).shape}, "
          f"value={np.asarray(info['command']).tolist()}  # 0=Right, 1=Left, 2=Forward")


def main():
    args = parse_args()

    data_root = osp.abspath(args.data_root)
    sequence = args.sequence
    sequence_dir = osp.join(data_root, 'sequences', sequence)
    ann_file = resolve_ann_file(data_root)
    if not osp.isdir(sequence_dir):
        raise FileNotFoundError(f'Missing sequence directory: {sequence_dir}')

    calib = read_calib(osp.join(sequence_dir, 'calib.txt'))
    tokens = list_frame_tokens(ann_file, sequence)
    poses_lidar = load_lidar_poses(data_root, sequence, calib)
    token_pose_indices = [token_to_pose_index(token) for token in tokens]

    if args.num_frames <= 0:
        raise ValueError(f'num_frames must be positive, got {args.num_frames}')
    start = args.frame_idx
    end = start + args.num_frames
    if start < 0 or start >= len(tokens):
        raise IndexError(f'frame_idx={start} out of range [0, {len(tokens) - 1}]')
    if end > len(tokens):
        raise IndexError(
            f'Requested [{start}, {end}) exceeds token length {len(tokens)}.')

    print(f'Loaded sequence-{sequence}: {len(tokens)} occupancy tokens, '
          f'{len(poses_lidar)} poses.')
    print(f'Build small window: token-list index [{start}, {end}).')

    infos = []
    for frame_idx in tqdm(
            range(start, end),
            total=end - start,
            desc=f'Build sequence-{sequence} infos'):
        infos.append(build_frame_info_from_cache(
            data_root=data_root,
            ann_file=ann_file,
            sequence=sequence,
            tokens=tokens,
            token_pose_indices=token_pose_indices,
            poses_lidar=poses_lidar,
            calib=calib,
            frame_idx=frame_idx,
            cmd_thresh=args.cmd_thresh,
            check_files=args.check_files))

    data = {
        'infos': infos,
        'metadata': {
            'data_root': data_root,
            'ann_file': ann_file,
            'sequence': sequence,
            'start_frame_idx': int(start),
            'num_frames': int(len(infos)),
        },
    }

    out_pkl = osp.abspath(args.out_pkl)
    os.makedirs(osp.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(data, f)

    print_summary(infos)
    print(f'Wrote PKL to: {out_pkl}')


if __name__ == '__main__':
    main()
