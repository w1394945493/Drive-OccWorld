#!/usr/bin/env python3
"""Build a small Drive-OccWorld-light style PKL for SemanticKITTI.

第一阶段目标：
    基于 SemanticKITTI 构建「单目图像 -> BEV -> future occupancy forecasting」
    的轻量数据入口。该脚本只负责生成一个 sequence 中 N 帧的小样本 PKL，
    用来验证数据字段、相机标定、ego pose、occupancy 路径是否可用。

与 nuScenes Drive-OccWorld 原始数据的区别：
    - 只使用 SemanticKITTI 单目相机 image_2；
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
    - cams["CAM_FRONT"]：单目图像路径、相机内参、lidar2img 等；
    - occ_path / voxel_path：FoundationSSC dense occupancy 标签路径；
    - ego2global / lidar2ego：自车位姿和简化外参；
    - prev / next：同一 sequence 内前后帧 token；
    - gt_ego_his_trajs / gt_ego_fut_trajs：由 pose 差分得到的自车运动标签，
      先作为调试和后续扩展字段保留，第一阶段模型可不使用。
"""

import argparse
import os
import os.path as osp
import pickle
import sys
import time

import numpy as np


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
    parser.add_argument('--his-ts', type=int, default=2,
                        help='history steps kept in metadata/debug fields')
    parser.add_argument('--fut-ts', type=int, default=4,
                        help='future steps kept in metadata/debug fields')
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


def list_frame_tokens(sequence_dir):
    """List available dense-occupancy frame tokens by scanning voxels/*.bin."""
    voxel_dir = osp.join(sequence_dir, 'voxels')
    if not osp.isdir(voxel_dir):
        raise FileNotFoundError(f'Missing voxels directory: {voxel_dir}')
    tokens = sorted(
        osp.splitext(entry.name)[0] for entry in os.scandir(voxel_dir)
        if entry.is_file() and entry.name.endswith('.bin'))
    if not tokens:
        raise RuntimeError(f'No .bin files found in {voxel_dir}')
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


def build_history_trajs(poses_lidar, token_pose_indices, frame_idx, his_ts):
    """Build his_ts adjacent historical ego displacements in current frame."""
    token_indices = [max(0, frame_idx - i) for i in range(his_ts, -1, -1)]
    pose_indices = [token_pose_indices[i] for i in token_indices]
    current_pose_idx = token_pose_indices[frame_idx]
    positions = relative_positions_in_current(
        poses_lidar, current_pose_idx, pose_indices)
    return (positions[1:] - positions[:-1])[:, :2].astype(np.float32)


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


def build_cam_front_info(sequence_dir, token, calib):
    """Build monocular CAM_FRONT meta from SemanticKITTI image_2 and calib."""
    image_path = osp.join(sequence_dir, 'image_2', f'{token}.png')

    # P2 is a 3x4 projection matrix from cam0 rect coordinates to image_2.
    # For first-stage BEVFormer-style projection, we store:
    #   lidar2img = P2 @ Tr_lidar_to_cam0.
    # P2 in calib has been converted to 4x4 with the original 3x4 values
    # in the first three rows, so P2[:3, :4] keeps the actual projection.
    p2 = calib['P2'][:3, :4].astype(np.float64)
    lidar_to_cam0 = calib['Tr'].astype(np.float64)
    lidar2img = p2 @ lidar_to_cam0

    cam_intrinsic = np.eye(4, dtype=np.float64)
    cam_intrinsic[:3, :3] = p2[:3, :3]

    # sensor2lidar means camera -> LiDAR.  It is useful for later dataset
    # compatibility even if the first-stage loader only consumes lidar2img.
    sensor2lidar = np.linalg.inv(lidar_to_cam0)

    return {
        'data_path': image_path,
        'type': 'CAM_FRONT',
        'cam_intrinsic': cam_intrinsic.tolist(),
        'lidar2img': lidar2img.tolist(),
        'sensor2lidar_rotation': sensor2lidar[:3, :3].tolist(),
        'sensor2lidar_translation': sensor2lidar[:3, 3].tolist(),
        # Keep these aliases for easier adaptation to nuScenes-style loaders.
        'sensor2ego_rotation': sensor2lidar[:3, :3].tolist(),
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
        his_ts,
        fut_ts,
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

    prev_token = tokens[frame_idx - 1] if frame_idx > 0 else ''
    next_token = tokens[frame_idx + 1] if frame_idx + 1 < len(tokens) else ''
    scene_name = f'sequence-{sequence}'
    pose = poses_lidar[pose_idx]
    cam_front = build_cam_front_info(sequence_dir, token, calib)

    occ_path = osp.join(ann_file, sequence, f'{token}_1_1.npy')
    voxel_path = occ_path

    if check_files:
        if not osp.isfile(cam_front['data_path']):
            raise FileNotFoundError(f'Missing image file: {cam_front["data_path"]}')
        if not osp.isfile(occ_path):
            raise FileNotFoundError(f'Missing occupancy file: {occ_path}')

    gt_ego_his_trajs = build_history_trajs(
        poses_lidar, token_pose_indices, frame_idx, his_ts)
    gt_ego_fut_trajs, gt_ego_fut_masks = build_future_trajs(
        poses_lidar, token_pose_indices, frame_idx, fut_ts)
    command = build_pseudo_command(gt_ego_fut_trajs, cmd_thresh)
    command_onehot = np.zeros(3, dtype=np.float32)
    command_onehot[command] = 1.0

    # 第一阶段不使用 action condition，但保留一个轻量 can_bus 占位：
    # 后续若复用 BEVFormer 旧代码中读取 can_bus 的接口，可以先避免 KeyError。
    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[:3] = pose[:3, 3].astype(np.float32)

    return {
        #* ================== 1. 基础时序字段 ==================
        'token': token,
        'sample_idx': token,
        'scene_token': scene_name,
        'scene_name': scene_name,
        'location': 'semantickitti',
        'frame_idx': int(frame_idx),
        'pose_idx': int(pose_idx),
        'timestamp': int(pose_idx),
        'prev': prev_token,
        'next': next_token,

        #* ================== 2. 单目图像与相机标定 ==================
        'cams': {'CAM_FRONT': cam_front},
        'img_filename': [cam_front['data_path']],
        'lidar2img': [cam_front['lidar2img']],

        #* ================== 3. occupancy 标签路径 ==================
        'occ_path': occ_path,
        'voxel_path': voxel_path,
        'lidar_path': osp.join(sequence_dir, 'velodyne', f'{token}.bin'),

        #* ================== 4. ego/LiDAR 位姿 ==================
        # 第一阶段把 LiDAR 坐标系直接作为 ego 坐标系。
        'lidar2ego_translation': [0.0, 0.0, 0.0],
        'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],
        'ego2global_translation': pose[:3, 3].astype(np.float64).tolist(),
        'ego2global_rotation': rotation_matrix_to_quaternion_wxyz(pose[:3, :3]),
        'can_bus': can_bus,

        #* ================== 5. 轨迹/指令调试字段，第一阶段可不使用 ==================
        'gt_ego_his_trajs': gt_ego_his_trajs,
        'gt_ego_fut_trajs': gt_ego_fut_trajs,
        'gt_ego_fut_masks': gt_ego_fut_masks,
        'gt_ego_fut_cmd': command_onehot,
        'command': int(command),
        'fut_valid_flag': bool(frame_idx + fut_ts < len(tokens)),

        #* ================== 6. 占位字段：后续 dataset 可按需忽略 ==================
        'sweeps': [],
        'ann_infos': [],
        'gt_boxes': np.zeros((0, 7), dtype=np.float32),
        'gt_names': np.asarray([], dtype=object),
    }


def print_summary(infos):
    """Print compact summary for quick sanity check."""
    print(f'Built {len(infos)} frame infos.')
    if not infos:
        return
    info = infos[0]
    print('First frame summary:')
    print(f"  token: {info['token']}")
    print(f"  image: {info['cams']['CAM_FRONT']['data_path']}")
    print(f"  occ_path: {info['occ_path']}")
    print(f"  lidar2img shape: {np.asarray(info['cams']['CAM_FRONT']['lidar2img']).shape}")
    print(f"  cam_intrinsic shape: {np.asarray(info['cams']['CAM_FRONT']['cam_intrinsic']).shape}")
    print(f"  ego2global_translation: {info['ego2global_translation']}")
    print(f"  gt_ego_fut_trajs shape: {info['gt_ego_fut_trajs'].shape}")
    print(f"  command: {info['command']}  # 0=Right, 1=Left, 2=Forward")


def main():
    args = parse_args()

    data_root = osp.abspath(args.data_root)
    sequence = args.sequence
    sequence_dir = osp.join(data_root, 'sequences', sequence)
    ann_file = osp.join(data_root, 'labels')
    if not osp.isdir(sequence_dir):
        raise FileNotFoundError(f'Missing sequence directory: {sequence_dir}')
    if not osp.isdir(ann_file):
        raise FileNotFoundError(f'Missing occupancy label root: {ann_file}')

    calib = read_calib(osp.join(sequence_dir, 'calib.txt'))
    tokens = list_frame_tokens(sequence_dir)
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
    t_start = time.time()
    for offset, frame_idx in enumerate(range(start, end), 1):
        infos.append(build_frame_info_from_cache(
            data_root=data_root,
            ann_file=ann_file,
            sequence=sequence,
            tokens=tokens,
            token_pose_indices=token_pose_indices,
            poses_lidar=poses_lidar,
            calib=calib,
            frame_idx=frame_idx,
            his_ts=args.his_ts,
            fut_ts=args.fut_ts,
            cmd_thresh=args.cmd_thresh,
            check_files=args.check_files))
        if offset == 1 or offset == args.num_frames or offset % 50 == 0:
            elapsed = time.time() - t_start
            fps = offset / max(elapsed, 1e-6)
            print(f'\rBuilding infos: {offset}/{args.num_frames} '
                  f'({fps:.1f} frame/s)', end='', file=sys.stderr, flush=True)
    print('', file=sys.stderr)

    data = {
        'infos': infos,
        'metadata': {
            'dataset': 'SemanticKITTI',
            'style': 'Drive-OccWorld-light',
            'version': 'small_sample_v1',
            'data_root': data_root,
            'ann_file': ann_file,
            'sequence': sequence,
            'start_frame_idx': int(start),
            'num_frames': int(len(infos)),
            'camera': 'image_2/CAM_FRONT',
            'note': (
                'First-stage PKL for monocular image-to-BEV future occupancy '
                'forecasting. No action condition or planning fields are required.'
            ),
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
