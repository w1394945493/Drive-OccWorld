"""SemanticKITTI 点级标签读取与相机图像稀疏监督投影。"""
from pathlib import Path
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES

#* （FoundationSSC 辅助深度&语义损失) 对齐原 FoundationSSC learning_map['kitti']。
# 原始语义 255 是 moving-motorcyclist，映射为 8，不能当作 occupancy ignore=255。
# 映射后的 0 为点级 unlabeled；此处不擅自改成 255，也不把它视为真实空体素。
LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
    51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
    99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}


@PIPELINES.register_module()
class LoadSemanticKITTIPointsAndLabels:
    def __init__(self, pts_label_root=None):
        self.pts_label_root = Path(pts_label_root) if pts_label_root else None

    def __call__(self, results):
        #* （FoundationSSC 辅助深度&语义损失) 优先显式根目录，其次 PKL；
        # 旧 PKL 仅缺标签字段时，从 lidar_path 推导标准同序列 labels 路径，不猜测其他布局。
        frame = results['current_input']
        if not frame.get('lidar_path'):
            raise ValueError('当前帧没有 lidar_path，请更新 PKL')
        lidar = Path(frame['lidar_path'])
        sequence = lidar.parent.parent.name
        label = (self.pts_label_root / sequence / 'labels' / (lidar.stem + '.label')
                 if self.pts_label_root else Path(frame['pts_label_path'])
                 if frame.get('pts_label_path') else lidar.parent.parent / 'labels' / (lidar.stem + '.label'))
        for path in (lidar, label):
            if not path.is_file():
                raise FileNotFoundError(f'辅助监督文件不存在：{path}；请检查 pts_label_root/PKL')
        if lidar.stem != label.stem:
            raise ValueError(f'点云与标签帧号不一致：{lidar} / {label}')
        if lidar.stat().st_size % 16 or label.stat().st_size % 4:
            raise ValueError('点云字节数必须整除16，uint32标签字节数必须整除4')
        points = np.fromfile(lidar, dtype='<f4').reshape(-1, 4)
        packed = np.fromfile(label, dtype='<u4')
        if len(points) != len(packed) or len(points) == 0:
            raise ValueError(f'点数/标签数异常：{len(points)} / {len(packed)}')
        if not np.isfinite(points).all():
            raise ValueError(f'点云包含 NaN/Inf：{lidar}，不单独过滤而破坏标签对应关系')
        #* （FoundationSSC 辅助深度&语义损失) 低16位语义、高16位实例；不重排点序。
        raw = (packed & 0xFFFF).astype(np.int64)
        instances = (packed >> 16).astype(np.int64)
        unknown = set(np.unique(raw).tolist()) - LEARNING_MAP.keys()
        if unknown:
            raise ValueError(f'未定义的原始语义ID：{sorted(unknown)}')
        lut = np.zeros(65536, dtype=np.int64)
        for key, value in LEARNING_MAP.items():
            lut[key] = value
        results['aux_lidar'] = dict(
            points=torch.from_numpy(points), semantic_raw=torch.from_numpy(raw),
            semantic_labels=torch.from_numpy(lut[raw]), instance_ids=torch.from_numpy(instances),
            lidar_path=str(lidar), pts_label_path=str(label))
        return results


@PIPELINES.register_module()
class ProjectFoundationSSCLidar:
    """使用当前 img_inputs 的标定/post transform，默认生成左图标签。"""
    def __init__(self, camera_indices=(0,)):
        self.camera_indices = tuple(camera_indices)
        if not self.camera_indices or len(set(self.camera_indices)) != len(self.camera_indices) or any(i not in (0, 1) for i in self.camera_indices):
            raise ValueError('camera_indices 应为 (0,)、(1,) 或 (0,1)')

    def __call__(self, results):
        #* （FoundationSSC 辅助深度&语义损失) 在加载图像/点云后、打包前投影。
        # 当前点云仍在原 LiDAR 坐标系；不对它重复应用 voxel/BDA 变换。
        aux = results['aux_lidar']
        xyz = aux['points'][:, :3].numpy().astype(np.float64)
        semantics = aux['semantic_labels'].numpy()
        images, rots, trans, intrinsics, post_rots, post_trans = results['img_inputs'][:6]
        height, width = images.shape[-2:]
        depths, targets = [], []
        for cam in self.camera_indices:
            rotation, translation = rots[cam].numpy(), trans[cam].numpy()
            #* （FoundationSSC 辅助深度&语义损失) camera→LiDAR 的逆变换，
            # 再乘原图内参；depth 保存相机 Z（米），不是欧氏距离。
            camera_xyz = np.linalg.solve(rotation, (xyz - translation).T).T
            depth = camera_xyz[:, 2]
            k = intrinsics[cam].numpy()
            projected = camera_xyz @ k[:3, :3].T
            if k.shape == (4, 4):
                projected += k[:3, 3]  # 当前 PKL 为零，不重复外参平移。
            valid = np.isfinite(projected).all(1) & np.isfinite(depth) & (depth > 0) & (projected[:, 2] > 0)
            source = np.flatnonzero(valid)
            uv1 = projected[valid] / projected[valid, 2:3]
            #* （FoundationSSC 辅助深度&语义损失) 对原图像素同步 resize/crop；
            # 不缩放深度数值、不对离散语义图插值。
            augmented = uv1 @ post_rots[cam].numpy().T + post_trans[cam].numpy()
            u, v = augmented[:, 0], augmented[:, 1]
            inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
            source = source[inside]
            x, y = np.rint(u[inside]).astype(np.int64), np.rint(v[inside]).astype(np.int64)
            pixel = y * width + x
            #* （FoundationSSC 辅助深度&语义损失) 按像素、深度、原始点序排序，
            # 每个像素只写最近点；避免重复索引赋值的覆盖顺序不确定。
            order = np.lexsort((source, depth[source], pixel))
            pixel, source = pixel[order], source[order]
            keep = np.r_[True, pixel[1:] != pixel[:-1]] if len(pixel) else np.zeros(0, dtype=bool)
            pixel, source = pixel[keep], source[keep]
            depth_map = np.zeros(height * width, dtype=np.float32)
            semantic_map = np.full(height * width, 255, dtype=np.int64)
            depth_map[pixel] = depth[source]
            semantic_map[pixel] = semantics[source]
            depths.append(torch.from_numpy(depth_map.reshape(height, width)))
            targets.append(torch.from_numpy(semantic_map.reshape(height, width)))
        #* （FoundationSSC 辅助深度&语义损失) 未命中像素 depth=0/semantic=255；
        # 命中的点级 unlabeled=0 保留原映射。右图仅用于几何检查，损失后续取左图。
        results['foundation_meta']['gt_depths'] = torch.stack(depths)
        results['foundation_meta']['projection_camera_indices'] = torch.tensor(self.camera_indices)
        results['gt_semantics'] = torch.stack(targets)
        return results
