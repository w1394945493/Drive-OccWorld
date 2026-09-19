"""FoundationSSC 第一阶段：仅从 PKL 构造当前帧双目 SSC 输入，无随机增强。"""
import numpy as np
import torch
from PIL import Image
from mmdet.datasets.builder import PIPELINES


CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')


@PIPELINES.register_module()
class LoadFoundationSSCStereo:
    def __init__(self, input_size=(384, 1280)):
        self.input_size = tuple(input_size)

    def __call__(self, results):
        frame = results['current_input']
        cams = [frame['cams'][name] for name in CAMERAS]
        camera_to_lidar, intrinsics, projections = [], [], []
        for cam in cams:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = np.asarray(cam['sensor2lidar_rotation'])
            transform[:3, 3] = np.asarray(cam['sensor2lidar_translation'])
            k = np.asarray(cam['cam_intrinsic'], dtype=np.float64)
            if k.shape != (3, 3):
                raise ValueError('PKL cam_intrinsic 必须为 3x3')
            np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5)
            projection = k @ np.linalg.inv(transform)[:3]
            np.testing.assert_allclose(projection, np.asarray(cam['lidar2img'])[:3], rtol=1e-5, atol=1e-4)
            #* PKL 已把左右相机平移放在各自外参中；4x4 内参第四列必须为 0，
            # 避免重复计入平移。几何与原 P2/P3 约定等价，但不是原始矩阵数值。
            k4 = np.eye(4)
            k4[:3, :3] = k
            intrinsics.append(k4)
            camera_to_lidar.append(transform)
            projections.append(projection)

        left_to_right = np.linalg.inv(camera_to_lidar[0]) @ camera_to_lidar[1]
        #* 右相机中心在左相机坐标系的 x 位移，即整流双目的正基线（米）。
        baseline = left_to_right[0, 3]
        if baseline <= 0 or not np.allclose(left_to_right[:3, :3], np.eye(3), atol=1e-4):
            raise ValueError('左右相机顺序或整流外参异常')
        if not np.allclose(left_to_right[1:3, 3], 0, atol=0.02):
            raise ValueError('相机不满足近似水平整流双目假设')

        raw, images, post_rots, post_trans = [], [], [], []
        source_size = None
        height, width = self.input_size
        for cam in cams:
            with Image.open(cam['data_path']) as image:
                image = image.convert('RGB')
                if source_size is None:
                    source_size = image.size
                if image.size != source_size:
                    raise ValueError('左右图像原始尺寸必须一致')
                #* 与 FoundationSSC eval pipeline 一致：按目标宽度缩放，底部对齐裁剪。
                scale = width / source_size[0]
                resized = (int(source_size[0] * scale), int(source_size[1] * scale))
                crop_x = int(max(0, resized[0] - width) / 2)
                crop_y = resized[1] - height
                image = image.resize(resized).crop((crop_x, crop_y, crop_x + width, crop_y + height))
                canvas = np.array(image, dtype=np.uint8)
            raw.append(torch.from_numpy(canvas.copy()))  # RGB uint8 HWC，供 FoundationStereo。
            tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1).float() / 255.
            images.append((tensor - torch.tensor([.485, .456, .406])[:, None, None]) /
                          torch.tensor([.229, .224, .225])[:, None, None])
            post_rots.append(np.diag([scale, scale, 1.]))
            post_trans.append([-crop_x, -crop_y, 0.])

        c2l = torch.tensor(np.stack(camera_to_lidar), dtype=torch.float32)
        #* 八项顺序对应 FoundationSSC.extract_img_feat()，均采用列向量几何约定。
        results['img_inputs'] = (
            torch.stack(images), c2l[:, :3, :3], c2l[:, :3, 3],
            torch.tensor(np.stack(intrinsics), dtype=torch.float32),
            torch.tensor(np.stack(post_rots), dtype=torch.float32),
            torch.tensor(post_trans, dtype=torch.float32), torch.eye(4), c2l)
        results['foundation_meta'] = dict(
            raw_img=raw, focal_length=torch.tensor(intrinsics[0][0, 0], dtype=torch.float32),
            baseline=torch.tensor(baseline, dtype=torch.float32),
            img_shape=torch.tensor(self.input_size), camera_names=list(CAMERAS),
            token=frame['token'], scene_name=frame['scene_name'],
            lidar2img=torch.tensor(np.stack(projections), dtype=torch.float32))
        return results


@PIPELINES.register_module()
class LoadFoundationSSCOccupancy:
    def __init__(self, occ_size=(256, 256, 32), point_cloud_range=(0, -25.6, -2, 51.2, 25.6, 4.4)):
        self.occ_size = tuple(occ_size)
        self.pc_range = point_cloud_range

    def __call__(self, results):
        #* 只读取当前帧 GT；不构造假深度/假语义监督，不读取未来图像。
        label = np.load(results['current_input']['occ_path'])
        if label.shape != self.occ_size:
            raise ValueError(f'occupancy shape={label.shape}, expected={self.occ_size}')
        if not np.isin(label, list(range(20)) + [255]).all():
            raise ValueError('occupancy 必须是映射后的 0..19/255 标签')
        results['gt_occ'] = torch.from_numpy(label.astype(np.int64, copy=True))
        results['foundation_meta'].update(
            pc_range=torch.tensor(self.pc_range, dtype=torch.float32),
            occ_size=torch.tensor(self.occ_size))
        return results


@PIPELINES.register_module()
class PackFoundationSSCInputs:
    def __init__(self, runner_format=False):
        self.runner_format = runner_format

    def __call__(self, results):
        #* 正式 MMCV runner：张量按 batch 堆叠；meta 按样本保留在 CPU，
        # 字符串 token 不参与 tensor scatter，原始双目图像由模型按需移到 GPU。
        if self.runner_format:
            from mmcv.parallel import DataContainer as DC
            return dict(
                img_inputs=tuple(DC(x, stack=True, pad_dims=None) for x in results['img_inputs']),
                img_metas=DC(results['foundation_meta'], cpu_only=True),
                gt_occ=DC(results['gt_occ'], stack=True, pad_dims=None))
        #* 独立验证脚本仍可使用普通 default_collate，不依赖并行 wrapper。
        return dict(img_inputs=results['img_inputs'],
                    img_metas=results['foundation_meta'], gt_occ=results['gt_occ'])
