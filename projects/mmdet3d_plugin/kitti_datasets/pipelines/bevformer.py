"""时序 BEVFormer 输入：严格保持原 Dataset 的预处理数值与打包约定。"""
import copy
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LoadTemporalKittiImages:
    def __call__(self, results):
        cfg = results['preprocess_cfg']
        if not cfg['load_img']:
            return results
        images = []
        #* 只加载历史/当前图像；未来图像路径不作为模型观测。
        for frame, meta in zip(results['input_frame_inputs'], results['img_metas']):
            cameras = []
            for path in frame['img_filename']:
                img = mmcv.imread(path, flag='color')  # BGR，与原实现一致。
                if img is None:
                    raise FileNotFoundError(f'Cannot read image: {path}')
                cameras.append(img.astype(np.float32) if cfg['to_float32'] else img)
            images.append(cameras)
            meta['ori_shape'] = [img.shape for img in cameras]
            meta['img_shape'] = list(meta['ori_shape'])
        results['_temporal_images'] = images
        return results


@PIPELINES.register_module()
class NormalizeTemporalKittiImages:
    def __call__(self, results):
        if not results['preprocess_cfg']['load_img']:
            return results
        cfg = results['preprocess_cfg']['img_norm_cfg']
        mean = np.asarray(cfg['mean'], dtype=np.float32)
        std = np.asarray(cfg['std'], dtype=np.float32)
        for cameras, meta in zip(results['_temporal_images'], results['img_metas']):
            for i, img in enumerate(cameras):
                if cfg.get('to_rgb', False):
                    img = img[..., ::-1]
                cameras[i] = (img - mean) / std
            meta['img_norm_cfg'] = copy.deepcopy(cfg)
        return results


@PIPELINES.register_module()
class PadTemporalKittiImages:
    def __call__(self, results):
        cfg = results['preprocess_cfg']
        if not cfg['load_img']:
            return results
        #* 归一化后仅向右/下补 0，不缩放或平移像素，无需改变 lidar2img。
        for cameras, meta in zip(results['_temporal_images'], results['img_metas']):
            shapes = []
            for i, img in enumerate(cameras):
                h, w = img.shape[:2]
                if cfg['pad_shape'] is not None:
                    th, tw = cfg['pad_shape']
                    if h > th or w > tw:
                        raise ValueError(f'Image shape {(h, w)} exceeds pad_shape {(th, tw)}')
                elif cfg['size_divisor'] is not None:
                    divisor = cfg['size_divisor']
                    th, tw = int(np.ceil(h / divisor) * divisor), int(np.ceil(w / divisor) * divisor)
                else:
                    th, tw = h, w
                padded = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
                padded[:h, :w] = img
                cameras[i] = padded
                shapes.append(padded.shape)
            meta['pad_shape'] = shapes
        return results


@PIPELINES.register_module()
class LoadTemporalKittiOccupancy:
    def __call__(self, results):
        if results['preprocess_cfg']['load_occ']:
            labels = []
            for path in results['occ_paths']:
                if not osp.isfile(path):
                    raise FileNotFoundError(f'Missing occupancy file: {path}')
                labels.append(np.load(path).astype(np.int64, copy=False))
            #* 顺序保持历史 + 当前 + 未来；不重映射标签，不改变 255 ignore。
            results['segmentation'] = np.stack(labels)
        return results


@PIPELINES.register_module()
class PackKittiWorldInputs:
    def __call__(self, results):
        cfg = results.pop('preprocess_cfg')
        images = results.pop('_temporal_images', None)
        if images is not None:
            results['img'] = np.stack([
                np.stack([img.transpose(2, 0, 1).copy() for img in cameras])
                for cameras in images])  # [T_input, N_cam, C, H, W]
        if not cfg['format_for_train']:
            return results  # 调试/可视化保留 numpy 及窗口信息。
        if results['img'] is None or results['segmentation'] is None:
            raise ValueError('format_for_train=True requires load_img=True and load_occ=True')
        packed = dict(
            img=DC(torch.from_numpy(results['img']).float(), stack=True),
            img_metas=DC(results['img_metas'], cpu_only=True),
            segmentation=DC(torch.from_numpy(results['segmentation']).long(), stack=False))
        #* 低维条件 Tensor 不允许默认的空间 padding；GT 保持 list[tensor]。
        for key in ('sdc_planning', 'sdc_planning_mask', 'command', 'vel_steering'):
            tensor = torch.from_numpy(results[key])
            tensor = tensor.long() if key == 'command' else tensor.float()
            packed[key] = DC(tensor, stack=True, pad_dims=None)
        return packed
