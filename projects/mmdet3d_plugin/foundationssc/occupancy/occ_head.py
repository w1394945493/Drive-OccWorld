"""移植 FoundationSSC OccHead：低分辨率分类，再三线性上采样 logits。"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmcv.cnn import build_conv_layer, build_norm_layer
from .losses import geo_scal_loss, sem_scal_loss, CE_ssc_loss


class OccHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        empty_idx=0,
        ignore_index=255,
        num_level=1,
        with_cp=True,
        occ_size=[256, 256, 32],
        loss_weight_cfg=None,
        balance_cls_weight=True,
        conv_cfg=dict(type="Conv3d", bias=False),
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        class_frequencies=None,
        train_cfg=None,
        test_cfg=None,
    ):
        super(OccHead, self).__init__()

        if type(in_channels) is not list:
            in_channels = [in_channels]

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.empty_idx = empty_idx
        self.ignore_index = ignore_index
        if num_level != 1 or len(in_channels) != 1:
            raise ValueError('当前按原配置仅使用最高分辨率 neck 特征，num_level/in_channels 长度应为 1')
        if out_channel < 2 or not 0 <= empty_idx < out_channel or 0 <= ignore_index < out_channel:
            raise ValueError('类别数、empty_idx 或 ignore_index 非法')
        if len(occ_size) != 3 or any(int(x) != x or x <= 0 for x in occ_size):
            raise ValueError('occ_size 必须为正整数 X,Y,Z')

        self.with_cp = with_cp

        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg

        self.occ_size = occ_size
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get(
            "loss_voxel_ce_weight", 1.0
        )
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get(
            "loss_voxel_sem_scal_weight", 1.0
        )
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get(
            "loss_voxel_geo_scal_weight", 1.0
        )

        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(
                build_conv_layer(
                    conv_cfg,
                    in_channels=self.in_channels[i],
                    out_channels=mid_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    conv_cfg,
                    in_channels=mid_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
            self.occ_convs.append(occ_conv)

        # loss functions
        if balance_cls_weight:
            frequencies = np.asarray(class_frequencies, dtype=np.float64)
            if frequencies.shape != (out_channel,) or not np.isfinite(frequencies).all() or (frequencies + .001 <= 1).any():
                raise ValueError('class_frequencies 需与类别数一致，且能生成有限正的 1/log 权重')
            class_weights = torch.from_numpy(1 / np.log(frequencies + 0.001)).float()
        else:
            class_weights = torch.ones(out_channel) / out_channel
        #* 修复原普通 tensor 不随模型迁移设备的问题；仍使用原 1/log(freq+0.001) 权重。
        self.register_buffer('class_weights', class_weights)

    def forward(self, voxel_feats, img_metas=None, img_feats=None, gt_occ=None):
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level

        output_occs = []
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            if self.with_cp:
                output_occs.append(checkpoint(occ_conv, feats, use_reentrant=False))
            else:
                output_occs.append(occ_conv(feats))

        result = {
            "output_voxels": F.interpolate(
                output_occs[0],
                size=self.occ_size,
                mode="trilinear",
                align_corners=False,
            ).contiguous()
        }
        return result

    def loss(self, output_voxels, target_voxels):
        #* 不缩小 GT、不把 ignore=255 当空类。轴顺序保持 [B,C,X,Y,Z] / [B,X,Y,Z]。
        expected = (output_voxels.shape[0], *output_voxels.shape[2:])
        if output_voxels.ndim != 5 or output_voxels.shape[1] != self.out_channel or tuple(target_voxels.shape) != expected:
            raise ValueError(f'occupancy logits/GT 形状不匹配：{tuple(output_voxels.shape)} / {tuple(target_voxels.shape)}')
        if target_voxels.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
            raise TypeError('gt_occ 必须是整型语义类别，不接受浮点标签')
        target_voxels = target_voxels.to(device=output_voxels.device, dtype=torch.long)
        valid = target_voxels != self.ignore_index
        if ((target_voxels[valid] < 0) | (target_voxels[valid] >= self.out_channel)).any():
            raise ValueError('gt_occ 含未映射类别；应为 0..num_classes-1 或 ignore_index')
        if not valid.any():
            #* 原 CE/semantic scaling 在全 ignore 时会 NaN/除零；返回可反传的零损失。
            zero = output_voxels.float().sum() * 0
            return {key: zero for key in ('loss_voxel_ce', 'loss_voxel_sem_scal', 'loss_voxel_geo_scal')}
        #* scaling 内含 BCE；强制 FP32 并关闭 autocast，避免混合精度下不安全的 BCE。
        with torch.autocast(device_type=output_voxels.device.type, enabled=False):
            return self._loss_fp32(output_voxels.float(), target_voxels)

    def _loss_fp32(self, output_voxels, target_voxels):
        #* 原三项占据监督：逐体素类别 CE、各类别 precision/recall/specificity、空/非空几何。
        loss_dict = {}
        loss_dict["loss_voxel_ce"] = self.loss_voxel_ce_weight * CE_ssc_loss(
            output_voxels,
            target_voxels,
            self.class_weights.type_as(output_voxels),
            ignore_index=self.ignore_index,
        )
        loss_dict["loss_voxel_sem_scal"] = (
            self.loss_voxel_sem_scal_weight
            * sem_scal_loss(output_voxels, target_voxels, ignore_index=self.ignore_index)
        )
        loss_dict["loss_voxel_geo_scal"] = (
            self.loss_voxel_geo_scal_weight
            * geo_scal_loss(
                output_voxels,
                target_voxels,
                ignore_index=self.ignore_index,
                non_empty_idx=self.empty_idx,
            )
        )

        return loss_dict
