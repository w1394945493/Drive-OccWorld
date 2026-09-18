# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time


@DETECTORS.register_module()
class BEVFormer(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 backwarded_prev_frame_num=0,
                 ):

        super(BEVFormer,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.backwarded_prev_frame_num = backwarded_prev_frame_num

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:

            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def _obtain_frozen_history_bev(self, imgs_queue, img_metas_list, drop_prev_index):
        """Obtain history BEV features iteratively.

        #* ================== Frozen history BEV：历史帧只前向，不反传 ==================
        # 这部分用于处理较早的历史帧。为了节省显存，图像 backbone/FPN 和
        # BEVFormer encoder 都放在 torch.no_grad() 中执行。
        #
        # 注意：no_grad 并不表示模型“不使用历史帧”。这些历史帧仍然会被编码成
        # prev_bev，并作为后续历史帧/当前帧 BEV encoder 的 temporal memory。
        # 只是 loss 不会沿这些历史帧的图像分支反向传播。
        #
        # imgs_queue: [B, L, Ncam, C, H, W]
        #   B    : batch size；
        #   L    : frozen history 的帧数，例如原 nuScenes 设置中可能是 -4/-3/-2；
        #   Ncam : 相机数量；
        #   C/H/W: 图像通道和尺寸。
        """
        is_training = self.training
        if is_training:
            #* [无梯度] 临时切到 eval 模式，保持原 BEVFormer 处理历史帧的做法：
            #* frozen history 仅用于生成 prev_bev，不更新 BN/dropout 状态，也不保留梯度。
            self.eval()

        #* [无梯度] frozen history 的图像 backbone/FPN 特征提取不参与反向传播。
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            # 将时间维 L 合并到 batch 维，便于一次性提取所有历史帧的图像特征。
            # [B, L, Ncam, C, H, W] -> [B*L, Ncam, C, H, W]
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            # extract_feat 只需要一个 meta 列表；这里用 frozen history 最后一帧的 meta
            # 作为占位/参考，真正逐帧送入 pts_bbox_head 时会重新取 each[i]。
            img_metas = [meta[len_queue - 1] for meta in img_metas_list]
            # 一次性提取所有 frozen history 帧的多尺度图像特征。
            # img_feats_list: num_levels 个特征图；
            # 每个尺度 shape 约为 [B, L, Ncam, C_feat, H_feat, W_feat]。
            img_feats_list = self.extract_feat( # stages*[B,Lin,Ncams,C,H,W]
                img=imgs_queue,
                len_queue=len_queue,
                img_metas=img_metas)

        prev_bev_list = []
        for i in range(len_queue):
            #* [无梯度] frozen history 的 BEVFormer encoder 也不参与反向传播。
            #* 因此 loss 不会通过这些较早历史帧回传到 BEV encoder / backbone。
            with torch.no_grad():
                # 逐帧取出当前历史帧的 meta 和多尺度图像特征。
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    # 场景首帧或跨 scene 时不能继承上一帧 BEV，需要重置 temporal memory。
                    prev_bev = None

                # 当前第 i 个历史帧的多尺度特征：
                # list[num_levels]，每个元素 shape 约为 [B, Ncam, C_feat, H_feat, W_feat]。
                img_feats = [each_scale[:, i] for each_scale in img_feats_list] # 4:(1 1 256 48 156) (1 1 256 24 78) (1 1 256 12 39) (1 1 256 6 20) # stages*[B,Ncams,C,H,W]  某一帧的feat
                # only_bev=True 表示只走 BEVFormer encoder，输出 BEV embedding，
                # 不走检测 decoder/box head。
                # prev_bev 会作为 temporal self-attention 的历史 BEV 输入。
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True) # (1 16384 256)
                # 保存每一帧生成的 BEV，后续 Drive-OccWorld 会从中构造 memory_queue。
                prev_bev_list.append(prev_bev)

            if i < drop_prev_index:
                # 数据增强/随机丢历史 BEV 时使用：在指定位置之前重置 prev_bev，
                # 迫使后续帧不依赖更早历史。
                prev_bev = None

        if len(prev_bev_list) > self.memory_queue_len - 1:
            # 只保留最近 memory_queue_len - 1 个历史 BEV。
            # 之后会再拼接当前参考帧 ref_bev，组成长度为 memory_queue_len 的 memory queue。
            prev_bev_list = prev_bev_list[-(self.memory_queue_len-1):]

        if is_training:
            #* [恢复梯度环境] 处理完 frozen history 后恢复训练模式，
            #* 保证当前帧/可反传历史帧正常训练。
            self.train()

        return prev_bev, prev_bev_list

    def _obtain_backwarded_history_bev(
            self, imgs_queue, prev_bev, prev_bev_list, backward_img_metas_list, backwarded_start_idx, backwarded_end_idx):
        """Obtain history BEV features iteratively, with gradients computed.

        #* ================== Backwarded history BEV：最近历史帧允许部分反传 ==================
        # 这部分处理 frozen history 之后、当前帧之前的最近若干历史帧。
        # 由 backwarded_start_idx/backwarded_end_idx 指定切片范围。
        #
        # 设计目的：
        # - 更早历史帧全部 no_grad，节省显存；
        # - 最近 backwarded_prev_frame_num 帧的 BEVFormer encoder 保留梯度，
        #   让模型能学习更近历史帧到当前帧的 temporal BEV 建模。
        #
        # 注意当前实现里：
        # - 图像 backbone/FPN 的特征提取仍在 torch.no_grad() 中；
        # - 但 pts_bbox_head(..., only_bev=True) 不在 no_grad 中，
        #   因此 BEVFormer encoder 这一段可以反向传播。
        """
        # 取需要保留 BEV encoder 梯度的最近历史帧。
        # 例如历史帧为 [-4,-3,-2,-1]，backwarded_start_idx=3、
        # backwarded_end_idx=4 时，这里只取 -1 帧。
        backward_prev_img = imgs_queue[:, backwarded_start_idx:backwarded_end_idx, ...]
        bs, len_queue, num_cams, C, H, W = backward_prev_img.shape
        # [B, L_back, Ncam, C, H, W] -> [B*L_back, Ncam, C, H, W]
        backward_prev_img = backward_prev_img.reshape(bs * len_queue, num_cams, C, H, W)
        # 作为 extract_feat 的 meta 输入；后面逐帧送入 BEV encoder 时会使用 cur_backward_img_metas。
        backward_img_metas = [meta[backwarded_start_idx] for meta in backward_img_metas_list]
        #* [无梯度] backwarded history 的图像 backbone/FPN 仍不保留梯度，用于控制显存。
        #* 因此这部分不会更新 img_backbone / img_neck。
        self.eval()
        with torch.no_grad():
            backward_img_feats_list = self.extract_feat(
                img=backward_prev_img,
                img_metas=backward_img_metas,
                len_queue=len_queue)
        self.train()
        #* [有梯度] 从这里开始恢复 train 模式，下面的 pts_bbox_head(..., only_bev=True)
        #* 不在 torch.no_grad() 中，因此最近历史帧的 BEVFormer encoder 可以反传。
        for idx, prev_idx in enumerate(range(backwarded_start_idx, backwarded_end_idx)):
            # prev_idx 是原 history 序列中的帧下标；
            # idx 是 backwarded 子序列内部的下标，用于从 backward_img_feats_list 取特征。
            cur_backward_img_metas = [each[prev_idx] for each in backward_img_metas_list]

            if not cur_backward_img_metas[0]['prev_bev_exists']:
                # 跨 scene 或没有有效上一帧时，重置 temporal memory。
                prev_bev = None

            # 当前 backwarded history 帧的多尺度图像特征。
            img_feats = [each_scale[:, idx] for each_scale in backward_img_feats_list]
            #* [有梯度] 这里不包 torch.no_grad()：
            #* 因此最近历史帧的 BEVFormer encoder 可以参与反向传播。
            #* [梯度截断] 但由于 img_feats 是 no_grad 提取的，
            #* 梯度不会继续传回图像 backbone/FPN。
            prev_bev = self.pts_bbox_head(
                img_feats, cur_backward_img_metas, prev_bev, only_bev=True)
            prev_bev_list.append(prev_bev)

        if len(prev_bev_list) > self.memory_queue_len - 1:
            # 同样只保留最近 memory_queue_len - 1 个历史 BEV。
            prev_bev_list = prev_bev_list[-(self.memory_queue_len-1):]

        return prev_bev, prev_bev_list

    def obtain_history_bev(self, img, img_metas, drop_prev_index=-1):
        #* ================== 历史帧 BEV 构建入口 ==================
        # img 只包含历史帧，不包含当前参考帧。
        # 例如总输入 queue 为 [-2, -1, current] 时，传入这里的是 [-2, -1]。
        # shape: [B, num_history_frames, Ncam, C, H, W]
        num_frames = img.shape[1]

        #* [梯度策略] 训练阶段可让最近若干历史帧走 backwarded history 分支；
        #* 测试阶段固定为 0，即全部历史帧只前向、不反传。
        backward_prev_frame_num = self.backwarded_prev_frame_num if self.training else 0

        # 将历史帧切成两段：
        #*   [0, backward_prev_start_idx)                     -> frozen history，无梯度；
        #*   [backward_prev_start_idx, backward_prev_end_idx) -> backwarded history，BEV encoder 有梯度。
        #
        # 例：num_frames=4，历史帧为 [-4,-3,-2,-1]，
        #     backward_prev_frame_num=1 时：
        #       backward_prev_start_idx = 4 - 1 = 3
        #       backward_prev_end_idx   = 3 + 1 = 4
        #     因此：
        #       frozen history     = img[:, :3]  -> [-4,-3,-2]
        #       backwarded history = img[:, 3:4] -> [-1]
        backward_prev_start_idx = num_frames - backward_prev_frame_num
        backward_prev_end_idx = backward_prev_start_idx + backward_prev_frame_num

        #* ================== 1. Frozen part：较早历史帧，只前向不反传 ==================
        #* [无梯度] 该分支中 backbone/FPN 和 BEVFormer encoder 都不反传。
        prev_img = img[:, :backward_prev_start_idx, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        # prev_bev: 最后一帧 frozen history 的 BEV，shape [B, bev_h*bev_w, C]。
        # prev_bev_list: 最近 memory_queue_len-1 个历史 BEV，用于后续拼接当前 ref_bev。
        prev_bev, prev_bev_list = self._obtain_frozen_history_bev(prev_img, prev_img_metas, drop_prev_index=drop_prev_index)

        #* ================== 2. Backwarded part：最近历史帧，BEV encoder 可反传 ==================
        #* [部分有梯度] 该分支中 backbone/FPN 不反传，BEVFormer encoder 反传。
        # 如果 backward_prev_frame_num=0，则跳过该分支，所有历史帧都走 frozen/no_grad。
        # 如果大于 0，则最近若干历史帧会在 BEV encoder 部分保留梯度。
        if backward_prev_frame_num > 0:
            prev_bev, prev_bev_list = self._obtain_backwarded_history_bev(
                img, prev_bev, copy.deepcopy(img_metas),
                backward_prev_start_idx, backward_prev_end_idx)
        return prev_bev, prev_bev_list

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
