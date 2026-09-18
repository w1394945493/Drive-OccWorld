# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
            mlvl_feats: stages*[B,Ncams,C,H,W]
            bev_queries: H*W, C
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) # HW,B,C
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)   # HW,B,C

        # ===========================================================#
        # Unified alignment method for both nuPlan and nuScenes.
        # obtain rotation angle and shift with ego motion
        #* （1）can_bus 用途一：计算 BEV temporal self-attention 的 shift。
        #* 这里期望 img_meta['can_bus'][:3] 表示“当前帧相对上一帧”的 global delta_xyz，
        #* 再通过 lidar2global_rotation 转到当前 LiDAR 坐标系，得到 BEV 平面 x/y shift。
        #*
        #* 对 nuScenes：该字段通常来自原始 can_bus / ego pose 在 Dataset queue 中整理出的帧间位移。
        #*
        #* 对 SemanticKITTI：数据集没有真实 CAN bus。当前 tools/semantickitti_converter.py
        #* 会根据“当前 occupancy 关键帧”和“上一个 occupancy 关键帧”的 poses.txt 位姿差分，
        #* 构造 pseudo can_bus：
        #*   - can_bus[:3] = 当前 occupancy 关键帧 - 上一个 occupancy 关键帧的 global delta_xyz；
        #*   - 首个 occupancy 关键帧没有上一关键帧，因此 can_bus[:3] = 0。
        #* 这里的“上一帧”不是原始图像序列中的上一张图，而是有 occupancy 标注的上一个 token，
        #* 例如 000000 -> 000005 -> 000010 中，000010 使用 000005 -> 000010 的位移。
        #*
        #* 注意：如果误把 can_bus[:3] 写成绝对全局位置，shift_x/shift_y 会被错误放大，
        #* 可能导致历史 BEV 对齐错误。
        delta_global = np.array([each['can_bus'][:3] for each in kwargs['img_metas']])  # total_len,3  -4帧为0,其余帧为global下的delta_xyz 
        lidar2global_rotation = np.array([each['lidar2global_rotation'] for each in kwargs['img_metas']])
        
        delta_lidar = []
        for i in range(bs):
            delta_lidar.append(np.linalg.inv(lidar2global_rotation[i]) @ delta_global[i])
        delta_lidar = np.array(delta_lidar) # total_len, 3  -4帧为0,其余帧为lidar坐标系下的delta_xyz
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        shift_y = delta_lidar[:, 1] / grid_length_y / bev_h
        shift_x = delta_lidar[:, 0] / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(np.array([shift_x, shift_y])).permute(1, 0)  # xy, bs -> bs, xy    bev平面下的delta_xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                rotated_prev_bev = []
                for i in range(bs):
                    # ========================================================#
                    # num_prev_bev = prev_bev.size(1)
                    #* （2）can_bus 用途二：旋转历史 BEV。
                    #* img_meta['can_bus'][-1] 被当作当前帧相对上一帧的 yaw delta，单位是 degree。
                    #* torchvision rotate() 接收角度制，因此这里直接使用 can_bus[-1]。
                    #*
                    #* 对 SemanticKITTI：tools/semantickitti_converter.py 会从 poses.txt 中提取
                    #* 当前/上一 occupancy 关键帧的 yaw，并写入：
                    #*   - can_bus[-2] = yaw delta，弧度制，主要留给 can_bus_mlp；
                    #*   - can_bus[-1] = yaw delta，角度制，供 rotate_prev_bev 使用。
                    #* 如果该字段仍是 0 或绝对 yaw，而不是相邻关键帧 yaw delta，
                    #* 历史 BEV 的旋转补偿会不准确。
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    rotated_prev_bev.append(tmp_prev_bev[:, 0])
                prev_bev = torch.stack(rotated_prev_bev, 1)

        # ===============================================================#
        # add can bus signals
        #* （3）can_bus 用途三：作为 BEV query 的运动/位姿条件。
        #* 18 维 can_bus 会经过 can_bus_mlp 投影到 embed_dims，并加到 bev_queries 上；
        #* self.use_can_bus=True 时生效，用于让 BEV encoder 感知自车运动状态。
        #*
        #* 对 SemanticKITTI：这里使用的是 pseudo can_bus，不是真实车辆 CAN bus：
        #*   - can_bus[:3]  由相邻 occupancy 关键帧 pose 差分得到 global delta_xyz；
        #*   - can_bus[3:7] 写入当前帧 ego/LiDAR -> global 朝向四元数，作为弱姿态条件；
        #*   - can_bus[-2] 写入 yaw delta rad；
        #*   - can_bus[-1] 写入 yaw delta degree。
        #* 这只能近似提供自车运动信息，不能等价替代真实速度、加速度、方向盘角等 CAN bus 信号。
        #* 如果希望做完全无运动条件的 ablation，可在配置中设置 use_can_bus=False。
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats): # mlvl_feats: stages*[B,Ncams,C,H,W]
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # Ncams,B,HW,C
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)   # Ncams,B,HW_total,C
        spatial_shapes = torch.as_tensor(           # stages,2  (h_stage,w_stage)
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, HW_total, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,    # HW,B,C
            feat_flatten,   # Ncams,HW_total,B,C
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,    # HW,B,C
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,  # HW,B,C
            shift=shift,        # bs, xy    bev平面下的delta_xy
            **kwargs
        )

        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
