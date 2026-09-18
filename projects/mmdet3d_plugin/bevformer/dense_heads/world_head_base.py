import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn import xavier_init, constant_init
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from torch.nn.init import normal_
from ..utils import e2e_predictor_utils
from mmdet3d.models import builder
from einops import rearrange, repeat
from ..modules.conditionalnorm import Fourier_Embed as Fourier_Embed_v1
import math

def fourier_embedding(input, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal input embeddings.

    :param input: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.

    :return: an [N x dim] Tensor of positional embeddings.
    """

    if repeat_only:
        embedding = repeat(input, "b -> b d", d=dim)
    else:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=input.device)
        args = input[:, None].float() * freqs[None]
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding

class Fourier_Embed(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.dim = dim
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * self.embed_dim, 256),
            nn.ReLU(inplace=True),
        )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        try:
            xavier_init(self.mlp, distribution='uniform', bias=0.)
        except:
            pass

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b, c = t.shape
        # fourier embed
        t = rearrange(t, "b c -> (b c)")
        t = fourier_embedding(t, self.embed_dim)
        t = rearrange(t, "(b c) c2 -> b (c c2)", b=b, c=c, c2=self.embed_dim)
        # mlp
        t = self.mlp(t)
        return t

@HEADS.register_module()
class WorldHeadTemplate(BaseModule):

    def __init__(self,
                 *args,
                 num_classes,
                 # Architecture.
                 prev_render_neck=None,
                 transformer=None,
                 num_pred_fcs=2,
                 num_pred_height=1,

                 # Memory Queue configurations.
                 memory_queue_len=1,
                 turn_on_flow=False,
                 sem_norm=False,
                 obj_motion_norm=False,

                 # Embedding configuration.
                 use_can_bus=False,
                 can_bus_norm=True,
                 can_bus_dims=(0, 1, 2, 17),
                 use_plan_traj=True,
                 use_command=False,
                 use_vel_steering=False,
                 use_vel=False,
                 use_steering=False,
                 condition_ca_add='add',
                 use_fourier=True,

                 # target BEV configurations.
                 bev_h=30,
                 bev_w=30,
                 pc_range=None,

                 # loss functions.
                 loss_weight=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),

                 # evaluation configuration.
                 eval_within_grid=False,
                 **kwargs):

        # BEV configuration of reference frame.
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        # action_condition
        self.condition_ca_add = condition_ca_add
        self.use_fourier = use_fourier

        # memory queue
        self.memory_queue_len = memory_queue_len
        self.turn_on_flow = turn_on_flow
        self.sem_norm = sem_norm
        self.obj_motion_norm = obj_motion_norm
        # fourier_embed
        self.fourier_nhidden = 64
        # Embedding configurations.
        self.can_bus_norm = can_bus_norm
        self.use_can_bus = use_can_bus
        self.can_bus_dims = can_bus_dims  # (delta_x, delta_y, delta_z, delta_yaw)
        if self.use_can_bus and self.use_fourier:
            self.fourier_embed_canbus = Fourier_Embed(len(self.can_bus_dims), self.fourier_nhidden)
        # vel_steering
        self.use_vel_steering = use_vel_steering
        self.vel_steering_dims = 4        # (vx, vy, v_yaw, steering)
        if self.use_vel_steering and self.use_fourier:
            self.fourier_embed_velsteering = Fourier_Embed(self.vel_steering_dims, self.fourier_nhidden)
        # vel
        self.use_vel = use_vel
        self.vel_dims = 2        # (vx, vy)
        if self.use_vel and self.use_fourier:
            self.fourier_embed_vel = Fourier_Embed(self.vel_dims, self.fourier_nhidden)
        # steering
        self.use_steering = use_steering
        self.steering_dims = 1        # (steering)
        if self.use_steering and self.use_fourier:
            self.fourier_embed_steering = Fourier_Embed(self.steering_dims, self.fourier_nhidden)
        # command
        self.use_command = use_command
        self.command_dims = 1             # command
        if self.use_command and self.use_fourier:
            self.fourier_embed_command = Fourier_Embed(self.command_dims, self.fourier_nhidden)
        # plan_traj
        self.use_plan_traj = use_plan_traj
        self.plan_traj_dims = 2
        if self.use_plan_traj and self.use_fourier:
            self.fourier_embed_plantraj = Fourier_Embed(self.plan_traj_dims, self.fourier_nhidden)
        # action_condition
        if self.use_fourier:
            self.action_condition_dims = 256 * use_can_bus + 256 * use_vel_steering + 256 * use_vel + 256 * use_steering + 256 * use_command + 256 * use_plan_traj
        elif not self.use_fourier:
            self.action_condition_dims = 4 * use_can_bus + 4 * use_vel_steering + 2 * use_vel + 1 * use_steering + 1 * use_command + 2 * use_plan_traj

        # Network configurations.
        self.num_pred_fcs = num_pred_fcs
        # How many bins predicted at the height dimensions.
        # By default, 1 for BEV prediction.
        self.num_pred_height = num_pred_height

        # build prev_render_neck
        if prev_render_neck is not None:
            self.prev_render_neck = builder.build_head(prev_render_neck)
        else:
            self.prev_render_neck = None

        # build transformer architecture.
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        # set loss weight.
        self.loss_weight = np.array(loss_weight)
        assert self.loss_weight.shape[-1] == 1

        # set evaluation configurations.
        self.eval_within_grid = eval_within_grid
        self._init_layers()

        if self.sem_norm and not self.turn_on_flow: # occ pred_head
            sem_raymarching_branch = []
            for _ in range(self.num_pred_fcs):
                sem_raymarching_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
                sem_raymarching_branch.append(nn.LayerNorm(self.embed_dims))
                sem_raymarching_branch.append(nn.ReLU(inplace=True))
            sem_raymarching_branch.append(nn.Linear(self.embed_dims, self.num_pred_height * self.num_classes))   # C -> cls
            self.sem_raymarching_branch = nn.Sequential(*sem_raymarching_branch)

            self.param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)

            nhidden = 32
            self.mlp_shared = nn.Sequential(
                nn.Conv3d(self.num_classes, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Conv3d(nhidden, self.embed_dims // self.num_pred_height, kernel_size=3, padding=1)
            self.mlp_beta = nn.Conv3d(nhidden, self.embed_dims // self.num_pred_height, kernel_size=3, padding=1)
            self.reset_parameters()

        if self.obj_motion_norm and self.turn_on_flow: # flow pred_head
            flow_raymarching_branch = []
            flow_raymarching_branch.append(nn.Linear(self.embed_dims, self.num_pred_height * self.num_classes))   # C -> cls
            self.flow_raymarching_branch = nn.Sequential(*flow_raymarching_branch)

            self.param_free_norm = nn.LayerNorm(self.embed_dims, elementwise_affine=False)

            nhidden = 32
            self.fourier_embed = Fourier_Embed_v1(nhidden)

            self.mlp_shared = nn.Sequential(
                nn.Conv3d(self.num_classes * nhidden, nhidden, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.mlp_gamma = nn.Conv3d(nhidden, self.embed_dims // self.num_pred_height, kernel_size=3, padding=1)
            self.mlp_beta = nn.Conv3d(nhidden, self.embed_dims // self.num_pred_height, kernel_size=3, padding=1)
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.ones_(self.mlp_gamma.bias)
        nn.init.ones_(self.mlp_beta.bias)

    def _init_layers(self):
        """Initialize BEV prediction head."""
        # BEV query for the next frame.
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        # Embeds for previous frame number.
        self.prev_frame_embedding = nn.Parameter(torch.Tensor(self.memory_queue_len, self.embed_dims))
        # Embeds for CanBus information.
        # Use position & orientation information of next frame's canbus.
        if self.use_can_bus or self.use_command or self.use_vel_steering or self.use_vel or self.use_steering or self.use_plan_traj:
            can_bus_input_dim = self.action_condition_dims
            self.fusion_mlp = nn.Sequential(
                nn.Linear(can_bus_input_dim, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(inplace=True),
            )
            if self.can_bus_norm:
                self.fusion_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        try:
            self.transformer.init_weights()
            # Initialization of embeddings.
            normal_(self.prev_frame_embedding)
            xavier_init(self.fusion_mlp, distribution='uniform', bias=0.)
        except:
            pass


    @auto_fp16(apply_to=('prev_features'))
    def _get_next_bev_features(self, prev_features, img_metas, target_frame_index,
                               action_condition_dict, cond_norm_dict, tgt_points, ref_points, bev_h, bev_w):
        """ Forward function for each frame.

        Args:
            prev_features (Tensor): BEV features from previous frames input, with
                shape of (bs, num_frames, bev_h * bev_w, embed_dim).
            img_metas: information of reference frame inputs.
                key "future_can_bus": can_bus information of future frames,
                    Note, 0 represents the reference frame.
            target_frame_index: next frame information.
                For indexing target can_bus information.
            tgt_points (Tensor): query point coordinates in target frame coordinates.
            ref_points (Tensor): query point coordinates in previous frame coordinates.
        """
        #* ================== 论文 3.2: World Decoder W_D 输入查询 ==================
        # 1. BEV query
        # 对应论文中 "WD takes learnable BEV queries as input"：
        # bev_embedding 是未来帧 BEV query，后续通过 self/cross/action attention 生成 future BEV。
        bs, num_frames, _, emebd_dim = prev_features.shape
        dtype = prev_features.dtype
        #  * BEV queries.
        bev_queries = self.bev_embedding.weight.to(dtype)  # bev_h * bev_w, bev_dims
        bev_queries = bev_queries.unsqueeze(0)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)  # bs, bev_dims, bev_h, bev_w


        #* ================== 论文 3.2: action condition a_{+t} ==================
        # 2. action condition
        # 对应论文 "Flexible action conditions can be injected into WD"：
        # 根据配置把 can_bus / plan_traj / command / velocity / steering 拼成 action embedding。
        action_condition = None
        plan_traj = action_condition_dict['plan_traj']
        command, vel_steering = action_condition_dict['command'][:, target_frame_index], action_condition_dict['vel_steering'][:, target_frame_index]
        if self.use_can_bus: # * use_can_bus
            #* 当前配置 use_can_bus=True：这里是 can_bus 进入模型作为 action condition 的核心位置。
            #* 注意它不是只用于 BEV 对齐；这里会直接影响 WorldDecoder 对未来 occupancy 的生成。
            #   * Can-bus information.
            # future_can_bus 由 dataset 的 union2one() 整理，按 future frame 存储未来 ego-motion 信息。
            cur_can_bus = [img_meta['future_can_bus'][target_frame_index] for img_meta in img_metas]
            # can_bus_dims=(0,1,2,17)：默认取 delta_x、delta_y、delta_z、delta_yaw。
            cur_can_bus = np.array(cur_can_bus)[:, self.can_bus_dims]  # bs, 18
            cur_can_bus = torch.from_numpy(cur_can_bus).to(dtype).to(bev_pos.device)    # bs,4  (delta_x, delta_y, delta_z, delta_yaw)
            # fourier embed
            if self.use_fourier:
                cur_can_bus = self.fourier_embed_canbus(cur_can_bus)
            action_condition = cur_can_bus  #* can_bus 成为 action_condition 的第一部分。

        elif self.use_plan_traj:
            #  * Plan Traj
            cur_can_bus = plan_traj[:, -1, :2].float() # bs, 2
            # fourier embed
            if self.use_fourier:
                cur_can_bus = self.fourier_embed_plantraj(cur_can_bus)
            action_condition = cur_can_bus

        if self.use_command:
            #* command 会和 can_bus 拼接，形成更完整的 action condition。
            command = command.unsqueeze(0)  # bs,1 (command)
            # fourier embed
            if self.use_fourier:
                command = self.fourier_embed_command(command)
            if action_condition is None:
                action_condition = command
            else:
                action_condition = torch.cat([action_condition, command], dim=-1)

        if self.use_vel_steering:
            # vel_steering: bs,4  (vx, vy, v_yaw, steering)
            # fourier embed
            if self.use_fourier:
                vel_steering = self.fourier_embed_velsteering(vel_steering)
            if action_condition is None:
                action_condition = vel_steering
            else:
                action_condition = torch.cat([action_condition, vel_steering], dim=-1)

        if self.use_vel:
            #* 当前配置 use_vel=True：从 vel_steering 中取 vx/vy，与 can_bus/command 拼接控制未来 occupancy。
            # vel_steering: bs,4  (vx, vy, v_yaw, steering)
            vel = vel_steering[:, :self.vel_dims]
            # fourier embed
            if self.use_fourier:
                vel = self.fourier_embed_vel(vel)
            if action_condition is None:
                action_condition = vel
            else:
                action_condition = torch.cat([action_condition, vel], dim=-1)

        if self.use_steering:
            # vel_steering: bs,4  (vx, vy, v_yaw, steering)
            steering = vel_steering[:, -1:]
            # fourier embed
            if self.use_fourier:
                steering = self.fourier_embed_steering(steering)
            if action_condition is None:
                action_condition = steering
            else:
                action_condition = torch.cat([action_condition, steering], dim=-1)

        if action_condition is not None:
            #* can_bus/command/velocity 拼接后经 fusion_mlp 投影到 embed_dims。
            #* 这个 action_condition 会送入 WorldDecoder 的 cross_attn_action，
            #* 作为 key/value 控制未来 BEV query，进而控制未来 occupancy logits。
            # fusion_mlp 将不同来源的动作条件投影到 transformer embed_dims，
            # 后续作为 cross_attn_action 的 key/value 或直接加到 BEV query 上。
            action_condition = self.fusion_mlp(action_condition.float())

        #  * sum different query embedding together.
        #    (bs, bev_h * bev_w, dims)
        if (self.use_can_bus or self.use_plan_traj) and self.condition_ca_add == 'add':
            bev_queries_input = bev_queries + action_condition.unsqueeze(1)
        else:
            bev_queries_input = bev_queries


        #* ================== 论文 3.2: Memory Queue W_M + ConditionalNorm ==================
        # 3. obtain prev embeddings (bs, num_frames, bev_h * bev_w, dims).
        # prev_features 是 WM 中的历史/当前 BEV embeddings；
        # prev_render_neck=ConditionalNorm，对应论文 Eq.5 的 semantic/motion 条件归一化。
        if self.prev_render_neck:
            render_dict = self.prev_render_neck(prev_features.view(bs, num_frames, bev_w, bev_h, emebd_dim).transpose(2,3), cond_norm_dict)
            prev_features = render_dict['bev_embed']
            if not self.turn_on_flow:
                bev_sem_pred = render_dict['bev_sem_pred']  # B,cls,H,W
            else:
                bev_sem_pred = None

        frame_embedding = self.prev_frame_embedding
        prev_features_input = (prev_features +
                               frame_embedding[None, :, None, :])

        #* ================== 论文 3.2: World Decoder W_D ==================
        # 4. do transformer layers to get BEV features.
        # transformer=PredictionTransformer/WorldDecoder：
        # self-attn 建模未来 query，cross-attn 读取 WM 历史特征，cross_attn_action 注入动作条件。
        next_bev_feat = self.transformer(
            prev_features_input,
            bev_queries_input,
            tgt_points=tgt_points,
            ref_points=ref_points,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            img_metas=img_metas,
            action_condition=action_condition,
        )

        # # 5.1 sem_norm
        # if self.sem_norm and not self.turn_on_flow:
        #     next_bev_feat, bev_sem_pred = self.forward_sem_norm(next_bev_feat)
        # # 5.2 obj_motion_norm
        # if self.obj_motion_norm and self.turn_on_flow:
        #     next_bev_feat_norm, bev_sem_pred = self.forward_obj_motion_norm(next_bev_feat.clone(), occ_3D)
        #     next_bev_feat = next_bev_feat_norm
        # elif self.turn_on_flow:
        #     bev_sem_pred = None

        return next_bev_feat, bev_sem_pred

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self,
                prev_feats,  # memory queue BEV 特征，[B, memory_queue_len, H*W, C]。
                img_metas,  # 当前参考帧 meta；可提供 future_can_bus / future pose 等信息。
                target_frame_index,  # 当前要预测的未来步编号，1 表示 t+1，2 表示 t+2。
                action_condition_dict,  # 动作条件字典；包含 command / vel_steering / plan_traj。
                cond_norm_dict,  # 条件归一化字典；包含 future2history / occ_gts 等。
                tgt_points,  # 目标坐标：future BEV query 在未来帧自身 BEV 坐标系中的位置，[B, H*W, 2]。
                ref_points,  # 采样坐标：future query 映射到各 memory BEV 后的位置，[B, H*W, memory_queue_len, 2]。
                bev_h, bev_w,):  # BEV 网格尺寸 H/W。
        f"""Forward function: a wrapper function for self._get_next_bev_features

        #* ================== WorldHeadBase.forward：预测一个未来步的 BEV feature ==================
        # 该函数由 Drive_OccWorld.future_pred() 在自回归循环中逐步调用：
        #   prev_feats + future query 坐标 + action condition
        #       -> WorldDecoder
        #       -> target_frame_index 对应的 future BEV feature。
        #
        # tgt_points 和 ref_points 的区别：
        #   - tgt_points: future BEV query 在“未来帧自身坐标系”中的规则网格位置；
        #   - ref_points: 同一批 future query 映射到“历史/当前 BEV memory”后的采样位置，
        #     用于 WorldDecoder 的 temporal cross-attention / deformable attention。
        #
        Args:
            prev_feats (Tensor): BEV features from previous frames input, with
                shape of (bs, num_frames, bev_h * bev_w, embed_dim).
            img_metas: information of reference frame inputs.
                key "future_can_bus": can_bus information of future frames,
                    Note, 0 represents the reference frame.
            target_frame_index: the future frame index to predict.
            action_condition_dict: dict of action/planning conditions.
            cond_norm_dict: dict for semantic-/motion-conditional normalization.
            tgt_points (Tensor): query point coordinates in target future frame coordinates,
                shape [B, H*W, 2].
            ref_points (Tensor): query point coordinates aligned to previous/memory BEV
                coordinates, shape [B, H*W, memory_queue_len, 2].
            bev_h, bev_w: spatial shape of BEV query map.
        Returns:
            next_bev_feature (Tensor): predicted BEV features of target future frame,
                shape [num_decoder_layers, B, H*W, C] when return_intermediate=True.
            bev_sem_pred: auxiliary semantic rendering output from ConditionalNorm branch.
        """
        bs, num_frames, bev_grids_num, bev_dims = prev_feats.shape
        assert bev_dims == self.embed_dims
        assert bev_h * bev_w == bev_grids_num
        assert bev_h * bev_w == tgt_points.shape[1]

        next_bev_feat, bev_sem_pred = self._get_next_bev_features(
            prev_feats, img_metas, target_frame_index, action_condition_dict,
            cond_norm_dict, tgt_points, ref_points, bev_h, bev_w)
        return next_bev_feat, bev_sem_pred

    def forward_head(self, next_bev_feats):
        """Get freespace estimation from multi-frame BEV feature maps.

        Args:
            next_bev_feats (torch.Tensor): with shape as
                [pred_frame_num, inter_num, bs, bev_h * bev_w, dims]
        """
        pass


@HEADS.register_module()
class WorldHeadBase(WorldHeadTemplate):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
