import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.cnn import xavier_init, constant_init
from mmdet.models import HEADS, build_head, build_loss
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import xavier_init
from torch.nn.init import normal_
from einops import rearrange
import copy
from projects.mmdet3d_plugin.bevformer.modules.collision_optimization import CollisionNonlinearOptimizer
from projects.mmdet3d_plugin.bevformer.utils.cost import Cost_Function

def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension

# Grid sampler
# Sample a smaller receptive-field bev from larger one
class BevFeatureSlicer(nn.Module):
    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()
        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False

            bev_resolution, bev_start_position, bev_dimension= calculate_birds_eye_view_parameters(
                grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
            )

            map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound']
            )

            self.map_x = torch.arange(
                map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])

            self.map_y = torch.arange(
                map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

            # convert to normalized coords
            self.norm_map_x = self.map_x / (- bev_start_position[0])
            self.norm_map_y = self.map_y / (- bev_start_position[1])

            tmp_m, tmp_n = torch.meshgrid(
                self.norm_map_x, self.norm_map_y)  # indexing 'ij'
            tmp_m, tmp_n = tmp_m.T, tmp_n.T  # change it to the 'xy' mode results

            self.map_grid = torch.stack([tmp_m, tmp_n], dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)

            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)


@HEADS.register_module()
class PoseEncoder(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=2,
        num_modes=3,
        num_fut_ts=1,
        init_cfg=None
    ):
        super().__init__(init_cfg)
        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1

        pose_encoder = []

        for _ in range(num_layers - 1):
            pose_encoder.extend([
                nn.Linear(in_channels, out_channels),
                nn.ReLU(True)])
            in_channels = out_channels
        pose_encoder.append(nn.Linear(out_channels, out_channels))
        self.pose_enc = nn.Sequential(*pose_encoder)

    def forward(self,x):
        # x: N*2,
        pose_feat = self.pose_enc(x)
        return pose_feat


@HEADS.register_module()
class PoseDecoder(BaseModule):

    def __init__(
            self,
            in_channels,
            num_layers=2,
            num_modes=3,
            num_fut_ts=1,
            init_cfg = None):
        super().__init__(init_cfg)

        self.num_modes = num_modes
        self.num_fut_ts = num_fut_ts
        assert num_fut_ts == 1

        pose_decoder = []
        for _ in range(num_layers - 1):
            pose_decoder.extend([
                nn.Linear(in_channels, in_channels),
                nn.ReLU(True)])
        pose_decoder.append(nn.Linear(in_channels, num_modes*num_fut_ts*2))
        self.pose_dec = nn.Sequential(*pose_decoder)

    def forward(self, x):
        # x: ..., D
        rel_pose = self.pose_dec(x).reshape(*x.shape[:-1], self.num_modes, 2)
        rel_pose = rel_pose.squeeze(1)
        return rel_pose

@HEADS.register_module()
class PlanHead_v1(BaseModule):
    """Head of Ego-Trajectory Planning.

    PlanHead_v1 是 Drive-OccWorld 中带「候选轨迹代价计算」的规划头：
    1. 输入当前/未来 BEV 特征、候选自车轨迹 sample_traj、语义 occupancy 和 command；
    2. 根据 command 从候选轨迹中筛选 left / forward / right 对应候选；
    3. 从 BEV 特征预测 cost volume，并结合 occupancy / drivable area 计算每条候选轨迹代价；
    4. 选择代价最小的候选轨迹作为 coarse plan；
    5. 再用 PlanTransformer 基于 BEV 特征进一步 refine，回归下一步自车位移。
    """

    def __init__(self,
                 # Architecture.
                 with_adapter=True,
                 transformer=None,
                 plan_grid_conf=None,

                 # class
                 instance_cls = [2,3,4,5,6,7,9,10],
                 drivable_area_cls = [11],
                 sample_num=1800,

                 # positional encoding
                 bev_h=200,
                 bev_w=200,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),

                 # loss
                 loss_planning=None,
                 loss_collision=None,

                 *args,
                 **kwargs):

        # BEV configuration of reference frame.
        super().__init__(**kwargs)
        #* ================== 1. 轨迹代价函数 ==================
        # Cost_Function 聚合多种 planning cost：
        # - cost volume cost: 从 BEV 特征预测出的可学习代价图；
        # - safety cost: 候选轨迹是否碰撞动态障碍物；
        # - headway cost: 自车前向安全距离；
        # - rule cost: 是否偏离可行驶区域。
        # forward/select 阶段会用它给每条候选轨迹打分，然后选择总代价最小的候选。
        self.cost_function = Cost_Function(plan_grid_conf)

        #* ================== 2. 语义类别定义：哪些是障碍物，哪些是可行驶区域 ==================
        # instance_cls: 在 fine-grained occupancy 中被视作动态/可碰撞目标的类别。
        # 默认对应 bicycle / bus / car / construction vehicle / motorcycle /
        # pedestrian / trailer / truck 等交通参与者。
        # forward 中会从 sem_occupancy 提取这些类别，得到 instance_occupancy，
        # 用于 safety cost 和 headway cost。
        # 原写法：只是普通 Tensor 属性，不会随 model.cuda()/model.to() 自动迁移设备。
        # self.instance_cls = torch.tensor(instance_cls, requires_grad=False)  # 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'pedestrian', 'trailer', 'truck'
        # register_buffer: 表示它不是可学习参数，但属于模块内常量状态；
        # persistent=False 表示不写入 checkpoint，因为类别 id 已由 config 决定。
        self.register_buffer(
            'instance_cls',
            torch.tensor(instance_cls, dtype=torch.long),
            persistent=False)

        # drivable_area_cls: 可行驶区域类别。
        # forward 中会从 sem_occupancy 提取 drivable_area，
        # 用于 rule cost，惩罚驶出可行驶区域的候选轨迹。
        # 原写法：只是普通 Tensor 属性，不会随 model.cuda()/model.to() 自动迁移设备。
        # self.drivable_area_cls = torch.tensor(drivable_area_cls, requires_grad=False)  # 'drivable_area'
        # 同样注册成 buffer，避免 CPU/GPU 设备不一致问题。
        self.register_buffer(
            'drivable_area_cls',
            torch.tensor(drivable_area_cls, dtype=torch.long),
            persistent=False)

        #* ================== 3. 候选轨迹组织方式 ==================
        # Dataset 中 sample_traj 默认生成 sample_num 条候选自车轨迹。
        # sample_num 默认 1800，可由 config 统一修改。
        # 默认按三组排列：[Left, Straight, Right] = [600, 600, 600]。
        # forward 时会根据 command 只取对应方向的一组候选，再 repeat 到同样数量，
        # 这样让 planner 在指定高层驾驶指令下选局部最优候选轨迹。
        # 注意：这里必须和 Dataset 中 candidate_sample_num 保持一致。
        self.sample_num = sample_num
        assert self.sample_num % 3 == 0
        self.num = int(self.sample_num / 3)

        #* ================== 4. BEV 坐标/分辨率对齐 ==================
        # BEVFormer 产生的 BEV 特征网格和 planning cost 使用的网格分辨率不同：
        # - bevformer_bev_conf: 原 BEVFormer BEV 范围/分辨率；
        # - plan_grid_conf: planner/cost volume 使用的范围/分辨率。
        # BevFeatureSlicer 负责把 BEV 特征采样/裁剪到 planning grid 上，
        # 后续 costvolume_head 和 PlanTransformer 都在该 planning grid 上工作。
        bevformer_bev_conf = {
            'xbound': [-51.2, 51.2, 0.512],
            'ybound': [-51.2, 51.2, 0.512],
            'zbound': [-10.0, 10.0, 20.0],
        }
        self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, plan_grid_conf)

        # TODO: reimplement it with down-scaled feature_map
        self.embed_dims = transformer.embed_dims
        self.with_adapter = with_adapter
        if with_adapter:
            #* BEV adapter: 对采样后的 BEV 特征做一个轻量 residual refinement。
            # 输入/输出 channel 都是 embed_dims，不改变 shape。
            # 作用可以理解为：让原本服务于 occupancy/world model 的 BEV 特征，
            # 进一步适配 planning/cost 任务。
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims // 2, self.embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)

        #* ================== 5. Cost volume 预测头 ==================
        # 从 BEV 特征预测一个单通道 cost map/cost volume: [B, C, H, W] -> [B, 1, H, W]。
        # 这个 costvolume 是可学习的轨迹代价图：候选轨迹经过的位置会在上面采样得到代价。
        # 它会与 occupancy-based safety/headway/rule cost 一起组成最终候选轨迹代价。
        self.costvolume_head = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.embed_dims),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dims, 1, kernel_size=1, padding=0),
        )

        #* ================== 6. 选中候选轨迹的 pose encoder ==================
        # select() 会从 sample_num 条候选中选出总代价最低的一条 coarse trajectory。
        # pose_encoder 将该候选轨迹的 [dx, dy, dyaw] / [x, y, yaw] 三维描述
        # 映射到 embed_dims 维特征，作为 PlanTransformer 的 prev_pose 条件。
        self.pose_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.ReLU(True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        #* ================== 7. PlanTransformer：基于 BEV refine 规划结果 ==================
        # positional_encoding 为 planning BEV grid 构造位置编码。
        # transformer 接收：
        # - plan query / navigation command embedding；
        # - 当前 BEV 特征；
        # - cost 最小候选轨迹编码后的 prev_pose；
        # 输出 refined plan query，用于最终回归下一步自车位移。
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)

        #* ================== 8. 规划回归头 ==================
        # 当前实现每次只回归下一步 planning pose，所以 planning_steps=1。
        # reg_branch 将 PlanTransformer 输出的 plan feature 映射为 [dx, dy]。
        # 多步未来规划是在 Drive_OccWorld.future_pred() 中逐帧自回归调用 plan_head 实现的。
        self.planning_steps = 1
        self.reg_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.planning_steps * 2),
        )

        #* ================== 9. 规划损失 ==================
        # loss_planning: 通常是 PlanningLoss，对预测自车轨迹和 GT sdc_planning 做 L2/ADE 类监督。
        # loss_collision: 可选的 collision loss 列表；当前配置常设为空列表。
        # 注意：候选轨迹 cost ranking 的 loss 不是这里定义的，
        # 而是在 loss_cost() 中用 Cost_Function 单独计算。
        self.loss_planning = build_loss(loss_planning)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = nn.ModuleList(self.loss_collision)

        self._init_layers()

    def _init_layers(self):
        """Initialize BEV prediction head."""
        # plan query for the next frame.
        self.plan_embedding = nn.Embedding(1, self.embed_dims)
        # navi embed.
        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        # mlp_fuser
        fuser_dim = 2
        self.mlp_fuser = nn.Sequential(
                nn.Linear(self.embed_dims*fuser_dim, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        try:
            self.transformer.init_weights()
            # Initialization of embeddings.
            normal_(self.plan_embedding)
            normal_(self.navi_embedding)
            xavier_init(self.mlp_fuser, distribution='uniform', bias=0.)
        except:
            pass

    def loss(self, outs_planning, sdc_planning, sdc_planning_mask, future_gt_bbox=None):
        """
            outs_planning:      B,Lout,mode=1,2
            sdc_planning:       B,Lout,3
            sdc_planning_mask:  B,Lout,2   valid_frmae=1
            future_gt_bbox:     Lout*[N_box个bbox_3d]
        """
        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](outs_planning, sdc_planning[..., :3], torch.any(sdc_planning_mask, dim=-1), future_gt_bbox)
            loss_dict[f'loss_collision_{i}'] = loss_collision
        loss_ade = self.loss_planning(outs_planning, sdc_planning, torch.any(sdc_planning_mask, dim=-1))
        loss_dict.update(dict(loss_ade=loss_ade))
        return loss_dict

    def loss_cost(self, trajs, gt_trajs, cost_volume, instance_occupancy, drivable_area):
        '''
        trajs: torch.Tensor (B, N, 3)
        gt_trajs: torch.Tensor (B, 3)
        cost_volume: torch.Tensor (B, 200, 200)
        instance_occupancy: torch.Tensor(B, 200, 200)
        drivable_area: torch.Tensor(B, 200, 200)
        '''
        if gt_trajs.ndim == 2:
            gt_trajs = gt_trajs[:, None]

        gt_cost_fo = self.cost_function(cost_volume, gt_trajs[:,:,:2], instance_occupancy, drivable_area)

        sm_cost_fo = self.cost_function(cost_volume, trajs[:,:,:2], instance_occupancy, drivable_area)

        L = F.relu(gt_cost_fo - sm_cost_fo)

        return torch.mean(L)

    def select(self, trajs, cost_volume, instance_occupancy, drivable_area, k=1):
        '''
        trajs: torch.Tensor (B, N, 3)
        cost_volume: torch.Tensor (B, 200, 200)
        instance_occupancy: torch.Tensor(B, 200, 200)
        drivable_area: torch.Tensor(B, 200, 200)
        '''
        sm_cost_fo = self.cost_function(cost_volume, trajs[:,:,:2], instance_occupancy, drivable_area)

        CS = sm_cost_fo
        CC, KK = torch.topk(CS, k, dim=-1, largest=False)   # B,N_sample

        # trajs 可能在 GPU；torch.arange 默认在 CPU，索引 CUDA tensor 会报设备不一致。
        ii = torch.arange(len(trajs), device=trajs.device)
        select_traj = trajs[ii[:,None], KK].squeeze(1) # (B, 3)

        return select_traj

    @auto_fp16(apply_to=('bev_feats'))
    def forward(self, bev_feats, trajs, sem_occupancy, command, gt_trajs=None):
        """Forward function for each frame.

        注意：PlanHead_v1 每次只处理“某一个时间步”的一步规划。
        外层 Drive_OccWorld.future_pred() 会对不同 future_frame_index 多次调用本函数：
        - sample_traj[:, :, 0] 在参考帧处预测 ref -> t+1；
        - sample_traj[:, :, 1] 预测 t+1 -> t+2；
        - sample_traj[:, :, 2] 预测 t+2 -> t+3；
        - ...
        多次输出的 pose_pred 再拼成完整未来轨迹。

        因此本函数内部的核心是“单步候选轨迹规划”：
        当前时间步的候选轨迹 -> cost 选择最优候选 -> transformer refine -> 输出一步 pose_pred。

        整体流程：
        1. 根据高层 command 从候选轨迹中筛选 left / forward / right 对应候选；
        2. 将当前 BEV 特征转换到 planning grid，并预测 cost volume；
        3. 从 semantic occupancy 中提取动态障碍物区域和可行驶区域；
        4. 用 occupancy cost + learned cost volume 给候选轨迹打分；
        5. 选择总代价最低的候选轨迹作为 coarse plan；
        6. 将 coarse plan 编码成 pose feature，与 plan query / command / BEV 特征一起送入
           PlanTransformer，进一步 refine；
        7. reg_branch 输出下一步自车位移 next_pose。

        Args:
            bev_feats: bev feats of current frame, with shape of (bs, bev_h * bev_w, embed_dim)
            trajs:    bs, sample_num, 3     current -> next frmae, under ref_lidar
            gt_trajs: bs, 2                 current -> next frame, under ref_lidar
            sem_occ:  bs, H,W,D             semantic occupancy
            command: bs                    0:Right  1:Left  2:Forward
            gt_trajs: bs, 3                 current -> next frame, under ref_liar
        """
        # *===============================================================
        # * 先选取候选轨迹，再基于候选轨迹预测/refine下一步轨迹
        #* ================== 1. 按 command 筛选候选轨迹 ==================
        # Dataset 生成的 trajs 按三组排列：
        #   [0:self.num]                 -> Left
        #   [self.num:self.num * 2]      -> Forward
        #   [self.num * 2:self.sample_num] -> Right
        # command: 0=Right, 1=Left, 2=Forward。
        #
        # 这里不是让网络预测候选轨迹，而是先根据 command 取对应方向的候选集合。
        # repeat(3, 1) 是为了把单方向的 self.num 条候选扩回 sample_num 条，
        # 保持后续 cost/select 代码输入 shape 仍为 [B, sample_num, 3]。
        cur_trajs = []
        for i in range(len(command)):
            command_i = command[i]
            traj = trajs[i]
            if command_i == 1:    # Left
                cur_trajs.append(traj[:self.num].repeat(3, 1))
            elif command_i == 2:  # Forward
                cur_trajs.append(traj[self.num:self.num * 2].repeat(3, 1))
            elif command_i == 0:  # Right
                cur_trajs.append(traj[self.num * 2:].repeat(3, 1))
            else:
                cur_trajs.append(traj)
        cur_trajs = torch.stack(cur_trajs)  # B,N_sample,3

        #* ================== 2. 当前 BEV 特征 -> planning grid BEV 特征 ==================
        # 输入 bev_feats 来自 world model / BEV encoder，shape 为 [B, H*W, C]。
        # 先 reshape 成 [B, C, H, W]，再用 bev_sampler 映射到 plan_grid_conf 指定的
        # planning grid 上，方便 cost volume 和轨迹采样在同一 BEV 坐标系中计算。
        bev_feats = rearrange(bev_feats, 'b (w h) c -> b c h w', h=self.bev_h, w=self.bev_w)
        bev_feats = self.bev_sampler(bev_feats)

        # 可选 BEV adapter：轻量残差模块，让 BEV 特征进一步适配 planning/cost 任务。
        if self.with_adapter:
            bev_feats = bev_feats + self.bev_adapter(bev_feats)  # residual connection

        #* ================== 3. 从 BEV 特征预测可学习 cost volume ==================
        # costvolume: [B, H_plan, W_plan]。
        #
        # 这里的 costvolume 是“学习型代价”，不是直接由语义 occupancy 规则计算出来的。
        # 它的作用是补充 sem_occupancy 难以显式表达的软约束/隐式驾驶偏好，例如：
        # - 两条候选轨迹都不碰撞，但其中一条离障碍物太近；
        # - 都在可行驶区域内，但其中一条更符合道路结构或更自然；
        # - occupancy 是离散类别图，可能丢失 BEV latent feature 中的细粒度上下文；
        # - 网络可以从数据中学习“哪些区域虽然可通行，但代价更高”。
        #
        # 后续 Cost_Volume 会沿每条候选轨迹在该 cost map 上采样，
        # 得到每条候选轨迹的 learned trajectory cost。
        # 若只想做纯 occupancy-based planner，可以在 Cost_Function 中去掉这一项。
        costvolume = self.costvolume_head(bev_feats).squeeze(1) # b,h,w

        #* ================== 4. 从 semantic occupancy 提取手工规则需要的 BEV mask ==================
        # sem_occupancy: [B, H, W, D]，包含每个 BEV 网格在高度维度上的语义类别。
        #
        # 与上面的 costvolume 不同，这里从 sem_occupancy 得到的是“显式规则代价”的输入：
        # - instance_occupancy 用于 safety/headway cost，判断候选轨迹是否碰撞动态目标、
        #   或者前方安全距离内是否存在障碍物；
        # - drivable_area 用于 rule cost，判断候选轨迹是否驶出可行驶区域。
        #
        # 因此 PlanHead_v1 的总代价是：
        #   occupancy/map 规则代价  +  BEV learned cost
        # 前者提供明确的安全/规则约束，后者补充数据驱动的隐式偏好。
        # instance_occupancy: 将动态障碍物类别压到 BEV 平面，得到 [B, H, W]。
        # 只作为 cost 计算的输入，不需要梯度，因此 detach。
        #=================================================================
        instance_occupancy = torch.isin(sem_occupancy, self.instance_cls.to(sem_occupancy)).float() # 判断每个voxel语义类别是否是动态障碍物类别
        instance_occupancy = instance_occupancy.max(-1)[0].detach()  # b,h,w # 某个bev网格柱子上的任意高度存在动态目标，即认为该BEV网格被动态障碍物占据


        #=================================================================
        # 只要某个 BEV 网格柱子中任意高度 voxel 被标成 driveable_surface，这个 BEV 网格就认为是可行驶区域。
        # drivable_area: 将可行驶区域类别压到 BEV 平面，得到 [B, H, W]。
        # Rule cost 会用它惩罚驶出可行驶区域的候选轨迹。
        drivable_area = torch.isin(sem_occupancy, self.drivable_area_cls.to(sem_occupancy)).float()
        drivable_area = drivable_area.max(-1)[0].detach()   # b,h,w

        #* ================== 5. 训练时计算候选轨迹 cost ranking loss ==================
        # loss_cost 的核心思想：GT 轨迹的 cost 应该小于候选轨迹的 cost。
        # 推理时不计算 loss，只使用 cost 来选择最优候选。
        if self.training:
            loss = self.loss_cost(cur_trajs, gt_trajs, costvolume, instance_occupancy, drivable_area)
        else:
            loss = None

        #* ================== 6. 选择总代价最小的候选轨迹 ==================
        # select() 内部会调用 Cost_Function：
        #   total_cost = safety + headway + rule + learned_costvolume
        # 然后 topk(largest=False) 取 cost 最小的一条。
        select_traj = self.select(cur_trajs, costvolume, instance_occupancy, drivable_area)  # B,3

        # 将选中的 coarse trajectory 编码成 pose feature，作为 transformer 的 prev_pose 条件。
        select_traj = self.pose_encoder(select_traj.float()).unsqueeze(1)   # B,1,C

        #* ================== 7. 准备 PlanTransformer 输入 ==================
        # bev_feats 从 [B, C, H, W] 拉平成 [B, H*W, C]，作为 cross-attention 的 key/value。
        bs = bev_feats.shape[0]
        dtype = bev_feats.dtype
        bev_feats = rearrange(bev_feats, 'b c h w -> b (w h) c')

        # plan_query: 可学习的规划 query，表示“下一步自车规划”的查询 token。
        plan_query = self.plan_embedding.weight.to(dtype)
        plan_query = plan_query[None]

        # navi_embed: command 对应的高层导航指令 embedding。
        # command=0/1/2 分别对应 Right/Left/Forward。
        navi_embed = self.navi_embedding.weight[command]
        navi_embed = navi_embed[None]

        # 将 plan query 和导航指令融合，得到带 command 条件的 planning query。
        plan_query = torch.cat([plan_query, navi_embed], dim=-1)
        plan_query = self.mlp_fuser(plan_query)

        # BEV 位置编码，提供 planning grid 上的空间位置信息。
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=plan_query.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)  # bs, bev_dims, bev_h, bev_w

        #* ================== 8. Transformer refine ==================
        # PlanTransformer 使用：
        # - plan_query: 带 command 的规划查询；
        # - bev_feats: 当前 BEV 场景上下文；
        # - prev_pose: cost 最小候选轨迹编码；
        # - bev_pos: BEV 位置编码。
        # 输出 refined plan feature。
        plan_query = self.transformer(
            plan_query,
            bev_feats,
            prev_pose=select_traj,
            bev_pos=bev_pos,
        )

        #* ================== 9. 回归下一步自车位移 ==================
        # 当前 planning_steps=1，因此输出 shape 为 [B, 1, 2]，
        # 表示 current -> next frame 的自车位移 [dx, dy]。
        # 多步未来规划由外层 future_pred() 自回归多次调用该 plan_head 实现。
        next_pose = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))   # B,mode=1,2
        return next_pose, loss

@HEADS.register_module()
class PlanHead_v2(BaseModule):
    """Head of Ego-Trajectory Planning.
    """

    def __init__(self,
                 # Architecture.
                 with_adapter=True,
                 transformer=None,
                 plan_grid_conf=None,

                 # positional encoding
                 bev_h=200,
                 bev_w=200,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),

                 # loss
                 loss_planning=None,
                 loss_collision=None,

                 *args,
                 **kwargs):

        # BEV configuration of reference frame.
        super().__init__(**kwargs)
        bevformer_bev_conf = {
            'xbound': [-51.2, 51.2, 0.512],
            'ybound': [-51.2, 51.2, 0.512],
            'zbound': [-10.0, 10.0, 20.0],
        }
        self.bev_sampler =  BevFeatureSlicer(bevformer_bev_conf, plan_grid_conf)

        # TODO: reimplement it with down-scaled feature_map
        self.embed_dims = transformer.embed_dims
        self.with_adapter = with_adapter
        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims // 2, self.embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)


        # build transformer architecture.
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.transformer = build_transformer(transformer)

        # build decoder
        self.planning_steps = 1
        self.reg_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.planning_steps * 2),
        )

        # loss
        self.loss_planning = build_loss(loss_planning)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = nn.ModuleList(self.loss_collision)

        self._init_layers()

    def _init_layers(self):
        """Initialize BEV prediction head."""
        # plan query for the next frame.
        self.plan_embedding = nn.Embedding(1, self.embed_dims)
        # navi embed.
        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        # mlp_fuser
        fuser_dim = 2
        self.mlp_fuser = nn.Sequential(
                nn.Linear(self.embed_dims*fuser_dim, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(inplace=True),
            )

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        try:
            self.transformer.init_weights()
            # Initialization of embeddings.
            normal_(self.plan_embedding)
            normal_(self.navi_embedding)
            xavier_init(self.mlp_fuser, distribution='uniform', bias=0.)
        except:
            pass

    def loss(self, outs_planning, sdc_planning, sdc_planning_mask, future_gt_bbox=None):
        """
            outs_planning:      B,Lout,mode=1,2
            sdc_planning:       B,Lout,3
            sdc_planning_mask:  B,Lout,2
            future_gt_bbox:     Lout*[N_box个bbox_3d]
        """
        loss_dict = dict()
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](outs_planning, sdc_planning[..., :3], torch.any(sdc_planning_mask, dim=-1), future_gt_bbox)
            loss_dict[f'loss_collision_{i}'] = loss_collision
        loss_ade = self.loss_planning(outs_planning, sdc_planning, torch.any(sdc_planning_mask, dim=-1))
        loss_dict.update(dict(loss_ade=loss_ade))
        return loss_dict

    @auto_fp16(apply_to=('bev_feats'))
    def forward(self, bev_feats, command):
        """ Forward function for each frame.

        Args:
            bev_feats: bev feats of current frame, with shape of (bs, bev_h * bev_w, embed_dim)
            command: bs                    0:Right  1:Left  2:Forward
        """
        # bev_feat
        # grid sample
        bev_feats = rearrange(bev_feats, 'b (w h) c -> b c h w', h=self.bev_h, w=self.bev_w)
        bev_feats = self.bev_sampler(bev_feats)
        # plugin adapter
        if self.with_adapter:
            bev_feats = bev_feats + self.bev_adapter(bev_feats)  # residual connection

        # bev refine
        bs = bev_feats.shape[0]
        dtype = bev_feats.dtype
        bev_feats = rearrange(bev_feats, 'b c h w -> b (w h) c')

        # # 1. plan_query
        # plan_query = select_traj
        plan_query = self.plan_embedding.weight.to(dtype)
        plan_query = plan_query[None]
        # navi_embed
        navi_embed = self.navi_embedding.weight[command]
        navi_embed = navi_embed[None]
        # mlp_fuser
        plan_query = torch.cat([plan_query, navi_embed], dim=-1)
        plan_query = self.mlp_fuser(plan_query)

        # 3. bev_feats
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=plan_query.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)  # bs, bev_dims, bev_h, bev_w

        # 5. do transformer layers to get pose features.
        plan_query = self.transformer(
            plan_query,
            bev_feats,
            bev_pos=bev_pos,
        )

        # 6. plan regression
        next_pose = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))   # B,mode=1,2
        return next_pose
