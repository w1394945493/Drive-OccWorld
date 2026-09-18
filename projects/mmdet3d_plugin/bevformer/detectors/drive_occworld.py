import mmcv
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
import copy
import numpy as np
import os
import torch.nn.functional as F
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.bevformer.losses.plan_reg_loss_lidar import plan_reg_loss
from projects.mmdet3d_plugin.bevformer.utils.metric_stp3 import PlanningMetric
from projects.mmdet3d_plugin.bevformer.utils.planning_metrics import PlanningMetric_v2
from torchvision.transforms.functional import rotate

from .bevformer import BEVFormer
from mmdet3d.models import builder
from ..utils import e2e_predictor_utils


@DETECTORS.register_module()
class Drive_OccWorld(BEVFormer):
    def __init__(self,
                 # Future predictions.
                 future_pred_head,
                 turn_on_flow,
                 future_pred_frame_num,  # number of future prediction frames.
                 test_future_frame_num,  # number of future prediction frames when testing.

                 # BEV configurations.
                 point_cloud_range,
                 bev_h,
                 bev_w,

                 # Plan Head configurations.
                 turn_on_plan=False,
                 plan_head=None,

                 # Memory Queue configurations.
                 memory_queue_len=1,
                 #* 显存优化实验开关，默认 0 表示保持原 Drive-OccWorld 自回归训练方式。
                 #* 若设置为 N>0，则训练时每 N 个 future step 将预测 BEV
                 #* 放回 memory_queue 前 detach，截断跨未来步反向传播链路。
                 future_bev_detach_interval=0,

                 # Augmentations.
                 # A1. randomly drop current image (to enhance temporal feature.)
                 random_drop_image_rate=0.0,
                 # A2. add noise to previous_bev_queue.
                 random_drop_prev_rate=0.0,
                 random_drop_prev_start_idx=1,
                 random_drop_prev_end_idx=None,
                 # A3. grid mask augmentation.
                 grid_mask_image=True,
                 grid_mask_backbone_feat=False,
                 grid_mask_fpn_feat=False,
                 grid_mask_prev=False,
                 grid_mask_cfg=dict(
                     use_h=True,
                     use_w=True,
                     rotate=1,
                     offset=False,
                     ratio=0.5,
                     mode=1,
                     prob=0.7
                 ),

                 # Supervision.
                 only_generate_dataset=False,
                 supervise_all_future=True,

                 _viz_pcd_flag=False,
                 _viz_pcd_path='dbg/pred_pcd',  # root/{prefix}

                 *args,
                 **kwargs,):

        super().__init__(*args, **kwargs)
        # occ head
        self.future_pred_head = builder.build_head(future_pred_head)
        # flow head
        self.turn_on_flow = turn_on_flow
        if self.turn_on_flow:
            future_pred_head_flow = future_pred_head
            future_pred_head_flow['num_classes'] = 3
            future_pred_head_flow['turn_on_flow'] = True
            future_pred_head_flow['prev_render_neck']['occ_flow'] = 'flow'
            self.future_pred_head_flow = builder.build_head(future_pred_head_flow)
            if self.future_pred_head.num_classes == 2:      # GMO
                self.vehicles_id = [1]
            elif self.future_pred_head.num_classes == 9:    # nus_MMO
                self.vehicles_id = [1,2,3,4,5,6,7,8]
            elif self.future_pred_head.num_classes == 17:   # nus_finegrained
                self.vehicles_id = [2,3,4,5,6,7,9,10]
            self.gmo_id = 1 # sem_clsses -> GMO
            self.iou_thresh_for_vpq = 0.2

        # plan head
        self.turn_on_plan = turn_on_plan
        if turn_on_plan:
            self.plan_head = builder.build_head(plan_head)
            self.plan_head_type = plan_head.type
            self.planning_metric = None
            self.planning_metric_v2 = PlanningMetric_v2(n_future=future_pred_frame_num+1)

        # memory queue
        self.memory_queue_len = memory_queue_len
        #* Truncated BPTT for autoregressive future BEV rollout.
        #* 默认 0 不做 detach，不影响原 nuScenes/Drive-OccWorld 行为；
        #* SemanticKITTI 显存优化实验可设置为 2，形成
        #* current -> t+1 -> t+2.detach() -> t+3 -> t+4.detach() 的局部反传链。
        self.future_bev_detach_interval = future_bev_detach_interval


        self.future_pred_frame_num = future_pred_frame_num
        self.test_future_frame_num = test_future_frame_num
        # if not predict any future,
        #  then only predict current frame.
        self.only_train_cur_frame = (future_pred_frame_num == 0)

        self.point_cloud_range = point_cloud_range
        self.bev_h = bev_h
        self.bev_w = bev_w

        # Augmentations.
        self.random_drop_image_rate = random_drop_image_rate
        self.random_drop_prev_rate = random_drop_prev_rate
        self.random_drop_prev_start_idx = random_drop_prev_start_idx
        self.random_drop_prev_end_idx = random_drop_prev_end_idx

        # Grid mask.
        self.grid_mask_image = grid_mask_image
        self.grid_mask_backbone_feat = grid_mask_backbone_feat
        self.grid_mask_fpn_feat = grid_mask_fpn_feat
        self.grid_mask_prev = grid_mask_prev
        self.grid_mask = GridMask(**grid_mask_cfg)

        # Training configurations.
        # randomly sample one future for loss computation?
        self.only_generate_dataset = only_generate_dataset
        self.supervise_all_future = supervise_all_future

        self._viz_pcd_flag = _viz_pcd_flag
        self._viz_pcd_path = _viz_pcd_path

        # remove the useless modules in pts_bbox_head
        #  * box/cls prediction head; decoder transformer.
        del self.pts_bbox_head.cls_branches, self.pts_bbox_head.reg_branches
        del self.pts_bbox_head.query_embedding
        del self.pts_bbox_head.transformer.decoder

        if self.only_train_cur_frame:
            # remove useless parameters.
            del self.future_pred_head.transformer
            del self.future_pred_head.bev_embedding
            del self.future_pred_head.prev_frame_embedding
            del self.future_pred_head.can_bus_mlp
            del self.future_pred_head.positional_encoding
            del self.future_pred_head_flow.transformer
            del self.future_pred_head_flow.bev_embedding
            del self.future_pred_head_flow.prev_frame_embedding
            del self.future_pred_head_flow.can_bus_mlp
            del self.future_pred_head_flow.positional_encoding

    def set_epoch(self, epoch):
        self.training_epoch = epoch

    ####################### Image Feature Extraction. #######################
    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        if ('aug_param' in img_metas[0] and
                img_metas[0]['aug_param'] is not None and
                img_metas[0]['aug_param']['CropResizeFlipImage_param'][-1] is True):
            img_feats = [torch.flip(x, dims=[-1, ]) for x in img_feats]

        return img_feats

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            # 原实现：
            # if img.dim() == 5 and img.size(0) == 1:
            #     img.squeeze_()
            # elif img.dim() == 5 and img.size(0) > 1:
            #     B, N, C, H, W = img.size()
            #     img = img.reshape(B * N, C, H, W)
            #
            # 修改原因：
            # 原 nuScenes 输入通常是 B=1、N_cam=6，squeeze_() 后仍是
            # [6, C, H, W]，可以送入 ResNet。但 SemanticKITTI 第一阶段
            # 单目输入是 B=1、N_cam=1，squeeze_() 会同时删掉 batch 和
            # camera 两个维度，变成 [C, H, W]，导致 ResNet/BatchNorm 报
            # “expected 4D input”。因此这里对所有 5D 图像统一 reshape 成
            # [B*N_cam, C, H, W]，既兼容 nuScenes 多相机，也兼容 KITTI 单目。
            if img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask and self.grid_mask_image:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            if self.use_grid_mask and self.grid_mask_backbone_feat:
                new_img_feats = []
                for img_feat in img_feats:
                    img_feat = self.grid_mask(img_feat)
                    new_img_feats.append(img_feat)
                img_feats = new_img_feats
        else:
            return None

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            if self.use_grid_mask and self.grid_mask_fpn_feat:
                new_img_feats = []
                for img_feat in img_feats:
                    img_feat = self.grid_mask(img_feat)
                    new_img_feats.append(img_feat)
                img_feats = new_img_feats

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped


    ############# Align coordinates between reference (current frame) to other frames. #############
    def _get_history_ref_to_previous_transform(self, tensor, num_frames, prev_img_metas, ref_img_metas):
        """Get transformation matrix from reference frame to all previous frames.

        Args:
            tensor: to convert {ref_to_prev_transform} to device and dtype.
            num_frames: total num of available history frames.
            img_metas_list: a list of batch_size items.
                In each item, there is {num_prev_frames} img_meta for transformation alignment.

        Return:
            ref_to_history_list (torch.Tensor): with shape as [bs, num_prev_frames, 4, 4]
        """
        ref_num_frames = 1  # 当前参考帧数量固定为 1，即 memory queue 最后会包含当前 ref_bev。
        history_num_frames = num_frames - ref_num_frames  # history BEV 数量；num_frames 通常等于 memory_queue_len。memory_queue_len=1时，实际没有用到历史帧

        # history
        ref_to_history_list = []  # 收集 batch 内每个样本的 ref/current -> history BEV 坐标变换。
        for img_metas in prev_img_metas:  # 遍历 batch；img_metas 是该样本所有历史帧 meta 列表。
            img_metas_len = len(img_metas)  # 该样本可用历史帧数量，通常等于输入历史帧数。
            # 只取最后 history_num_frames 个历史帧，和 prev_bev_input 中保留的历史 BEV 对齐。
            # 例：history_num_frames=3 时对应 [-3,-2,-1]；history_num_frames=0 时为空列表。
            #* 注意字段名 ref_lidar_to_cur_lidar 中的 cur_lidar 指“该 meta 对应帧”，
            #* 在这里该 meta 来自历史帧，因此变换方向是：当前参考帧 -> 历史帧。
            cur_ref_to_prev = [img_metas[i]['ref_lidar_to_cur_lidar']  # 4x4: 当前参考帧 ref/current lidar -> 该历史帧 lidar。
                               for i in range(img_metas_len - history_num_frames, img_metas_len)]
            ref_to_history_list.append(cur_ref_to_prev)  # 当前样本追加到 batch list 中。
        ref_to_history_list = tensor.new_tensor(np.array(ref_to_history_list))  # shape: [B, history_num_frames, 4, 4]；可能第二维为 0。

        # ref 当前参考帧 ref -> 当前参考帧 ref
        ref_to_ref_list = []  # 收集 ref/current -> ref/current 的变换，一般是 identity。(应是单位矩阵)
        for img_metas in ref_img_metas:  # 遍历 batch；ref_img_metas 每个样本只包含当前参考帧 meta。
            cur_ref_to_prev = [img_metas[i]['ref_lidar_to_cur_lidar']  # 4x4: ref/current lidar -> ref/current lidar，通常为单位阵。
                               for i in range(ref_num_frames)]  # 当前参考帧自身的变换列表，长度 ref_num_frames=1。
            ref_to_ref_list.append(cur_ref_to_prev)  # 当前样本追加到 batch list 中。
        ref_to_ref_list = tensor.new_tensor(np.array(ref_to_ref_list))  # shape: [B, 1, 4, 4]；device/dtype 跟随 tensor。


        # concat
        if ref_to_history_list.shape[1] == 0:   # not use history；memory_queue_len=1 时 history_num_frames=0。
            ref_to_history_list = ref_to_ref_list  # 只使用当前 ref_bev，最终 shape [B, 1, 4, 4]。
        else:
            ref_to_history_list = torch.cat([ref_to_history_list, ref_to_ref_list], dim=1)  # [history transforms, ref transform]，shape [B, memory_queue_len, 4, 4]。
        return ref_to_history_list  # 返回 ref/current -> memory queue 各帧 lidar 坐标系的 4x4 变换。

    def _align_bev_coordnates(self, frame_idx, ref_to_history_list, img_metas, plan_traj):
        """Align the bev_coordinates of frame_idx to each of history_frames.

        Args:
            frame_idx: the index of target future frame.
            ref_to_history_list (torch.Tensor): a tensor with shape as [bs, num_prev_frames, 4, 4]
                indicating the transformation matrix from reference/current frame to each BEV memory frame.
            img_metas: a list of batch_size items.
                In each item, there is one img_meta (reference frame)
                whose {future2ref_lidar_transform} & {ref2future_lidar_transform} are for
                transformation alignment.
            plan_traj: accumulated or step-wise ego trajectory condition used by Drive-OccWorld.

        #* 核心目的：
        #*   预测未来第 frame_idx 帧 BEV 时，future BEV query 位于未来自车坐标系；
        #*   prev_bev_input/memory_queue 中的 BEV 位于历史/当前各自坐标系。
        #*   因此这里要把 future BEV 网格点变换到每个 memory BEV 坐标系，
        #*   得到 aligned_bev_grids，供 WorldDecoder cross-attention/deformable attention
        #*   在历史/当前 BEV memory 的正确位置采样特征。
        """
        bs, num_frame = ref_to_history_list.shape[:2]  # bs: batch size；num_frame: memory queue 中 BEV 帧数。
        translation_xy = torch.cumsum(plan_traj, dim=1)[:, -1, :2].float()  # 当前参考帧到目标未来帧的累计 x/y 位移。

        # 1. get future2ref and ref2future_matrix of frame_idx.
        # future2ref: 目标未来帧 lidar 坐标 -> 当前参考帧 ref lidar 坐标。
        # ref2future: 当前参考帧 ref lidar 坐标 -> 目标未来帧 lidar 坐标。
        #
        #* 重要：这里读取到的 4x4 矩阵使用 Drive-OccWorld 原始约定：
        #*   row-vector 右乘形式，即 [x, y, z, 1] @ transform。
        #*
        #* 常见 4x4 位姿矩阵有两种写法：
        #*   A. column-vector 左乘格式，常见于机器人/SLAM/标准几何推导：
        #*      p_dst = T @ [x, y, z, 1]^T，平移在最后一列 T[:3, 3]。
        #*      示例：
        #*        [[r11, r12, r13, tx],
        #*         [r21, r22, r23, ty],
        #*         [r31, r32, r33, tz],
        #*         [0,   0,   0,   1 ]]
        #*   B. row-vector 右乘格式，本函数/原 Drive-OccWorld 使用：
        #*      p_dst = [x, y, z, 1] @ T，平移在最后一行 T[3, :3]。
        #*      示例：
        #*        [[r11, r21, r31, 0],
        #*         [r12, r22, r32, 0],
        #*         [r13, r23, r33, 0],
        #*         [tx,  ty,  tz,  1]]
        #*
        #* 如何判断是右乘格式：
        #*   1) 看实际使用代码：下面是 aligned_bev_coords @ future_to_history_list，
        #*      点坐标在左、矩阵在右，所以是 row-vector 右乘；
        #*   2) 看平移项位置：row-vector 矩阵的平移在最后一行 transform[3, :3]，
        #*      column-vector 矩阵的平移在最后一列 transform[:3, 3]；
        #*   3) 看这里覆盖 plan_traj 时会先 transpose，再写 [:2, 3]，
        #*      说明原矩阵中平移实际位于转置前的 [3, :2]，即 row-vector 格式。
        #* 如何判断是左乘格式：
        #*   1) 看实际使用代码：如果写成 transform @ point，或者 batch 中写成
        #*      transform @ points[..., None]，点坐标在右、矩阵在左，就是 column-vector 左乘；
        #*   2) 看平移项位置：标准左乘矩阵直接用 transform[:3, 3] 作为 tx/ty/tz；
        #*   3) 看矩阵组合顺序：column-vector 中连续变换通常写成 T_dst_mid @ T_mid_src @ p_src，
        #*      而 row-vector 中则通常写成 p_src @ T_src_mid @ T_mid_dst。
        #* 原 nuScenes Dataset 在构造这些矩阵时会通过 .T 转成 row-vector 格式；
        #* SemanticKITTI Dataset 适配层也需要做同样转换。
        #* 因此 img_meta['future2ref_lidar_transform'][frame_idx] 虽然语义上是
        #* future -> ref，但实际存储的是可被 row-vector 右乘直接使用的矩阵。
        #
        # 原 nuScenes/离线 SemanticKITTI Dataset 会在当前参考帧 img_meta 中预先写入这些未来位姿变换。
        future2ref = [img_meta['future2ref_lidar_transform'][frame_idx] for img_meta in img_metas]
        future2ref = ref_to_history_list.new_tensor(np.array(future2ref))  # shape: [B, 4, 4]，device/dtype 跟随 ref_to_history_list。

        # use translation_xy
        if self.future_pred_head.use_plan_traj:
            # 如果启用 plan_traj，则用规划/GT 轨迹里的累计 x/y 位移覆盖 future2ref 的平移部分。
            # 注意这里先 transpose，是因为当前代码里的 4x4 矩阵按 row-vector 形式使用：
            #   aligned_bev_coords = aligned_bev_coords @ future_to_history_list
            # 因此平移项实际位于 transpose 后的 [:2, 3]。
            future2ref = future2ref.transpose(-1, -2)
            future2ref[:, :2, 3] = translation_xy  # 用 plan_traj 累计位移替代 Dataset 中的 x/y 平移。
            future2ref = future2ref.transpose(-1, -2)
            future2ref = future2ref.detach().clone()  # 这里作为几何对齐条件使用，不让梯度回传到 plan_traj。

        # ref2future: 当前参考帧 ref lidar 坐标 -> 目标未来帧 lidar 坐标。
        #* 同样是 row-vector 右乘格式，主要返回给外部，用于更新 ref_to_history_list。
        #* 自回归预测 t+1/t+2/... 时，当前预测出的 future BEV 会被加入 memory queue；
        #* ref_to_history_list 也要同步追加 ref -> future 的相对位姿，
        #* 这样下一步预测更远未来 BEV 时，才能把新 memory BEV 和目标 future query 对齐。
        ref2future = [img_meta['ref2future_lidar_transform'][frame_idx] for img_meta in img_metas]
        ref2future = ref_to_history_list.new_tensor(np.array(ref2future))  # shape: [B, 4, 4]。

        # use translation_xy
        if self.future_pred_head.use_plan_traj:
            # future2ref 平移被 plan_traj 覆盖后，ref2future 需要同步更新为其逆向平移。
            # 这里保持原实现的 row-vector 矩阵约定，先 transpose 后修改平移项。
            ref2future = ref2future.transpose(-1, -2)
            rot = ref2future[:, :3, :3]  # ref -> future 的旋转部分。
            translation_xyz = future2ref[:, 3, :3].unsqueeze(2)     # future -> ref 的平移，记作 cur2ref。
            translation_xyz = -(rot @ translation_xyz).squeeze(2)   # 根据逆变换关系得到 ref -> future 平移。
            ref2future[:, :3, 3] = translation_xyz  # 写回 ref2future 平移项。
            ref2future = ref2future.transpose(-1, -2)
            ref2future = ref2future.detach().clone()  # 几何条件不参与反传。

        # 2. compute the transformation matrix from current frame to all previous frames.
        # 将 future -> ref 扩展到每个 memory BEV 帧。
        future2ref = future2ref.unsqueeze(1).repeat(1, num_frame, 1, 1).contiguous()  # [B, num_frame, 4, 4]。
        # 组合得到 future -> history/current memory：
        #   future -> ref/current  再  ref/current -> each memory frame。
        # future_to_history_list: [B, num_frame, 4, 4]。
        future_to_history_list = torch.matmul(future2ref, ref_to_history_list)

        # 3. compute coordinates of future frame.
        # 生成目标未来 BEV 的标准 grid，范围通常是 [-1, 1]，shape 初始为 [B*num_frame, H*W, 2]。
        bev_grids = e2e_predictor_utils.get_bev_grids(self.bev_h, self.bev_w, bs * num_frame)
        bev_grids = bev_grids.view(bs, num_frame, -1, 2)  # [B, num_frame, H*W, 2]。
        # 将归一化 grid 坐标转成实际 BEV/点云坐标，例如米制 x/y 坐标。
        bev_coords = e2e_predictor_utils.bev_grids_to_coordinates(bev_grids, self.point_cloud_range)

        # 4. align target coordinates of future frame to each of previous frames.
        # 给 BEV x/y 坐标补齐 z/齐次坐标，变成 [x, y, 1, 1] 风格，便于乘 4x4 矩阵。
        # 注意这里保持原实现：用 ones_like(bev_coords[..., :2]) 一次补两个维度。
        aligned_bev_coords = torch.cat([bev_coords, torch.ones_like(bev_coords[..., :2])], -1)
        # 将 future BEV 中每个 query 的实际坐标变换到每个 memory BEV 坐标系。
        # aligned_bev_coords: [B, num_frame, H*W, 4] @ [B, num_frame, 4, 4]
        #                   -> [B, num_frame, H*W, 4]
        aligned_bev_coords = torch.matmul(aligned_bev_coords, future_to_history_list)
        aligned_bev_coords = aligned_bev_coords[..., :2]  # 只保留变换后的 x/y 坐标。
        # 将实际 x/y 坐标再转回 BEV grid 坐标，便于 attention/grid_sample 使用。
        aligned_bev_grids, _ = e2e_predictor_utils.bev_coords_to_grids(
            aligned_bev_coords, self.bev_h, self.bev_w, self.point_cloud_range)
        aligned_bev_grids = (aligned_bev_grids + 1) / 2.  # 原 grid 范围 [-1,1]，这里转成 [0,1]。
        # b, h*w, num_frame, 2
        # WorldDecoder 期望 ref_points shape 类似 [B, H*W, num_frame, 2]：
        # 对每个 future query，给出它在每个 memory BEV 中应采样的位置。
        aligned_bev_grids = aligned_bev_grids.permute(0, 2, 1, 3).contiguous()

        # 5. get target bev_grids at target future frame.
        # tgt_grids 是目标 future BEV 自身的 query grid，shape [B, H*W, 2]。
        # 由于每个 memory 帧使用同一套 target future grid，这里取最后一个 num_frame 位置即可。
        tgt_grids = bev_grids[:, -1].contiguous()
        # 返回：
        # - tgt_grids: future BEV query 自身坐标，shape [B, H*W, 2]；
        # - aligned_bev_grids: future query 对齐到各 memory BEV 后的采样坐标，
        #   shape [B, H*W, memory_queue_len, 2]，数值范围约为 [0,1]；
        # - ref2future: ref/current -> future，用于更新下一步 memory 的位姿关系，
        #   shape [B, 4, 4]；
        # - future_to_history_list.transpose: 按后续 ConditionalNorm/ego-motion norm 约定返回，
        #   shape [B, memory_queue_len, 4, 4]。
        return tgt_grids, aligned_bev_grids, ref2future, future_to_history_list.transpose(-1, -2)


    def obtain_ref_bev(self, img, img_metas, prev_bev):
        # Extract current BEV features.
        # C1. Forward.
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        # C3. BEVFormer Encoder Forward.
        # ref_bev: bs, bev_h * bev_w, c
        ref_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)
        return ref_bev

    def obtain_ref_bev_with_plan(self, img, img_metas, prev_bev, ref_sample_traj, ref_sem_occupancy, ref_command, ref_real_traj=None):
        # Extract current BEV features.
        # C1. Forward.
        img_feats = self.extract_feat(img=img, img_metas=img_metas) # 4:(1 6 256 116 200) (1 6 256 58 100) (1 6 256 29 50) (1 6 256 15 25)
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None

        # C2. BEVFormer Encoder Forward.
        # ref_bev: bs, bev_h * bev_w, c
        ref_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True) # (1 40000 256)

        # C3. PlanHead
        if 'v1' in self.plan_head_type: # PlanHead_v1
            if ref_sem_occupancy is None:   # use pred_occupancy to calculate sample_traj cost during inference, GT_occupancy during training
                ref_sem_occupancy = self.future_pred_head.forward_head(ref_bev.unsqueeze(0).unsqueeze(0))[-1, -1, 0].argmax(-1).detach() # (1 40000 16)
                bs, hw, d = ref_sem_occupancy.shape
                ref_sem_occupancy = ref_sem_occupancy.view(bs, self.bev_w, self.bev_h, d).transpose(1,2) # (1 200 200 16)

            # *================================================#
            #* 规划Head
            ref_pose_pred, ref_pose_loss = self.plan_head(ref_bev, ref_sample_traj, ref_sem_occupancy, ref_command, ref_real_traj)
        elif 'v2' in self.plan_head_type:
            ref_pose_pred = self.plan_head(ref_bev, ref_command)
            ref_pose_loss = None

        return ref_bev, ref_pose_pred, ref_pose_loss


    def future_pred(self, prev_bev_input, action_condition_dict, cond_norm_dict, plan_dict,
                    valid_frames, img_metas, prev_img_metas, num_frames, occ_flow='occ'):
        #* ================== 自回归未来预测总入口 ==================
        # 这个函数同时负责两条自回归链路：
        # 1. BEV/Occupancy 链路：
        #    当前 memory_queue -> 预测 future_frame_index 的 BEV -> 放回 memory_queue，
        #    再继续预测更远未来。
        # 2. Planning 链路（仅 turn_on_plan=True 且 occ_flow='occ'）：
        #    single_test()/forward_train 中已经先用 sample_traj[:, :, 0] 预测了
        #    ref -> t+1 的第一步 ref_pose_pred；
        #    这里从 future_frame_index=1 开始，继续用 sample_traj[:, :, i]
        #    逐步预测 t+i -> t+i+1 的后续一步轨迹，并 cat 到 next_pose_preds。
        #
        # 因此 next_pose_preds 的构建方式是：
        #   初始: [ref_pose_pred]，对应 sample_traj[:, :, 0]
        #   循环: append pose_pred_i，对应 sample_traj[:, :, i], i=1..future_frame_num
        # 最终得到多步自车轨迹预测。
        if occ_flow == 'occ':
            future_pred_head = self.future_pred_head
        elif occ_flow == 'flow':
            future_pred_head = self.future_pred_head_flow
        else:
            AssertionError('Not Implemented')

        #* ================== 4.1 准备未来预测输入 ==================
        # prev_bev_input 是 memory_queue，最后一帧是当前参考 BEV；后续会被逐帧预测结果滚动更新。
        # prev_bev_input: B,memory_queue_len,HW,C
        ref_bev = prev_bev_input[:, -1].unsqueeze(0).repeat(
                len(self.future_pred_head.bev_pred_head), 1, 1, 1).contiguous() # (3 1 65536 256) 65536=256x256

        next_bev_feats, next_bev_sem, next_pose_loss = [ref_bev], [], []
        # next_pose_preds 初始就是参考帧规划出的第一步轨迹：
        # ref_pose_pred: ref/current -> t+1，对应 sample_traj[:, :, 0]。
        # 后续 for future_frame_index in range(1, ...) 会继续 append t+1->t+2,
        # t+2->t+3, ... 的逐步规划结果。
        next_pose_preds = plan_dict['ref_pose_pred'] # B,Lout,2 None


        #* ================== 4.2 建立历史帧到参考帧的坐标变换 ==================
        # 后续每个 future query 都要根据 future pose/action condition 对齐到历史 BEV memory。
        ref_img_metas = [[each[num_frames-1]] for each in prev_img_metas] # 取当前帧meta 这里将历史帧+当前帧拆为历史帧、当前帧
        prev_img_metas = [[each[i] for i in range(num_frames-1)] for each in prev_img_metas] # 取历史帧meta prev_img_metas一开始包含历史+当前
        # ** (1) 收集当前帧到历史/当前 memory 帧的坐标变换；得到 aligned_bev_grids，供 WorldDecoder cross-attention/deformable attention
        ref_to_history_list = self._get_history_ref_to_previous_transform(
            prev_bev_input, prev_bev_input.shape[1], prev_img_metas, ref_img_metas)


        #* ================== 4.3 自回归预测未来 BEV feature / future pose ==================
        # 每一步预测一个 future frame，并把该预测 BEV 放回 memory_queue，继续预测更远未来。
        # 如果开启 planner，则每一步还会基于该未来 BEV/Occupancy 预测下一段自车位移。
        if self.training:
            future_frame_num = self.future_pred_frame_num
        else:
            future_frame_num = self.test_future_frame_num

        for future_frame_index in range(1, future_frame_num + 1):
            #* future_frame_index 的含义：
            # - 在 single_test()/forward_train 外层，已经用 sample_traj[:, :, 0]
            #   预测了 ref/current -> t+1 的 ref_pose_pred；
            # - 这里从 1 开始循环：
            #     future_frame_index=1 使用 sample_traj[:, :, 1]，预测 t+1 -> t+2；
            #     future_frame_index=2 使用 sample_traj[:, :, 2]，预测 t+2 -> t+3；
            #     ...
            # 同时 WorldHead 也用相同的 future_frame_index 预测对应未来时刻的 BEV/Occ。
            if (not self.turn_on_plan) or (self.turn_on_plan and self.training and self.training_epoch < 12):   # use GT planning during training
                plan_traj = plan_dict['gt_traj'][:, :future_frame_index, :2] # (1 1 2)
            else:
                plan_traj = next_pose_preds
            action_condition_dict['plan_traj'] = plan_traj # (1 1 2)
            #* can_bus/action condition 关键链路：
            #* future_frame_index 会作为 target_frame_index 传入 WorldHeadBase，
            #* 之后按这个时间步读取 img_meta['future_can_bus'][future_frame_index]，
            #* 把 can_bus 当作 action condition，用来控制该未来帧的 occupancy 预测。

            # ====================================================================#
            # ** (2) 再结合未来运动，把未来 BEV query 对齐到 memory BEV；
            # * 把 future BEV 网格点变换到每个 memory BEV 坐标系，
            tgt_grids, aligned_prev_grids, ref2future, future2history = self._align_bev_coordnates(
                future_frame_index, ref_to_history_list, img_metas, plan_traj)
            cond_norm_dict['future2history'] = future2history


            # * =========================================================
            # * 预测未来帧BEV特征
            # 2. transform for generating freespace of future frame.
            #* tgt_grids 和 aligned_prev_grids 的区别：
            #* - tgt_grids 是“要生成的未来 BEV query 自己在哪里”；
            #* - aligned_prev_grids 是“这些未来 query 应该去历史/当前 BEV memory 的哪里采样特征”。
            # pred_feat: inter_num, bs, bev_h * bev_w, c
            if future_frame_index in valid_frames:  # compute loss if it is a valid frame.
                pred_feat, bev_sem_pred = future_pred_head(
                    prev_bev_input,  # memory queue BEV 特征，[B, memory_queue_len, H*W, C]。
                    img_metas,  # 当前参考帧 meta；提供 future_can_bus / 位姿等条件信息。
                    future_frame_index,  # 当前正在预测的未来步编号，1 表示 t+1，2 表示 t+2。
                    action_condition_dict,  # 动作条件字典，如 command、vel_steering、plan_traj。
                    cond_norm_dict,  # 条件归一化字典，如 future2history / occ_gts。
                    tgt_points=tgt_grids,  # 目标坐标：future BEV query 在未来帧自身 BEV 坐标系中的规则网格位置，[B, H*W, 2]。
                    bev_h=self.bev_h,  # BEV 网格高度 H。
                    bev_w=self.bev_w,  # BEV 网格宽度 W。
                    ref_points=aligned_prev_grids)  # 采样坐标：同一批 future query 映射到各 memory BEV 后的位置，用于从 prev_bev_input 取特征，[B, H*W, memory_queue_len, 2]。

                next_bev_feats.append(pred_feat)  # 保存当前 future step 的 BEV feature；pred_feat shape [inter_num, B, H*W, C]。
                next_bev_sem.append(bev_sem_pred)  # 保存 sem_norm/render 分支输出，用于后续 loss_sem_norm。
            else:
                with torch.no_grad():
                    pred_feat, bev_sem_pred = future_pred_head(
                        prev_bev_input,  # memory queue BEV 特征，[B, memory_queue_len, H*W, C]。
                        img_metas,  # 当前参考帧 meta；提供 future_can_bus / 位姿等条件信息。
                        future_frame_index,  # 当前正在预测的未来步编号。
                        action_condition_dict,  # 动作条件字典，如 command、vel_steering、plan_traj。
                        cond_norm_dict,  # 条件归一化字典，如 future2history / occ_gts。
                        tgt_points=tgt_grids,  # 目标坐标：future BEV query 在未来帧自身 BEV 坐标系中的规则网格位置，[B, H*W, 2]。
                        bev_h=self.bev_h,  # BEV 网格高度 H。
                        bev_w=self.bev_w,  # BEV 网格宽度 W。
                        ref_points=aligned_prev_grids)  # 采样坐标：同一批 future query 映射到各 memory BEV 后的位置，用于从 prev_bev_input 取特征，[B, H*W, memory_queue_len, 2]。
                    next_bev_feats.append(pred_feat)  # 非监督帧只用于自回归 rollout，不保留梯度。


            # 3. Planning based on semantic occupancy.
            if self.turn_on_plan and occ_flow == 'occ':
                #* ================== 4.4 当前未来步的 planning ==================
                # 当前 future_frame_index 的轨迹预测遵循同一套“候选轨迹 -> cost 选择 -> transformer refine”流程：
                #
                #   sample_traj[:, :, future_frame_index]       # 当前未来步的候选一步轨迹
                #       ↓
                #   sem_occupancy_i / pred_feat[-1]             # 当前未来步的占用预测和 BEV 特征
                #       ↓
                #   PlanHead_v1 根据 command_i 筛选 left/forward/right 候选组
                #       ↓
                #   Cost_Function 计算候选轨迹代价：
                #       safety + headway + rule + learned_costvolume
                #       ↓
                #   select() 选代价最小候选轨迹
                #       ↓
                #   PlanTransformer 融合 BEV feature / command / selected trajectory
                #       ↓
                #   reg_branch 输出当前未来步 refined pose_pred
                #
                # 多个 future_frame_index 的 pose_pred 会被 cat 到 next_pose_preds，
                # 从而形成多步未来自车轨迹。
                # sample_traj_i: 当前 future_frame_index 对应的一步候选轨迹，
                #   shape [B, sample_num, 3]。
                # gt_traj_i: 同一个未来步的一步 GT 位移，用于训练 loss。
                # 例：sample_traj shape 为 [B, 1800, 5, 3] 时：
                #   sample_traj[:, :, 0] 已在外层预测 ref -> t+1；
                #   sample_traj[:, :, 1] 在这里预测 t+1 -> t+2；
                #   sample_traj[:, :, 2] 在这里预测 t+2 -> t+3；
                #   ...
                sample_traj_i,  gt_traj_i = plan_dict['sample_traj'][:,:,future_frame_index], plan_dict['gt_traj'][:,future_frame_index]
                # command_i: 当前未来步的高层驾驶指令，控制 left/right/forward 候选组选择。
                command_i = action_condition_dict['command'][:,future_frame_index]

                # forward plan_head：基于当前预测的未来 BEV/Occupancy，
                # 对 sample_traj_i 中的候选轨迹打分、选最优候选，并 refine 出 pose_pred。
                if 'v1' in self.plan_head_type:    # used for fine-grained_MMO when sem_occupancy distinguish categories in MMO
                    # sem_occupancy_i:
                    # - 测试/推理时从当前 future BEV 的 occupancy head 预测得到；
                    # - 训练前期可使用 GT occupancy 稳定 planner。
                    if plan_dict['sem_occupancy'] is None:   # use_pred
                        sem_occupancy_i = future_pred_head.forward_head(pred_feat.unsqueeze(0))[-1, -1, 0].argmax(-1).detach()
                        bs, hw, d = sem_occupancy_i.shape
                        sem_occupancy_i = sem_occupancy_i.view(bs, self.bev_w, self.bev_h, d).transpose(1,2)
                    else:   # use_gt  traning_epoch < 12
                        sem_occupancy_i = plan_dict['sem_occupancy'][:,future_frame_index]
                    pose_pred, pose_loss = self.plan_head(pred_feat[-1], sample_traj_i, sem_occupancy_i, command_i, gt_traj_i)
                    # update prev_pose and store pred:
                    # 将当前一步 pose_pred 追加到 next_pose_preds，
                    # 后续更远未来的 BEV 对齐/规划会使用已经预测出的多步 plan_traj。
                    next_pose_preds = torch.cat([next_pose_preds, pose_pred], dim=1)
                    next_pose_loss.append(pose_loss)
                elif 'v2' in self.plan_head_type:   # used for inflated_GMO when sem_occupancy does not distinguish categories in GMO
                    pose_pred = self.plan_head(pred_feat[-1], command_i)
                    next_pose_preds = torch.cat([next_pose_preds, pose_pred], dim=1)


            # 4. update pred_feat to prev_bev_input and update ref_to_history_list.
            memory_feat = pred_feat[-1]

            # *===============================================================================#
            #* 可选截断自回归 future BEV 的跨步梯度：detach 当前 step 的 BEV 后再作为下一步 memory。
            #* 例如 interval=2 时，t+2 自身 loss 仍可回传至更早步骤，但 t+3/t+4 不会越过 t+2 回传。
            if (self.training and self.future_bev_detach_interval is not None
                    and self.future_bev_detach_interval > 0
                    and future_frame_index % self.future_bev_detach_interval == 0):
                memory_feat = memory_feat.detach()


            prev_bev_input = torch.cat([prev_bev_input, memory_feat.unsqueeze(1)], 1)
            prev_bev_input = prev_bev_input[:, 1:, ...].contiguous()
            # update ref2future to ref_to_history_list.
            ref_to_history_list = torch.cat([ref_to_history_list, ref2future.unsqueeze(1)], 1)
            ref_to_history_list = ref_to_history_list[:, 1:].contiguous()
            # update occ_gts
            if cond_norm_dict['occ_gts'] is not None:
                cond_norm_dict['occ_gts'] = cond_norm_dict['occ_gts'][:, 1:, ...].contiguous()


        #* ================== 4.4 BEV feature -> occupancy logits ==================
        # WorldHeadV1.forward_head 将当前+未来 BEV feature 投影成 occupancy 分类 logits。
        next_bev_feats = torch.stack(next_bev_feats, 0)
        # forward head
        next_bev_preds = future_pred_head.forward_head(next_bev_feats)

        return next_bev_preds, next_bev_sem, next_pose_preds, next_pose_loss


    def compute_occ_loss(self, occ_preds, occ_gts):
        # preds
        occ_preds = occ_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
        inter_num, select_frames, bs, num_cls, hw, d = occ_preds.shape
        occ_preds = occ_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:]
        occ_gts = occ_gts.view(select_frames*bs, *occ_gts.shape[-3:])

        # occ loss
        losses_occupancy = self.future_pred_head.loss_occ(occ_preds, occ_gts)
        return losses_occupancy

    def compute_sem_norm_loss(self, bev_sem_preds, occ_gts):
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:-1]

        loss_dict = {}
        # loss sem
        if bev_sem_preds[0] is not None:
            bev_sem_preds = torch.stack(bev_sem_preds, dim=0).transpose(0,1)
            loss_sem_norm = self.future_pred_head.loss_sem_norm(bev_sem_preds, occ_gts)
        return loss_sem_norm

    def compute_sem_norm(self, bev_sem_preds, occ_gts):
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:]

        # loss sem
        if bev_sem_preds[0] is not None:
            bev_sem_preds = torch.stack(bev_sem_preds, dim=0).permute(1,2,0,3,4,5,6).flatten(1,2)
            loss_sem_norm = self.future_pred_head.loss_sem_norm(bev_sem_preds, occ_gts)
        return loss_sem_norm

    def compute_obj_motion_norm(self, flow_preds, flow_gts):
        # gts
        flow_gts = flow_gts[0][self.future_pred_head_flow.history_queue_length:]

        # preds
        if flow_preds[0] is not None:
            flow_preds = torch.stack(flow_preds, dim=0).permute(1,2,0,3,4,5,6).flatten(1,2)
            losses_flow = self.future_pred_head_flow.loss_obj_motion_norm(flow_preds, flow_gts)
        return losses_flow

    def get_one_hot(self, label, N):
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(N).to(label)
        ones = ones.index_select(0, label.long())
        size.append(N)
        ones = ones.view(*size)
        ones = ones.transpose(2, 3)
        ones = ones.transpose(1, 2)
        return ones

    def compute_flow_loss(self, flow_preds, flow_gts):
        # preds
        flow_preds = flow_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
        inter_num, select_frames, bs, num_cls, hw, d = flow_preds.shape
        flow_preds = flow_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # gts
        flow_gts = flow_gts[0][self.future_pred_head_flow.history_queue_length:]
        flow_gts = flow_gts.view(select_frames*bs, *flow_gts.shape[-4:])
        # flow loss
        losses_flow = self.future_pred_head_flow.loss_flow(flow_preds, flow_gts)
        return losses_flow

    def compute_plan_loss(self, outs_planning, sdc_planning, sdc_planning_mask, gt_future_boxes):
        ## outs_planning, sdc_planning: under ref_lidar coord
        pred_under_ref = torch.cumsum(outs_planning, dim=1)
        gt_under_ref = torch.cumsum(sdc_planning, dim=1)

        losses_plan = self.plan_head.loss(pred_under_ref, gt_under_ref, sdc_planning_mask, gt_future_boxes)
        return losses_plan

    def evaluate_occ(self, occ_preds, occ_gts, img_metas):
        # preds
        occ_preds = occ_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
        inter_num, select_frames, bs, num_cls, hw, d = occ_preds.shape
        occ_preds = occ_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:]
        occ_gts = occ_gts.view(select_frames*bs, *occ_gts.shape[-3:])

        hist_for_iou = self.evaluate_occupancy_forecasting(occ_preds[-1], occ_gts, img_metas=img_metas, save_pred=self._viz_pcd_flag, save_path=self._viz_pcd_path)
        hist_for_iou_current = self.evaluate_occupancy_forecasting(occ_preds[-1][0:1], occ_gts[0:1], img_metas=img_metas, save_pred=False)
        hist_for_iou_future = self.evaluate_occupancy_forecasting(occ_preds[-1][1:], occ_gts[1:], img_metas=img_metas, save_pred=False)
        hist_for_iout_future_time_weighting = self.evaluate_occupancy_forecasting(occ_preds[-1][1:], occ_gts[1:], img_metas=img_metas, time_weighting=True)
        #* 额外返回逐时间步 confusion matrix，用于 Dataset.evaluate()
        #* 输出类似论文表格的精简结果：
        #*   0-step/current, 1-step, 2-step, ... future Avg。
        # 原有 hist_for_iou_current / hist_for_iou_future 是 current 和 future
        # 聚合后的指标，无法区分每个未来步；因此这里逐帧单独计算 hist。
        hist_for_iou_per_frame = [
            self.evaluate_occupancy_forecasting(
                occ_preds[-1][frame_idx:frame_idx + 1],
                occ_gts[frame_idx:frame_idx + 1],
                img_metas=img_metas,
                save_pred=False)
            for frame_idx in range(select_frames * bs)
        ]
        return (hist_for_iou, hist_for_iou_current, hist_for_iou_future,
                hist_for_iout_future_time_weighting, hist_for_iou_per_frame)

    def evaluate_instance(self, occ_preds, flow_preds, occ_gts, instance_gts):
        # occ_preds
        occ_preds = occ_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
        inter_num, select_frames, bs, num_cls, hw, d = occ_preds.shape
        occ_preds = occ_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # occ_gts
        occ_gts = occ_gts[0][self.future_pred_head.history_queue_length:]
        occ_gts = occ_gts.view(select_frames*bs, *occ_gts.shape[-3:])
        # flow_preds
        flow_preds = flow_preds.permute(1, 0, 3, 2, 6, 4, 5).squeeze(3)
        inter_num, select_frames, bs, num_cls, hw, d = flow_preds.shape
        flow_preds = flow_preds.view(inter_num, select_frames*bs, num_cls, self.bev_w, self.bev_h, d).transpose(3,4)
        # instance_gts
        instance_gts = instance_gts[0][self.future_pred_head.history_queue_length:]
        instance_gts = instance_gts.view(select_frames*bs, *instance_gts.shape[-3:])

        vpq = self.evaluate_instance_prediction(occ_preds[-1], flow_preds[-1], occ_gts, instance_gts)
        return vpq

    def evaluate_plan(self, next_pose_preds, sdc_planning, sdc_planning_mask, segmentation_bev, img_metas):
        """
            pred_ego_fut_trajs: B,Lout,2
            gt_ego_fut_trajs:   B,Lout,2
            segmentation_bev:   B,Lout,h,w
        """
        next_pose_gts = sdc_planning

        # pred, gt: under ref_lidar coord
        pred_under_ref = torch.cumsum(next_pose_preds[..., :2], dim=1)
        gt_under_ref = torch.cumsum(next_pose_gts[..., :2], dim=1).float()

        if self._viz_pcd_flag:
            save_data = np.load(os.path.join(self._viz_pcd_path, img_metas[0]["scene_token"]+'_'+img_metas[0]["lidar_token"]+'.npz'), allow_pickle=True)
            np.savez(os.path.join(self._viz_pcd_path, img_metas[0]["scene_token"]+'_'+img_metas[0]["lidar_token"]),
                                occ_pred=save_data['occ_pred'], pose_pred=pred_under_ref[0].detach().cpu().numpy())

        self.planning_metric_v2(pred_under_ref, gt_under_ref, sdc_planning_mask, segmentation_bev)


    @auto_fp16(apply_to=('img', 'segmentation', 'flow', 'sdc_planning'))
    def forward_train(self,
                      img_metas=None,
                      img=None, # (1 3 6 3 992 1760)
                      # occ_flow
                      segmentation=None,
                      instance=None,
                      flow=None,
                      # sdc-plan
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      # sample_traj
                      sample_traj=None,
                      # vel_sterring
                      vel_steering=None,
                      ):
        """Forward training function.
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            segmentation (list[torch.Tensor])
            flow (list[torch.Tensor])
            sample_traj
        Returns:
            dict: Losses of different branches.
        """

        # manually stop forward
        if self.only_generate_dataset:
            return {"pseudo_loss": torch.tensor(0.0, device=img.device, requires_grad=True)}


        #* ================== 1. 输入增强/随机丢帧 ==================
        # 对当前帧或历史帧做随机 drop，用于增强模型对时序缺失的鲁棒性。
        # A1. Randomly drop cur image input.
        if np.random.rand() < self.random_drop_image_rate:
            img[:, -1:, ...] = torch.zeros_like(img[:, -1:, ...])
        # A2. Randomly drop previous image inputs.
        num_frames = img.size(1) # 过去帧数量
        if np.random.rand() < self.random_drop_prev_rate:
            random_drop_prev_v2_end_idx = (
                self.random_drop_prev_end_idx if self.random_drop_prev_end_idx is not None
                else num_frames)
            drop_prev_index = np.random.randint(
                self.random_drop_prev_start_idx, random_drop_prev_v2_end_idx)
        else:
            drop_prev_index = -1

        prev_img = img[:, :-1, ...] # (1 2 1 3 384 1248)
        prev_img_metas = copy.deepcopy(img_metas)
        # B2. Randomly grid-mask prev_bev.
        prev_bev, prev_bev_list = self.obtain_history_bev(prev_img, prev_img_metas, drop_prev_index=drop_prev_index) # (1 40000 256)

        # B2. Randomly grid-mask prev_bev.
        if self.grid_mask_prev and prev_bev is not None:
            b, n, c = prev_bev.shape
            assert n == self.bev_h * self.bev_w
            prev_bev = prev_bev.view(b, self.bev_h, self.bev_w, c)
            prev_bev = prev_bev.permute(0, 3, 1, 2).contiguous()
            prev_bev = self.grid_mask(prev_bev)
            prev_bev = prev_bev.view(b, c, n).permute(0, 2, 1).contiguous()

        img = img[:, -1, ...] # (1 6 3 992 1760)
        img_metas = [each[num_frames-1] for each in img_metas]
        if self.turn_on_plan:
            ref_sample_traj = sample_traj[:, :, 0]
            ref_real_traj = sdc_planning[:, 0]
            ref_command = command[:, 0]
            sem_occupancy = segmentation[0][self.future_pred_head.history_queue_length:].unsqueeze(0)   # using GT occupancy to calculate sample_traj cost during training
            sem_occupancy = F.interpolate(sem_occupancy, size=(self.bev_h, self.bev_w, self.future_pred_head.num_pred_height), mode='nearest')
            ref_sem_occupancy = sem_occupancy[:, 0]
            # 预测当前帧的BEV特征，轨迹
            ref_bev, ref_pose_pred, ref_pose_loss = self.obtain_ref_bev_with_plan(img, img_metas, prev_bev, ref_sample_traj, ref_sem_occupancy, ref_command, ref_real_traj)
        else:
            # 历史BEV特征：目前仅用于构建当前时序BEV特征
            ref_bev = self.obtain_ref_bev(img, img_metas, prev_bev) # (1 40000 256)
            sem_occupancy, ref_pose_pred, ref_pose_loss = None, None, None


        #* ================== 4. 当前/历史 BEV -> 自回归未来 BEV/Occupancy ==================
        #* 对应论文 3.2 的 Memory Queue W_M + World Decoder W_D：
        #* 先把历史/当前 BEV 组成 WM，再由 W_D 自回归预测未来 BEV 和 semantic occupancy。
        # 用 memory_queue、future pose/action condition 和 WorldHeadV1 逐帧预测未来 BEV。
        #* 自回归：future_pred() 内部会先预测 t+1，再把 t+1 的预测 BEV 放回 memory_queue，
        #* 继续预测 t+2/t+3/...，所以更远未来会依赖更近未来的预测结果。
        valid_frames = [0]  # 参与 loss 的预测帧编号；0 表示当前参考帧 occupancy。
        if not self.only_train_cur_frame:  # 默认 False，说明不仅训练当前帧，还训练未来帧预测。
            # * supervise_all_future:
            if self.supervise_all_future:  # 默认 True，对所有未来帧都计算 occupancy loss。
                valid_frames.extend(list(range(1, self.future_pred_frame_num + 1)))  # 加入 1..future_pred_frame_num，默认 1..4。[0 1 2 3 4]
            else:  # randomly select one future frame for computing loss to save memory cost.
                train_frame = np.random.choice(np.arange(1, self.future_pred_frame_num + 1), 1)[0]  # 随机抽一个未来帧省显存。
                valid_frames.append(train_frame)  # 只监督当前帧 + 抽中的未来帧。

            prev_bev_list = torch.stack(prev_bev_list, dim=1) # (1 2 40000 256) 历史BEV特征
            prev_bev_list = torch.cat([prev_bev_list, ref_bev.unsqueeze(1)], dim=1)[:, -self.memory_queue_len:, ...] # (1 40000 256) 将历史和当前bev特征拼接 # (1 1 65536 256) memory queue 长度；设为 1 表示每一步只保留最新的 BEV 作为下一步预测的 memory。
            # D2. prepare conditional-normalization dict.
            if self.future_pred_head.prev_render_neck.sem_norm and self.future_pred_head.prev_render_neck.sem_gt_train and self.training_epoch < 12: # False
                occ_gts = segmentation[0][self.future_pred_head.history_queue_length+1-self.memory_queue_len:-1]  # 取与 memory_queue 对齐的 GT occ。
                occ_gts = F.interpolate(occ_gts.unsqueeze(1), size=(self.bev_h, self.bev_w, self.future_pred_head.prev_render_neck.pred_height), mode='nearest').transpose(0,1)  # resize 到 BEV/head 使用的空间尺寸。
            else:
                occ_gts = None  # 默认路径：ConditionalNorm 不使用 GT occupancy，只使用预测/特征自身。

            cond_norm_dict = {'occ_gts': occ_gts}  # 传给 future_pred_head.prev_render_neck 使用。
            # D3. prepare action condition dict.
            action_condition_dict = {'command':command, 'vel_steering': vel_steering} # command:(1 5) vel_steering:(1 5 4)
            # D4. prepare planning dict.
            plan_dict = {'sem_occupancy': sem_occupancy, 'sample_traj': sample_traj, 'gt_traj': sdc_planning, 'ref_pose_pred': ref_pose_pred} # sem_occupancy: None sample_traj:(1 1800 5 3) sdc_planning:(1 5 3) ref_pose_pred: None

            # D5. predict future occ in auto-regressive manner
            #* 对应论文 Future Forecasting with World Decoder：
            next_bev_preds, next_bev_sem, next_pose_preds, next_pose_loss = self.future_pred(prev_bev_list, action_condition_dict, cond_norm_dict, plan_dict,
                                                                            valid_frames, img_metas, prev_img_metas, num_frames, occ_flow='occ')


            # D6. predict future flow in auto-regressive manner
            if self.turn_on_flow:  # 当前配置 False，默认不预测 flow，也不计算 VPQ/flow loss。
                next_bev_preds_flow, _, _, _ = self.future_pred(prev_bev_list, action_condition_dict, cond_norm_dict, plan_dict,
                                                                valid_frames, img_metas, prev_img_metas, num_frames, occ_flow='flow')


        #* ================== 5. 计算训练损失 ==================
        # 当前配置主要使用 occupancy loss；turn_on_flow/turn_on_plan 打开时才会额外计算 flow/plan loss。
        losses = dict()
        # E1. Compute loss for occ predictions.
        losses_occupancy = self.compute_occ_loss(next_bev_preds, segmentation)
        losses.update(losses_occupancy)
        # E2. Compute loss for flow predictions.
        if self.turn_on_flow: # False
            losses_flow = self.compute_flow_loss(next_bev_preds_flow, flow)
            losses.update(losses_flow)

        # E3. Compute loss for plan regression.
        if self.turn_on_plan: # False
            if 'v1' in self.plan_head_type: # used for fine-grained_MMO when sem_occupancy distinguish categories in MMO
                gt_future_boxes = gt_future_boxes[0]   # Lout,[boxes]  NOTE: Current Support bs=1
                losses_plan = self.compute_plan_loss(next_pose_preds, sdc_planning, sdc_planning_mask, gt_future_boxes)
                losses_plan_cost = ref_pose_loss + sum(next_pose_loss)
                losses_plan.update(losses_plan_cost = 0.1 * losses_plan_cost)
            elif 'v2' in self.plan_head_type:   # used for inflated_GMO when sem_occupancy does not distinguish categories in GMO
                gt_future_boxes = gt_future_boxes[0]   # Lout,[boxes]  NOTE: Current Support bs=1
                losses_plan = self.compute_plan_loss(next_pose_preds, sdc_planning, sdc_planning_mask, gt_future_boxes)
            losses.update(losses_plan)
        # E4. Compute loss for bev rendering
        if self.future_pred_head.prev_render_neck.sem_norm: # True
            losses_bev_render = self.compute_sem_norm_loss(next_bev_sem, segmentation)
            losses.update(losses_bev_render)
        # if self.future_pred_head.sem_norm:
        #     losses_sem_norm = self.compute_sem_norm(next_bev_sem, segmentation)
        #     losses.update(losses_sem_norm)
        # if self.turn_on_flow and self.future_pred_head_flow.obj_motion_norm:
        #     losses_obj_motion_norm = self.compute_obj_motion_norm(next_bev_flow, flow)
        #     losses.update(losses_obj_motion_norm)

        return losses



    def forward_test(self,
                     img_metas,
                     img=None,
                     # occ_flow
                     segmentation=None,
                     instance=None,
                     flow=None,
                     # sdc-plan
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
                     segmentation_bev=None,
                     # sample_traj
                     sample_traj=None,
                     # vel_sterring
                     vel_steering=None,
                     **kwargs):
        """has similar implementation with train forward."""

        # manually stop forward
        if self.only_generate_dataset:
            return {'hist_for_iou': 0, 'pred_c': 0, 'vpq':0}


        self.eval()
        #* ================== 1. 测试：历史帧图像 -> 历史 BEV ==================
        # B. Forward previous frames.
        num_frames = img.size(1)
        prev_img = img[:, :-1, ...]
        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev, prev_bev_list = self.obtain_history_bev(prev_img, prev_img_metas)


        #* ================== 2. 测试：当前帧图像 -> 当前参考 BEV ==================
        img = img[:, -1, ...]
        img_metas = [each[num_frames-1] for each in img_metas]
        if self.turn_on_plan:
            #* ref_sample_traj 是规划自回归链路的“第 0 步”候选轨迹：
            # sample_traj: [B, sample_num, future_step, 3]，例如 [1, 1800, 5, 3]。
            # sample_traj[:, :, 0] 表示 ref/current -> t+1 的 1800 条候选一步位移。
            # obtain_ref_bev_with_plan() 会：
            #   1. 先从当前帧图像提取 ref_bev；
            #   2. 用 ref_bev 预测当前参考帧 occupancy；
            #   3. 用该 occupancy 对 ref_sample_traj 计算 cost；
            #   4. 选/细化得到 ref_pose_pred，即第一步规划结果。
            #
            # 后续 t+1->t+2、t+2->t+3 ... 不在这里完成，
            # 而是在下面 self.future_pred() 内部的 for future_frame_index 循环中完成。
            ref_sample_traj = sample_traj[:, :, 0] # (1 1800 5 3) -> (1  1800 3) # 从一整段未来候选轨迹序列中，选取下一帧 的 1800 条候选一步位移
            ref_command = command[:, 0] # (1,)
            ref_sem_occupancy = None
            ref_bev, ref_pose_pred, _ = self.obtain_ref_bev_with_plan(img, img_metas, prev_bev, ref_sample_traj, ref_sem_occupancy, ref_command)
        else:
            ref_bev = self.obtain_ref_bev(img, img_metas, prev_bev)
            ref_pose_pred = None


        #* ================== 3. 测试：自回归预测未来 BEV/Occupancy ==================
        valid_frames = [] # no frame have grad
        # D1. prepare memory_queue
        prev_bev_list = torch.stack(prev_bev_list, dim=1)
        prev_bev_list = torch.cat([prev_bev_list, ref_bev.unsqueeze(1)], dim=1)[:, -self.memory_queue_len:, ...]
        # D2. prepare conditional-normalization dict
        cond_norm_dict = {'occ_gts': None}
        # D3. prepare action condition dict
        action_condition_dict = {'command':command, 'vel_steering': vel_steering}
        # D4. prepare planning dict
        # ref_pose_pred 是第 0 步规划结果 ref/current -> t+1。
        # sample_traj 仍保留完整未来候选序列 [B, sample_num, future_step, 3]，
        # future_pred() 会从 future_frame_index=1 开始逐步取 sample_traj[:, :, i]，
        # 继续预测 t+i -> t+i+1，并把结果追加到 next_pose_preds。
        plan_dict = {'sem_occupancy': None, 'sample_traj': sample_traj, 'gt_traj': sdc_planning, 'ref_pose_pred': ref_pose_pred}

        # D5. predict future occ in auto-regressive manner
        #* 注意：future_pred() 不只预测未来 occupancy。
        # 当 turn_on_plan=True 且 occ_flow='occ' 时，它还会在内部循环中继续逐步预测未来轨迹：
        #   初始化 next_pose_preds = ref_pose_pred       # ref -> t+1
        #   future_frame_index=1 追加 pose_pred          # t+1 -> t+2
        #   future_frame_index=2 追加 pose_pred          # t+2 -> t+3
        #   ...
        # 因此这里返回的 next_pose_preds 是多步规划结果。
        next_bev_preds, _, next_pose_preds, _ = self.future_pred(prev_bev_list, action_condition_dict, cond_norm_dict, plan_dict,
                                                                valid_frames, img_metas, prev_img_metas, num_frames, occ_flow='occ')

        # D6. predict future flow in auto-regressive manner
        if self.turn_on_flow:
            next_bev_preds_flow, _, _, _ = self.future_pred(prev_bev_list, action_condition_dict, cond_norm_dict, plan_dict,
                                                            valid_frames, img_metas, prev_img_metas, num_frames, occ_flow='flow')


        #* ================== 4. 测试：评估 occupancy / flow / planning ==================
        test_output = {}
        # evaluate occ
        (occ_iou, occ_iou_current, occ_iou_future,
         occ_iou_future_time_weighting,
         occ_iou_per_frame) = self.evaluate_occ(
            next_bev_preds, segmentation, img_metas)
        test_output.update(hist_for_iou=occ_iou, hist_for_iou_current=occ_iou_current,
                           hist_for_iou_future=occ_iou_future,
                           hist_for_iou_future_time_weighting=occ_iou_future_time_weighting,
                           hist_for_iou_per_frame=occ_iou_per_frame)
        # evaluate flow(instance)
        if self.turn_on_flow:
            vpq = self.evaluate_instance(next_bev_preds, next_bev_preds_flow, segmentation, instance)
            test_output.update(vpq=vpq)
        else:
            # test_output.update(vpq=0.1)
            # todo: 曾导致多卡评估额外卡住/出错的问题点：
            # todo: 原来即使 turn_on_flow=False，也返回 vpq=0.1 占位。
            # todo: custom_multi_gpu_test() 看到 result 里有 vpq，就会继续走
            # todo: VPQ collect 分支，导致当前 SemanticKITTI occupancy-only 实验
            # todo: 多一次完全无意义的分布式 pkl 收集。
            #! 修复原因：
            #! SemanticKITTI 第一阶段配置 turn_on_flow=False，不预测 flow / instance，
            #! 因此 VPQ 没有实际意义。原来的占位 vpq=0.1 会让多卡评估
            #! custom_multi_gpu_test() 误以为需要收集 VPQ 指标，额外触发一次
            #! collect_results_cpu()，增加 part_*.pkl 同步/覆盖风险。
            #! 关闭 flow 时不返回 vpq，让评估只汇总 occupancy IoU/mIoU。
            pass
        # =============================================#
        # evluate plan
        if self.turn_on_plan:
            self.evaluate_plan(next_pose_preds, sdc_planning, sdc_planning_mask, segmentation_bev, img_metas)

        # 原始写法：直接返回单个 dict。
        # return test_output
        #! 修改原因：
        #! tools/train.py 中的 EvalHook 会调用 mmdet.apis.single_gpu_test。
        #! single_gpu_test 期望模型每个 batch 的测试输出是 list，
        #! list 中每个元素对应 batch 内一个样本的预测/评估结果。
        #! 如果这里直接返回 dict，single_gpu_test 内部访问 result[0] 时
        #! 会把 0 当成 dict key，从而触发 KeyError: 0。
        #!
        #! 当前 Drive-OccWorld / SemanticKITTI 评估默认只支持 bs=1，
        #! 因此把单样本 test_output 包成 [test_output]。这样 EvalHook
        #! 后续 results.extend(result) 得到 list[dict]，再交给
        #! SemanticKITTIWorldDataset.evaluate() 汇总 hist_for_iou / mIoU /
        #! binary IoU。
        return [test_output]



    def evaluate_occupancy_forecasting(self, pred, gt, img_metas=None, save_pred=False, save_path=None, time_weighting=False):

        B, H, W, D = gt.shape
        pred = F.interpolate(pred, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()

        hist_all = 0
        iou_per_pred_list = []
        pred_list = []
        gt_list = []
        for i in range(B):
            pred_cur = pred[i,...]
            pred_cur = torch.argmax(pred_cur, dim=0).cpu().numpy()
            gt_cur = gt[i, ...].cpu().numpy()
            gt_cur = gt_cur.astype(np.int)

            pred_list.append(pred_cur)
            gt_list.append(gt_cur)

            # ignore noise
            noise_mask = gt_cur != 255

            # GMO and others for max_label=2
            # multiple movable objects for max_label=9
            hist_cur, iou_per_pred = fast_hist(pred_cur[noise_mask], gt_cur[noise_mask], max_label=self.future_pred_head.num_classes)
            if time_weighting:
                hist_all = hist_all + 1 / (i+1) * hist_cur
            else:
                hist_all = hist_all + hist_cur
            iou_per_pred_list.append(iou_per_pred)

        # whether save prediction results
        if save_pred:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            pred_for_save_list = []
            for k in range(B):
                pred_for_save = torch.argmax(pred[k], dim=0).cpu()
                x_grid = torch.linspace(0, H-1, H, dtype=torch.long)
                x_grid = x_grid.view(H, 1, 1).expand(H, W, D)
                y_grid = torch.linspace(0, W-1, W, dtype=torch.long)
                y_grid = y_grid.view(1, W, 1).expand(H, W, D)
                z_grid = torch.linspace(0, D-1, D, dtype=torch.long)
                z_grid = z_grid.view(1, 1, D).expand(H, W, D)
                segmentation_for_save = torch.stack((x_grid, y_grid, z_grid), -1)
                segmentation_for_save = segmentation_for_save.view(-1, 3)
                segmentation_label = pred_for_save.squeeze(0).view(-1,1)
                segmentation_for_save = torch.cat((segmentation_for_save, segmentation_label), dim=-1) # N,4
                kept = segmentation_for_save[:,-1]!=0
                segmentation_for_save= segmentation_for_save[kept].cpu().numpy()
                pred_for_save_list.append(segmentation_for_save)
            np.savez(os.path.join(save_path, img_metas[0]["scene_token"]+'_'+img_metas[0]["lidar_token"]), occ_pred=pred_for_save_list)

        return hist_all

    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        sdc_planning_mask,
        segmentation_bev
    ):
        """Compute planner metric for one sample same as stp3
            pred_ego_fut_trajs: B,Lout,2
            gt_ego_fut_trajs: B,Lout,2
            sdc_planning_mask: B,Lout
            segmentation_bev: B,Lout,h,w
        """
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            'plan_L2_1s_single':0,
            'plan_L2_2s_single':0,
            'plan_L2_3s_single':0,
            'plan_obj_col_1s_single':0,
            'plan_obj_col_2s_single':0,
            'plan_obj_col_3s_single':0,
            'plan_obj_box_col_1s_single':0,
            'plan_obj_box_col_2s_single':0,
            'plan_obj_box_col_3s_single':0,

        }
        future_second = 1
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        for i in range(future_second):
            cur_time = (i+1)*2
            traj_L2 = self.planning_metric.compute_L2(
                pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                gt_ego_fut_trajs[0, :cur_time],
                sdc_planning_mask[0, :cur_time]
            )
            traj_L2_single = self.planning_metric.compute_L2(
                pred_ego_fut_trajs[0, cur_time-2:cur_time].detach().to(gt_ego_fut_trajs.device),
                gt_ego_fut_trajs[0, cur_time-2:cur_time],
                sdc_planning_mask[0, cur_time-2:cur_time]
            )
            obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                pred_ego_fut_trajs[:, :cur_time].detach(),
                gt_ego_fut_trajs[:, :cur_time],
                segmentation_bev[:, :cur_time])
            obj_coll_single, obj_box_coll_single = self.planning_metric.evaluate_coll(
                pred_ego_fut_trajs[:, cur_time-2:cur_time].detach(),
                gt_ego_fut_trajs[:, cur_time-2:cur_time],
                segmentation_bev[:, cur_time-2:cur_time])
            metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
            metric_dict['plan_L2_{}s_single'.format(i+1)] = traj_L2_single
            metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean()
            metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.mean()
            metric_dict['plan_obj_col_{}s_single'.format(i+1)] = obj_coll_single.mean()
            metric_dict['plan_obj_box_col_{}s_single'.format(i+1)] = obj_box_coll_single.mean()

        return metric_dict

    def evaluate_instance_prediction(self, pred_seg, pred_flow, gt_seg, gt_instance):
        """
            pred_seg:  pred_occ:  B*Lout,C,H,W,D
            pred_flow: pred_flow: B*Lout,3,H,W,D
            gt_seg:    gt_occ:    B*Lout,H,W,D
            gt_instance: gt_instance_id: B*Lout,H,W,D
        """

        B, H, W, D = gt_seg.shape

        pred_consistent_instance_seg = self.predict_instance_segmentation(pred_seg, pred_flow)

        # add one feature dimension for interpolate
        pred_consistent_instance_seg = F.interpolate(pred_consistent_instance_seg.float(), size=[H, W, D], mode='nearest').contiguous()
        pred_consistent_instance_seg = pred_consistent_instance_seg.squeeze(1)

        iou = 0
        true_positive = 0
        false_positive = 0
        false_negative = 0

        # starting from the present frame
        pred_instance = pred_consistent_instance_seg
        gt_instance = gt_instance.long()

        assert gt_instance.min() == 0, 'ID 0 of gt_instance must be background'
        pred_segmentation = (pred_instance > 0).long()
        gt_segmentation = (gt_instance > 0).long()

        unique_id_mapping = {}
        for t in range(pred_segmentation.shape[0]):
            result = self.panoptic_metrics(
                pred_segmentation[t].detach(),
                pred_instance[t].detach(),
                gt_segmentation[t],
                gt_instance[t],
                unique_id_mapping,
            )

            iou += result['iou']
            true_positive += result['true_positive']
            false_positive += result['false_positive']
            false_negative += result['false_negative']

        denominator = torch.maximum(
            (true_positive + false_positive / 2 + false_negative / 2),
            torch.ones_like(true_positive)
        )
        pq = iou / denominator

        return pq.cpu().numpy()

    def find_instance_centers(self, center_prediction, conf_threshold=0.1, nms_kernel_size=3, dist_threshold=2):
        assert len(center_prediction.shape) == 4

        center_prediction = F.threshold(center_prediction, threshold=conf_threshold, value=-1)

        nms_padding = (nms_kernel_size - 1) // 2
        maxpooled_center_prediction = F.max_pool3d(
            center_prediction, kernel_size=nms_kernel_size, stride=1, padding=nms_padding
        )

        # Filter all elements that are not the maximum (i.e. the center of the heatmap instance)
        center_prediction[center_prediction != maxpooled_center_prediction] = -1
        centers = torch.nonzero(center_prediction > 0)[:, 1:].float()

        # distance threshold
        if len(self.vehicles_id) == 2:
            return centers
        else:
            distances = torch.cdist(centers, centers, p=2)
            combine_mask = (distances < dist_threshold).float()
            combine_centers = torch.mm(combine_mask, centers) / combine_mask.sum(-1).unsqueeze(1)
            combine_centers = torch.unique(combine_centers, dim=0).long()

            return combine_centers # Nc,3

    def group_pixels(self, centers, offset_predictions):
        dx, dy, dz = offset_predictions.shape[-3:]
        x_grid = (
            torch.arange(dx, dtype=offset_predictions.dtype, device=offset_predictions.device)
            .view(1, dx, 1, 1)
            .repeat(1, 1, dy, dz)
        )
        y_grid = (
            torch.arange(dy, dtype=offset_predictions.dtype, device=offset_predictions.device)
            .view(1, 1, dy, 1)
            .repeat(1, dx, 1, dz)
        )
        z_grid = (
            torch.arange(dz, dtype=offset_predictions.dtype, device=offset_predictions.device)
            .view(1, 1, 1, dz)
            .repeat(1, dx, dy, 1)
        )

        pixel_grid = torch.cat((x_grid, y_grid, z_grid), dim=0)
        center_locations = (pixel_grid + offset_predictions).view(3, dx*dy*dz, 1).permute(2, 1, 0)
        centers = centers.view(-1, 1, 3)

        distances = torch.norm(centers - center_locations, dim=-1)

        instance_id = torch.argmin(distances, dim=0).reshape(1, dx, dy, dz) + 1
        return instance_id

    def update_instance_ids(self, instance_seg, old_ids, new_ids):
        indices = torch.arange(old_ids.max() + 1, device=instance_seg.device)
        for old_id, new_id in zip(old_ids, new_ids):
            indices[old_id] = new_id

        return indices[instance_seg].long()

    def make_instance_seg_consecutive(self, instance_seg):
        # Make the indices of instance_seg consecutive
        unique_ids = torch.unique(instance_seg)
        new_ids = torch.arange(len(unique_ids), device=instance_seg.device)
        instance_seg = self.update_instance_ids(instance_seg, unique_ids, new_ids)
        return instance_seg

    def get_instance_segmentation_and_centers(self,
        center_predictions,
        offset_predictions,
        foreground_mask,
        conf_threshold=0.1,
        nms_kernel_size=5,
        max_n_instance_centers=100,):

        dx, dy, dz = offset_predictions.shape[-3:]
        center_predictions = center_predictions.view(1, -1, dx, dy, dz) # 1,cls,x,y,z
        offset_predictions = offset_predictions.view(3, dx, dy, dz)
        foreground_mask = foreground_mask.view(1, dx, dy, dz)

        # class-wise center proposal
        cls_kernel_size = [7, 15, 11, 17, 7, 11, 13, 5]
        center_dist_threshold = [2, 10, 4, 8, 2.3, 12, 6.4, 1.6]  # bicycle, bus, car, construction, motorcycle, trailer, truck
        center_dist_threshold = [x * 2 for x in center_dist_threshold]  # voxel-wise 0.5m
        centers = []
        for i in range(center_predictions.shape[1]):
            center_cls = self.find_instance_centers(center_predictions[:,i], conf_threshold=conf_threshold, nms_kernel_size=nms_kernel_size, dist_threshold=center_dist_threshold[i])    # Nc,3
            centers.append(center_cls)
        centers = torch.cat(centers, dim=0)

        if not len(centers):
            return torch.zeros(foreground_mask.shape, dtype=torch.int64, device=center_predictions.device)

        if len(centers) > max_n_instance_centers:
            centers = centers[:max_n_instance_centers].clone()

        instance_ids = self.group_pixels(centers, offset_predictions * foreground_mask.float())
        instance_seg = (instance_ids * foreground_mask.float()).long()

        # Make the indices of instance_seg consecutive
        instance_seg = self.make_instance_seg_consecutive(instance_seg)

        return instance_seg.long()  # 1,H,W,D

    def flow_warp(self, occupancy, flow, mode='nearest', padding_mode='zeros'):
        '''
        Warp ground-truth flow-origin occupancies according to predicted flows
        '''

        _, num_waypoints, _, grid_dx_cells, grid_dy_cells, grid_dz_cells = occupancy.size()

        dx = torch.linspace(-1, 1, steps=grid_dx_cells)
        dy = torch.linspace(-1, 1, steps=grid_dy_cells)
        dz = torch.linspace(-1, 1, steps=grid_dz_cells)

        x_idx, y_idx, z_idx = torch.meshgrid(dx, dy, dz)
        identity_indices = torch.stack((x_idx, y_idx, z_idx), dim=0).to(device=occupancy.device)

        warped_occupancy = []
        for k in range(num_waypoints):  # 1
            flow_origin_occupancy = occupancy[:, k]  # B T 1 dx dy dz -> B 1 dx dy dz
            pred_flow = flow[:, k]  # B T 3 dx dy dz -> B 3 dx dy dz
            # Normalize along the width and height direction
            normalize_pred_flow = torch.stack(
                (2.0 * pred_flow[:, 0] / (grid_dx_cells - 1),
                2.0 * pred_flow[:, 1] / (grid_dy_cells - 1),
                2.0 * pred_flow[:, 2] / (grid_dz_cells - 1),),
                dim=1,
            )

            warped_indices = identity_indices + normalize_pred_flow
            warped_indices = warped_indices.permute(0, 2, 3, 4, 1)

            flow_origin_occupancy = flow_origin_occupancy.permute(0, 1, 4, 3, 2)

            sampled_occupancy = F.grid_sample(
                input=flow_origin_occupancy,
                grid=warped_indices,
                mode=mode,
                padding_mode='zeros',
                align_corners=True,
            )
            warped_occupancy.append(sampled_occupancy)
        return warped_occupancy[0]

    def make_instance_id_temporally_consecutive(self, pred_inst, preds, backward_flow, ignore_index=255.0):

        assert pred_inst.shape[0] == 1, 'Assumes batch size = 1'

        # Initialise instance segmentations with prediction corresponding to the present
        consistent_instance_seg = [pred_inst.unsqueeze(0)]
        backward_flow = backward_flow.clone().detach()
        backward_flow[backward_flow == ignore_index] = 0.0
        seq_len, _, dx, dy, dz = preds.shape

        for t in range(1, seq_len):

            init_warped_instance_seg = self.flow_warp(consistent_instance_seg[-1].unsqueeze(0).float(), backward_flow[t:t+1].unsqueeze(0)).int()

            warped_instance_seg = init_warped_instance_seg * preds[t:t+1, 0]

            consistent_instance_seg.append(warped_instance_seg)

        consistent_instance_seg = torch.cat(consistent_instance_seg, dim=1)
        return consistent_instance_seg

    def predict_instance_segmentation(self, pred_seg, pred_flow):
        """
            pred_seg:  pred_occ:  B*Lout,C,H,W,D
            pred_flow: pred_flow: B*Lout,3,H,W,D
        """
        pred_seg_sm = pred_seg.detach()
        pred_seg_sm = torch.argmax(pred_seg_sm, dim=1, keepdims=True)
        vehicles_id = torch.tensor(self.vehicles_id).to(pred_seg_sm)
        foreground_masks = torch.isin(pred_seg_sm.squeeze(1), vehicles_id)

        pred_inst_batch = self.get_instance_segmentation_and_centers(
            torch.softmax(pred_seg, dim=1)[0:1, self.vehicles_id].detach(),
            pred_flow[1:2].detach(),
            foreground_masks[1:2].detach(),
            nms_kernel_size=7,
        )

        pred_seg_sm = torch.tensor(pred_seg_sm.detach() > 0, dtype=torch.int)   # sem_classes -> GMO
        consistent_instance_seg = self.make_instance_id_temporally_consecutive(
                pred_inst_batch,
                pred_seg_sm[1:],
                pred_flow[1:].detach(),
                )

        consistent_instance_seg = torch.cat([pred_inst_batch.unsqueeze(0), consistent_instance_seg], dim=1)

        return consistent_instance_seg.permute(1, 0, 2, 3, 4).long()

    def combine_mask(self, segmentation: torch.Tensor, instance: torch.Tensor, n_classes: int, n_all_things: int):
        '''
        Shift all things ids by num_classes and combine things and stuff into a single mask
        '''
        instance = instance.view(-1)
        instance_mask = instance > 0
        instance = instance - 1 + n_classes

        segmentation = segmentation.clone().view(-1)
        segmentation_mask = torch.bitwise_and(segmentation > 0, segmentation < n_classes+1) # things_mask

        # Build an index from instance id to class id.
        instance_id_to_class_tuples = torch.cat(
            (
                instance[instance_mask & segmentation_mask].unsqueeze(1),
                segmentation[instance_mask & segmentation_mask].unsqueeze(1),
            ),
            dim=1,
        )   # N_ins_points, 2 [ins_id, sem_cls]

        instance_id_to_class = -instance_id_to_class_tuples.new_ones((n_all_things,))
        instance_id_to_class[instance_id_to_class_tuples[:, 0]] = instance_id_to_class_tuples[:, 1] # instance_id -- sem_class
        instance_id_to_class[torch.arange(n_classes, device=segmentation.device)] = torch.arange(
            n_classes, device=segmentation.device
        )

        segmentation[instance_mask] = instance[instance_mask]
        segmentation[~segmentation_mask] = 0

        return segmentation, instance_id_to_class
        # segmentation: ins_id
        # instance_id_to_class[ins_id] = sem_class

    def panoptic_metrics(self, pred_segmentation, pred_instance, gt_segmentation, gt_instance, unique_id_mapping):
        # GMO and others
        n_classes = 1   # numebr of things_class  (GMO=1)
        self.keys = ['iou', 'true_positive', 'false_positive', 'false_negative'] # hard coding
        result = {key: torch.zeros(n_classes, dtype=torch.float32, device=gt_instance.device) for key in self.keys}

        assert pred_segmentation.dim() == 3
        assert pred_segmentation.shape == pred_instance.shape == gt_segmentation.shape == gt_instance.shape

        n_instances = int(torch.cat([pred_instance, gt_instance]).max().item())
        n_all_things = n_instances + n_classes  # Classes + instances.
        n_things_and_void = n_all_things + 1

        pred_segmentation = pred_segmentation.long().detach().cpu()
        pred_instance = pred_instance.long().detach().cpu()
        gt_segmentation = gt_segmentation.long().detach().cpu()
        gt_instance = gt_instance.long().detach().cpu()

        prediction, pred_to_cls = self.combine_mask(pred_segmentation, pred_instance, n_classes, n_all_things)
        target, target_to_cls = self.combine_mask(gt_segmentation, gt_instance, n_classes, n_all_things)

        # Compute ious between all stuff and things
        # hack for bincounting 2 arrays together
        x = prediction + n_things_and_void * target
        bincount_2d = torch.bincount(x.long(), minlength=n_things_and_void ** 2)
        if bincount_2d.shape[0] != n_things_and_void ** 2:
            raise ValueError('Incorrect bincount size.')
        conf = bincount_2d.reshape((n_things_and_void, n_things_and_void))
        # Drop void class
        conf = conf[1:, 1:]
        # Confusion matrix contains intersections between all combinations of classes
        union = conf.sum(0).unsqueeze(0) + conf.sum(1).unsqueeze(1) - conf
        iou = torch.where(union > 0, (conf.float() + 1e-9) / (union.float() + 1e-9), torch.zeros_like(union).float())

        mapping = (iou > self.iou_thresh_for_vpq).nonzero(as_tuple=False)

        # Check that classes match.
        is_matching = pred_to_cls[mapping[:, 1]] == target_to_cls[mapping[:, 0]]
        mapping = mapping[is_matching.detach().cpu().numpy()]
        tp_mask = torch.zeros_like(conf, dtype=torch.bool)
        tp_mask[mapping[:, 0], mapping[:, 1]] = True

        # First ids correspond to "stuff" i.e. semantic seg.
        # Instance ids are offset accordingly
        for target_id, pred_id in mapping:
            cls_id = pred_to_cls[pred_id]
            if cls_id == 0 or cls_id == -1:
                continue

            self.temporally_consistent = True # hard coding !
            if self.temporally_consistent and cls_id == self.gmo_id:
                if target_id.item() in unique_id_mapping and unique_id_mapping[target_id.item()] != pred_id.item():
                    # Not temporally consistent
                    result['false_negative'][target_to_cls[target_id]-1] += 1
                    result['false_positive'][pred_to_cls[pred_id]-1] += 1
                    unique_id_mapping[target_id.item()] = pred_id.item()
                    continue

            result['true_positive'][cls_id-1] += 1
            result['iou'][cls_id-1] += iou[target_id][pred_id]
            unique_id_mapping[target_id.item()] = pred_id.item()

        for target_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[target_id, n_classes:].any():
                continue
            # If this target instance didn't match with any predictions and was present set it as false negative.
            if target_to_cls[target_id] != -1:
                result['false_negative'][target_to_cls[target_id]-1] += 1

        for pred_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[n_classes:, pred_id].any():
                continue
            # If this predicted instance didn't match with any prediction, set that predictions as false positive.
            if pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
                result['false_positive'][pred_to_cls[pred_id]-1] += 1

        return result

    def _viz_pcd(self, pred_pcd, pred_ctr,  output_path, gt_pcd=None):
        """Visualize predicted future point cloud."""
        color_map = np.array([
            [0, 0, 230], [219, 112, 147], [255, 0, 0]
        ])
        pred_label = np.ones_like(pred_pcd)[:, 0].astype(np.int) * 0
        if gt_pcd is not None:
            gt_label = np.ones_like(gt_pcd)[:, 0].astype(np.int)

            pred_label = np.concatenate([pred_label, gt_label], 0)
            pred_pcd = np.concatenate([pred_pcd, gt_pcd], 0)

        e2e_predictor_utils._dbg_draw_pc_function(
            pred_pcd, pred_label, color_map, output_path=output_path,
            ctr=pred_ctr, ctr_labels=np.zeros_like(pred_ctr)[:, 0].astype(np.int)
        )

def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    iou_per_pred = (bin_count[-1]/(bin_count[-1]+bin_count[1]+bin_count[2]))
    return bin_count[:max_label ** 2].reshape(max_label, max_label),iou_per_pred
