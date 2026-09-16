import copy
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from .nuscenes_dataset import CustomNuScenesDataset
import mmcv
from mmdet.datasets import DATASETS
import numpy as np
import cv2
import torch
from pyquaternion import Quaternion
from projects.mmdet3d_plugin.datasets.formating import cm_to_ious, format_iou_results
from projects.mmdet3d_plugin.datasets.trajectory_api import NuScenesTraj
from projects.mmdet3d_plugin.datasets.samplers import sampler as trajectory_sampler
from projects.mmdet3d_plugin.bevformer.dense_heads.plan_head import calculate_birds_eye_view_parameters
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from prettytable import PrettyTable


@DATASETS.register_module()
class NuScenesWorldDatasetTemplate(CustomNuScenesDataset):
    r"""World dataset for visual point cloud forecasting.

    整体数据链路：
    1. 父类从 ann_file 指向的 ``nuscenes_infos_temporal_*.pkl`` 读取 data_infos，
       其中包含 sample token、相机路径、标定、ego/lidar 位姿、3D 框、速度等基础信息。
    2. 本类初始化 nuScenes SDK、CanBus SDK 和轨迹工具，用于补充 PKL 中尚未离线保存的
       scene/location、规划标签、候选轨迹、实例时序等监督信息。
    3. ``_prepare_data_info`` 按 queue_length/future_length 取历史帧、当前帧和未来帧；
       其中当前帧作为参考帧，负责触发 occupancy、规划、未来框和实例监督的准备。
    4. ``_prepare_data_info_single`` 先用 ``get_data_info`` 组装单帧基础字段，再调用
       config 中的 pipeline 读取图像、做图像增强/归一化/pad，并通过 ``LoadOccupancy``
       从 occ_path 读取 fine-grained occupancy 标签。
    5. 子类 ``NuScenesWorldDatasetV1.union2one`` 把多帧结果合并成模型输入：历史+当前图像、
       当前参考帧坐标系下的 occupancy 序列、future can_bus、规划标签和 action condition。

    注释约定：普通 ``#`` 表示当前在数据集初始化或取样阶段执行的预处理；
    ``#!`` 表示该结果可在离线数据转换阶段写入 PKL，训练时直接读取。
    """

    def __init__(self,
                 classes,
                 use_separate_classes,
                 use_fine_occ,
                 turn_on_flow,
                 future_length,
                 ego_mask=None,
                 load_frame_interval=None,
                 rand_frame_interval=(1,),
                 plan_grid_conf=None,
                 candidate_sample_num=1800,
                 can_bus_root='',
                 *args,
                 **kwargs):
        """
        Args:
            future_length: the number of predicted future point clouds.
            ego_mask: mask points belonging to the ego vehicle.
            load_frame_interval: partial of training set.
            rand_frame_interval: augmentation for future prediction.
        """
        # Hack the original {self._set_group_flag} function.
        self.usable_index = []

        super().__init__(*args, **kwargs)
        self.classes = classes
        self.use_separate_classes = use_separate_classes
        self.use_fine_occ = use_fine_occ
        self.turn_on_flow = turn_on_flow

        #* ================== 1. Dataset 初始化 ==================
        # 这里的 ann_file/pkl 已由父类加载成 self.data_infos；本类再补充在线查询接口。
        # self.nusc: 查 nuScenes 原始 meta，如 sample、sample_annotation、scene、log。
        # self.nusc_can: 查 CAN bus，如 pose、steering，用于候选轨迹/action condition。
        # self.traj_api: 生成自车未来规划标签 sdc_planning、mask 和 command。
        # usable_index: 过滤掉历史帧或未来帧不够、或者跨 scene 的参考帧。
        # 训练前预处理：加载 nuScenes 主数据库和 CAN bus 数据，供实例、轨迹及规划标签查询。
        # SDK = Software Development Kit，这里指 nuScenes 官方 Python 查询工具包；
        # 它不是模型，而是帮代码读取 json meta、CAN bus 消息，并根据 token 查关联记录。
        #! 可在生成 PKL 时完成下述查询；若所需字段均已写入 PKL，训练阶段无需初始化这两个 SDK。
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=self.data_root, verbose=False)
        self.nusc_can = NuScenesCanBus(dataroot=can_bus_root)

        # 初始化预处理：建立 scene_name -> location 映射，用于判断新加坡左侧通行场景。
        #! location 可按帧或按场景写入 PKL，避免每次创建 Dataset 时遍历 scene/log 表。
        self.scene2map = {}
        for sce in self.nusc.scene:
            log = self.nusc.get('log', sce['log_token'])
            self.scene2map[sce['name']] = log['location']

        # 训练时预处理接口：根据 nuScenes 主表计算自车未来轨迹、有效掩码和驾驶命令。
        #! sdc_planning、sdc_planning_mask、command 可逐帧预计算并写入 PKL。
        self.traj_api = NuScenesTraj(self.nusc,
                                     self.CLASSES,
                                     self.box_mode_3d,
                                     planning_steps=future_length+1)

        # ignore_label_name
        self.ignore_bbox_label_name = ['barrier', 'traffic_cone', 'animal', 'noise',
                                       'movable_object.debris', 'movable_object.pushable_pullable', 'static_object.bicycle_rack']

        self.plan_grid_conf = plan_grid_conf
        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            plan_grid_conf['xbound'], plan_grid_conf['ybound'], plan_grid_conf['zbound'],
        )
        # convert numpy
        self.bev_resolution = self.bev_resolution.numpy()   # [0.5 0.5 20]
        self.bev_start_position = self.bev_start_position.numpy()
        self.bev_dimension = self.bev_dimension.numpy()     # [200 200 1]

        self.future_length = future_length  # 2
        self.ego_mask = ego_mask            # (-0.8, -1.5, 0.8, 2.5)
        self.load_frame_interval = load_frame_interval  # 8
        self.rand_frame_interval = rand_frame_interval  # (-1, 1) # * 默认是(1,) 即为连续帧
        #* 候选轨迹数量：默认 1800，可由 config 中的 candidate_sample_num 统一修改。
        # 需要和 PlanHead_v1.sample_num 保持一致，并且必须能被 3 整除，
        # 因为候选轨迹按 [Left, Straight, Right] 三组组织。
        assert candidate_sample_num % 3 == 0
        self.candidate_sample_num = candidate_sample_num

        # 初始化预处理：过滤历史帧或未来帧数量不足、以及跨越场景边界的样本。
        #! usable_index 或等价的 valid 标志可随固定 queue/future 配置写入 PKL；若配置会变化则应在线重算。
        # Remove data_infos without enough history & future.
        # if test, assert all history frames are available
        #  Align with the setting of 4D-occ: https://github.com/tarashakhurana/4d-occ-forecasting
        last_scene_index = None
        last_scene_frame = -1
        usable_index = []
        # valid_prev_length = (self.queue_length if self.test_mode else 0)
        valid_prev_length = self.queue_length
        for index, info in enumerate(mmcv.track_iter_progress(self.data_infos)):
            if last_scene_index != info['scene_token']:
                last_scene_index = info['scene_token']
                last_scene_frame = -1
            last_scene_frame += 1
            if last_scene_frame >= valid_prev_length:
                # has enough previous frame.
                # now, let's check whether it has enough future frame.
                tgt_future_index = index + self.future_length
                if tgt_future_index >= len(self.data_infos):
                    break
                if last_scene_index != self.data_infos[tgt_future_index]['scene_token']:
                    # the future scene is not corresponded to the current scene
                    continue
                usable_index.append(index)

        # Remove useless frame index if load_frame_interval is assigned.
        if self.load_frame_interval is not None:
            usable_index = usable_index[::self.load_frame_interval]
        self.usable_index = usable_index

        if not self.test_mode:
            self._set_group_flag()

    def reframe_boxes(self, boxes, t_init, t_curr):
        # 取样预处理：把未来帧中的 3D 框统一变换到参考帧坐标系。
        #! 对固定参考帧和 future_length，可离线保存变换后的 gt_future_boxes 或其数值数组。
        l2e_r_mat_curr = t_curr['l2e_r']
        l2e_t_curr = t_curr['l2e_t']
        e2g_r_mat_curr = t_curr['e2g_r']
        e2g_t_curr = t_curr['e2g_t']

        l2e_r_mat_init = t_init['l2e_r']
        l2e_t_init = t_init['l2e_t']
        e2g_r_mat_init = t_init['e2g_r']
        e2g_t_init = t_init['e2g_t']

        # to bbox under curr ego frame  # TODO: Uncomment
        boxes.rotate(l2e_r_mat_curr.T)
        boxes.translate(l2e_t_curr)

        # to bbox under world frame
        boxes.rotate(e2g_r_mat_curr.T)
        boxes.translate(e2g_t_curr)

        # to bbox under initial ego frame, first inverse translate, then inverse rotate
        boxes.translate(- e2g_t_init)
        m1 = np.linalg.inv(e2g_r_mat_init)
        boxes.rotate(m1.T)

        # to bbox under curr ego frame, first inverse translate, then inverse rotate
        boxes.translate(- l2e_t_init)
        m2 = np.linalg.inv(l2e_r_mat_init)
        boxes.rotate(m2.T)

        return boxes

    def get_future_bboxes(self, index):
        # 取样预处理：收集未来 3D 框，并将框底面栅格化为规划碰撞评估所需的 BEV mask。
        #! gt_future_boxes 与 segmentation_bev 可逐参考帧预计算进 PKL，避免训练时反复做坐标变换和栅格化。
        cur_info = self.data_infos[index]

        # ref pose
        dtype = torch.float32
        l2e_r = cur_info['lidar2ego_rotation']
        l2e_t = cur_info['lidar2ego_translation']
        e2g_r = cur_info['ego2global_rotation']
        e2g_t = cur_info['ego2global_translation']
        l2e_r_mat = torch.from_numpy(Quaternion(l2e_r).rotation_matrix).to(dtype)
        e2g_r_mat = torch.from_numpy(Quaternion(e2g_r).rotation_matrix).to(dtype)
        l2e_t_vec = torch.tensor(l2e_t).to(dtype)
        e2g_t_vec = torch.tensor(e2g_t).to(dtype)
        t_ref = dict(l2e_r=l2e_r_mat, l2e_t=l2e_t_vec, e2g_r=e2g_r_mat, e2g_t=e2g_t_vec)

        segmentations = []
        gt_future_boxes = []

        # generate the future
        index_list = list(range(index + 1, index + (self.future_length + 2)))

        for fur_index in index_list:
            if fur_index < len(self.data_infos) and self.data_infos[fur_index]['scene_token'] == cur_info['scene_token']:
                fur_info = self.data_infos[fur_index]
                # future_gt_bbox
                gt_bboxes_3d = fur_info['gt_boxes'].copy()
                # not exist gt_bbox
                if gt_bboxes_3d.shape[0] == 0:
                    segmentation = np.zeros(
                        (self.bev_dimension[1], self.bev_dimension[0]))
                else:
                    gt_velocity = fur_info['gt_velocity']
                    nan_mask = np.isnan(gt_velocity[:, 0])
                    gt_velocity[nan_mask] = [0.0, 0.0]
                    gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
                    gt_bboxes_3d = LiDARInstance3DBoxes(
                        gt_bboxes_3d,
                        box_dim=gt_bboxes_3d.shape[-1],
                        origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
                    # future pose
                    l2e_r = fur_info['lidar2ego_rotation']
                    l2e_t = fur_info['lidar2ego_translation']
                    e2g_r = fur_info['ego2global_rotation']
                    e2g_t = fur_info['ego2global_translation']
                    l2e_r_mat = torch.from_numpy(Quaternion(l2e_r).rotation_matrix).to(dtype)
                    e2g_r_mat = torch.from_numpy(Quaternion(e2g_r).rotation_matrix).to(dtype)
                    l2e_t_vec = torch.tensor(l2e_t).to(dtype)
                    e2g_t_vec = torch.tensor(e2g_t).to(dtype)
                    t_curr = dict(l2e_r=l2e_r_mat, l2e_t=l2e_t_vec, e2g_r=e2g_r_mat, e2g_t=e2g_t_vec)
                    # reframe bboxes
                    gt_bboxes_3d = self.reframe_boxes(gt_bboxes_3d, t_ref, t_curr)  # N_box

                    # segmentation
                    segmentation = np.zeros((self.bev_dimension[1], self.bev_dimension[0]))
                    # select box
                    gt_bboxes_names = fur_info['gt_names'].copy().tolist()
                    select_mask = [name not in self.ignore_bbox_label_name for name in gt_bboxes_names]
                    select_gt_bboxes = gt_bboxes_3d[select_mask]
                    # valid sample andd has objects
                    if len(select_gt_bboxes.tensor) > 0:
                        bbox_corners = select_gt_bboxes.corners[:, [
                            0, 3, 7, 4], :2].numpy()
                        bbox_corners = bbox_corners[..., [1, 0]]    # NOTE: H:lidar_x  W:lidar_y
                        bbox_corners = np.round(
                            (bbox_corners - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
                        # plot segmentation
                        for poly_region in bbox_corners:
                            cv2.fillPoly(segmentation, [poly_region], 1.0)
            else:
                gt_bboxes_3d = None
                segmentation = np.zeros(
                        (self.bev_dimension[1], self.bev_dimension[0])) # H,W = ignore

            gt_future_boxes.append(gt_bboxes_3d)
            segmentations.append(segmentation)

        return gt_future_boxes, segmentations

    def get_trajectory_sampling(self, rec, future_length, SAMPLE_INTERVAL=0.5):
        #* ================== 候选自车轨迹采样 sample_traj ==================
        # 取样预处理：按当前帧时间戳对齐 CAN bus 的速度和转向角，再生成候选自车轨迹。
        # 这部分和 tools/gen_new_data.py 中的离线版 get_trajectory_sampling 对应：
        # - 在线 Dataset 版本：每次 __getitem__ 时临时查 nuScenes / CAN bus 并采样；
        # - 离线脚本版本：提前把结果写入 PKL，训练时直接从 info['sample_traj'] 读取。
        #
        # 注意：这里生成的 sample_traj 不是 nuScenes 已经发生的未来 GT 轨迹，
        # 而是从“当前自车速度 + 当前方向盘转角/曲率”出发构造的一批候选动作轨迹。
        # 后续 planner 会结合预测未来 occupancy，对这些候选轨迹计算 cost，再选更优轨迹。
        #
        # 输出 shape 约为 [self.candidate_sample_num, future_length, 3]：
        # - self.candidate_sample_num: 候选轨迹条数，默认 1800，可由配置修改；
        # - future_length: 未来相邻位移 step 数；
        # - 3: 每个 step 的位移/朝向增量，通常可理解为 dx/dy/dyaw。
        #! 对固定 future_length、采样间隔和轨迹采样参数，可把 sample_traj 直接写入逐帧 PKL。
        try:
            # 当前帧所属 scene。CAN bus 消息是按 scene name 组织的，所以先由 scene_token 找到 scene。
            ref_scene = self.nusc.get("scene", rec['scene_token'])

            # vm_msgs = self.nusc_can.get_messages(ref_scene['name'], 'vehicle_monitor')
            # vm_uts = [msg['utime'] for msg in vm_msgs]
            # pose 消息：包含当前帧附近的自车状态，这里主要使用 vel[0] 作为纵向速度。
            pose_msgs = self.nusc_can.get_messages(ref_scene['name'],'pose')    # 167
            pose_uts = [msg['utime'] for msg in pose_msgs]
            # steeranglefeedback 消息：方向盘转角反馈，这里用于近似当前曲率 Kappa。
            steer_msgs = self.nusc_can.get_messages(ref_scene['name'], 'steeranglefeedback')
            steer_uts = [msg['utime'] for msg in steer_msgs]

            ref_utime = rec['timestamp']
            # vm_index = locate_message(vm_uts, ref_utime)
            # vm_data = vm_msgs[vm_index]
            # CAN bus 频率和 nuScenes keyframe 频率不完全一样：
            # locate_message 会找距离当前 sample timestamp 最近的 CAN bus 消息，
            # 因此这里使用的是“当前帧附近/最近”的速度和转角，不是历史序列，也不是未来 GT。
            pose_index = trajectory_sampler.locate_message(pose_uts, ref_utime)
            pose_data = pose_msgs[pose_index]
            steer_index = trajectory_sampler.locate_message(steer_uts, ref_utime)
            steer_data = steer_msgs[steer_index]

            # 当前初速度 v0，单位 m/s。
            # v0 = vm_data["vehicle_speed"] / 3.6  # km/h to m/s
            v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s

            # 当前方向盘转角 steering，用于估计曲率 Kappa。
            # Kappa > 0 通常表示向左转；Kappa < 0 通常表示向右转。
            # steering = np.deg2rad(vm_data["steering"])
            steering = steer_data["value"]

            location = self.scene2map[ref_scene['name']]
            # 新加坡是左侧通行，原代码会翻转 steering 符号，以统一左右转方向定义。
            flip_flag = True if location.startswith('singapore') else False
            if flip_flag:
                steering *= -1
            # 用方向盘转角近似曲率，2.588 可理解为车辆轴距相关常数；保持和原实现一致。
            Kappa = 2 * steering / 2.588
        except: # self.nusc_can.can_blacklist: some scenes does not have vehicle monitor data
            # 某些 scene 没有 CAN bus 数据，原实现使用固定速度 + 直行曲率作为 fallback。
            v0 = 6.6
            Kappa = 0

        # 初始局部坐标方向：
        # - T0: tangent/front，车辆前向；这里 y 轴表示前方；
        # - N0: normal/side，车辆侧向；根据曲率符号选择左右侧法向。
        T0 = np.array([0.0, 1.0])  # define front
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])  # define side

        # 构造细粒度时间轴。
        # 例如 future_length=5、SAMPLE_INTERVAL=0.5 时：
        # - t_end = 5 * 0.5 = 2.5s；
        # - t_interval = 0.5 / 10 = 0.05s；
        # - tt = [0.00, 0.05, ..., 2.50]，共 51 个细粒度时间点。
        t_start = 0  # second
        t_end = future_length * SAMPLE_INTERVAL  # second
        t_interval = SAMPLE_INTERVAL / 10
        tt = np.arange(t_start, t_end + t_interval, t_interval)

        #* 根据当前速度/曲率采样 self.candidate_sample_num 条候选轨迹。
        # trajectory_sampler.sample 内部会混合直线、圆弧、clothoid 曲线等运动形态，
        # 并随机采样加速度/目标速度，使候选轨迹覆盖不同速度和转向可能性。
        # M=self.candidate_sample_num 表示候选轨迹总数；默认 1800。
        # 注意：PlanHead_v1 会把候选按三等分理解为 Left / Straight / Right，
        # 因此这里的数量必须和 plan_head.sample_num 一致，且能被 3 整除。
        sampled_trajectories_fine = trajectory_sampler.sample(
            v0, Kappa, T0, N0, tt, self.candidate_sample_num)  # sample_num, fine_time, 3

        # sampled_trajectories_fine 的时间分辨率是 0.05s，shape 约为 [sample_num, 51, 3]。
        # 每 10 个点取一次，相当于恢复到 nuScenes keyframe 的 0.5s 间隔：
        # 默认取到 [0, 0.5, 1.0, 1.5, 2.0, 2.5]，即 future_length+1 个位置点。
        sampled_trajectories = sampled_trajectories_fine[:, ::10]   # sample_num, start+future, 3

        #* 将“累计位置点”转成“相邻 step 位移”。
        # 因为包含起点 t=0，所以 future_length+1 个位置点做差后得到 future_length 段位移。
        # 例如 6 个位置点 -> 5 段位移，最终默认 shape 为 [sample_num, 5, 3]。
        sampled_trajectories = sampled_trajectories[:, 1:] - sampled_trajectories[:, :-1]  # sample_num, future, 3
        return sampled_trajectories

    def get_data_info(self, index):
        """Also return lidar2ego transformations."""
        # 轻量组装：基础相机/标定信息由父类从 PKL 读取，这里补充本项目需要的字段。
        #! lidar2ego、lidar_token、vel_steering 当前本就来自 PKL，不依赖训练时 SDK 查询。
        input_dict = super().get_data_info(index)

        info = self.data_infos[index]

        input_dict.update(dict(
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            cam2img=input_dict['cam_intrinsic'],
            lidar_token=info['lidar_token'],
            vel_steering=info['vel_steering'],        # 1,4   vx(m/s),vy(m/s),v_yaw(rad/s),steering
        ))
        return input_dict

    def get_lidar_pose(self, rec):
        '''
        Get global poses for following bbox transforming
        '''
        # 取样预处理：由 PKL 内 ego2global 位姿计算 global -> ego/LiDAR 变换。
        #! 变换后的 translation、rotation 可离线保存，但保留原始位姿通常更灵活且占用更小。
        ego2global_translation = rec['ego2global_translation']
        ego2global_rotation = rec['ego2global_rotation']
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse

        return trans, rot

    def get_ego2lidar_pose(self, rec):
        '''
        Get LiDAR poses in ego system
        '''
        # 取样预处理：由 PKL 内 lidar2ego 标定计算其逆变换。
        #! ego2lidar 也可预计算进 PKL；若下游仍需原始标定，建议同时保留 lidar2ego。
        lidar2ego_translation = rec['lidar2ego_translation']
        lidar2ego_rotation = rec['lidar2ego_rotation']
        trans = -np.array(lidar2ego_translation)
        rot = Quaternion(lidar2ego_rotation).inverse
        return trans, rot

    def record_instance(self, idx, instance_map):
        """
        Record information about each visible instance in the sequence and assign a unique ID to it

        中文说明：
            这个函数并不是为“idx 这一帧”单独返回一个完整结果，而是在当前参考帧
            的 history/current/future 时序窗口构造过程中，被循环调用多次。

            每调用一次，它处理窗口中的一帧 ``idx``：
              1. 从 self.data_infos[idx] 读取该帧的 scene/lidar/ego pose 信息；
              2. 通过 nuScenes SDK 查询该帧所有 sample_annotation；
              3. 按 self.classes 过滤需要建模的动态目标类别；
              4. 给跨帧出现的同一个 instance_token 分配稳定的 instance_id；
              5. 把该实例在当前窗口 timestep 上的 3D 状态追加到
                 self.instance_dict 中。

            self.instance_dict 里不只是保存 instance id，也保存了构造动态
            occupancy/flow 需要的目标 3D 信息。每个 instance_token 对应一条记录：
              - timestep: 该目标出现在当前窗口的哪些相对时刻；
              - translation: 该目标每个 timestep 的 3D 中心位置；
              - rotation: 该目标每个 timestep 的 3D 朝向四元数；
              - size: 该目标 3D box 尺寸，通常是 w/l/h；
              - instance_id: 当前样本内部连续实例 ID；
              - semantic_id: 目标语义类别 ID；
              - attribute_label: 可见性/属性标签，这里来自 visibility_token。

            因此它收集的是“窗口内每一帧可见/需要建模的周围动态目标”的
            类别 + 实例 ID + 3D 框几何/姿态 + 可见性，而不仅仅是 ID。

            因此，self.instance_dict 是“当前参考帧样本”的窗口级实例轨迹字典，
            收集范围由外层 ``cur_index_list`` 决定：
                [index - queue_length, ..., index, ..., index + future_length]

            这意味着当前实现会在训练取样时在线查 annotation 并动态构造
            instance_dict；并不是每个未来帧已经提前有完整 instance_dict。
            如果后续在 PKL 中预存每帧 sample_annotations，这里可以改为直接从
            rec['sample_annotations'] 读取，从而去掉对 self.nusc.get(...) 的依赖。
        """
        # 取样预处理：查询当前帧的 sample_annotation，筛选目标类别并建立跨帧实例 ID。
        #! 每帧 annotation 的 token、类别、位姿、尺寸和可见度可先写入 PKL；进一步还可按时序窗口
        #! 预生成 instance_dict/instance_map，但后者会依赖 queue_length、future_length 和类别配置。
        rec = self.data_infos[idx]
        # 记录窗口中这一帧的 scene/lidar token；LoadOccupancy 后续会用这些 token
        # 定位不同时间戳的 occupancy 文件或相关标注。
        self.scene_token.append(rec['scene_token'])
        self.lidar_token.append(rec['lidar_token'])
        # 记录这一帧 LiDAR 在世界/ego 坐标中的位姿，用于把不同时刻的 occupancy/实例
        # 统一变换到当前参考帧坐标系。
        translation, rotation = self.get_lidar_pose(rec)
        self.egopose_list.append([translation, rotation])
        ego2lidar_translation, ego2lidar_rotation = self.get_ego2lidar_pose(rec)
        self.ego2lidar_list.append([ego2lidar_translation, ego2lidar_rotation])

        # 在线查询这一帧的 sample，再遍历该 sample 下所有 3D annotation。
        #! 若 gen_new_data.py 已把每帧 annotation 写入 PKL，可用 rec['sample_annotations']
        #! 替代下面 self.nusc.get('sample') / self.nusc.get('sample_annotation')。
        current_sample = self.nusc.get('sample', rec['token'])
        for annotation_token in current_sample['anns']:
            annotation = self.nusc.get('sample_annotation', annotation_token)
            # Instance extraction for Cam4DOcc-V1
            # Filter out all non vehicle instances
            # if 'vehicle' not in annotation['category_name']:
            #     continue
            gmo_flag = False
            for class_name in self.classes:
                if class_name in annotation['category_name']:
                    gmo_flag = True
                    break
            if not gmo_flag:
                continue
            # Specify semantic id if use_separate_classes
            # semantic_id 控制 fine occupancy/instance 中的目标类别：
            # - use_separate_classes=False: 所有动态目标合并为 1 类；
            # - use_separate_classes=True : 按 bicycle/bus/car/... 细分类别赋不同 ID。
            semantic_id = 1
            if self.use_separate_classes:
                if 'bicycle' in annotation['category_name']:
                    semantic_id = 1
                elif 'bus'  in annotation['category_name']:
                    semantic_id = 2
                elif 'car'  in annotation['category_name']:
                    semantic_id = 3
                elif 'construction'  in annotation['category_name']:
                    semantic_id = 4
                elif 'motorcycle'  in annotation['category_name']:
                    semantic_id = 5
                elif 'trailer'  in annotation['category_name']:
                    semantic_id = 6
                elif 'truck'  in annotation['category_name']:
                    semantic_id = 7
                elif 'pedestrian'  in annotation['category_name']:
                    semantic_id = 8

            # Filter out invisible vehicles
            FILTER_INVISIBLE_VEHICLES = True
            if FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and annotation['instance_token'] not in self.visible_instance_set:
                continue
            # Filter out vehicles that have not been seen in the past
            # 当前窗口进入未来段后，如果某个 instance 在历史/当前从未出现过，
            # 则不把它加入当前样本的 instance_dict，避免模型需要预测“突然凭空出现”的目标。
            if self.counter >= (self.queue_length+1) and annotation['instance_token'] not in self.visible_instance_set:
                continue
            self.visible_instance_set.add(annotation['instance_token'])

            # instance_map 把 nuScenes 全局 instance_token 映射成当前样本内部连续 ID。
            # 同一个 instance_token 跨多帧会复用同一个 instance_id。
            if annotation['instance_token'] not in instance_map:
                instance_map[annotation['instance_token']] = len(instance_map) + 1  # instance_map={'instance_token': instance_id}
            instance_id = instance_map[annotation['instance_token']]
            instance_attribute = int(annotation['visibility_token'])

            if annotation['instance_token'] not in self.instance_dict:
                # 该实例第一次出现在当前窗口中：创建一条实例轨迹记录。
                # timestep 使用 self.counter，表示它在当前窗口中的相对时刻，
                # 不是全局帧号；0 通常对应窗口最早历史帧。
                # 这条记录包含的不只是 id：
                # - translation/rotation/size 共同描述该目标当前 3D box；
                # - semantic_id 描述类别；
                # - attribute_label 记录可见性；
                # 后续 LoadOccupancy/flow 相关逻辑可利用这些时序 3D box 信息
                # 生成/修正细粒度动态 occupancy 与实例运动。
                self.instance_dict[annotation['instance_token']] = {
                    'timestep': [self.counter],                    # 当前窗口内的相对时间步，如 0/1/2/...
                    'translation': [annotation['translation']],     # 3D box 中心位置 [x, y, z]，nuScenes 全局坐标
                    'rotation': [annotation['rotation']],           # 3D box 朝向四元数 [w, x, y, z]
                    'size': annotation['size'],                     # 3D box 尺寸 [w, l, h]
                    'instance_id': instance_id,                     # 当前样本内部连续实例 ID
                    'semantic_id': semantic_id,                     # 动态目标类别 ID，是否细分由 use_separate_classes 控制
                    'attribute_label': [instance_attribute],         # 可见性标签，来自 visibility_token
                }
            else:
                # 该实例之前已经在窗口内出现过：追加它在当前 timestep 的状态，
                # 从而形成跨历史/当前/未来的 instance 轨迹。
                self.instance_dict[annotation['instance_token']]['timestep'].append(self.counter)
                self.instance_dict[annotation['instance_token']]['translation'].append(annotation['translation'])
                self.instance_dict[annotation['instance_token']]['rotation'].append(annotation['rotation'])
                self.instance_dict[annotation['instance_token']]['attribute_label'].append(instance_attribute)

        return instance_map

    @staticmethod
    def _check_consistency(translation, prev_translation, threshold=1.0):
        """
        Check for significant displacement of the instance adjacent moments
        """
        x, y = translation[:2]
        prev_x, prev_y = prev_translation[:2]

        if abs(x - prev_x) > threshold or abs(y - prev_y) > threshold:
            return False
        return True

    def refine_instance_poly(self, instance):
        """
        Fix the missing frames and disturbances of ground truth caused by noise
        """
        # 取样预处理：补齐实例缺失帧，并抑制相邻帧标注位置的异常跳变。
        #! 若时序窗口配置固定，可将修正后的 instance_dict 离线写入 PKL 或独立标注文件。
        pointer = 1
        for i in range(instance['timestep'][0] + 1, self.queue_length+1+self.future_length):
            # Fill in the missing frames
            if i not in instance['timestep']:
                instance['timestep'].insert(pointer, i)
                instance['translation'].insert(pointer, instance['translation'][pointer-1])
                instance['rotation'].insert(pointer, instance['rotation'][pointer-1])
                instance['attribute_label'].insert(pointer, instance['attribute_label'][pointer-1])
                pointer += 1
                continue

            # Eliminate observation disturbances
            if self._check_consistency(instance['translation'][pointer], instance['translation'][pointer-1]):
                instance['translation'][pointer] = instance['translation'][pointer-1]
                instance['rotation'][pointer] = instance['rotation'][pointer-1]
                instance['attribute_label'][pointer] = instance['attribute_label'][pointer-1]
            pointer += 1

        return instance

    def _prepare_data_info_single(self, index, occ_load_flag=None, aug_param=None):
        #* ================== 3. 单帧组装 + pipeline ==================
        # 单帧预处理入口：
        # 1) get_data_info 从 pkl 中取相机路径、lidar2img、pose、can_bus、3D 框等基础字段；
        # 2) occ_load_flag=True 只出现在当前参考帧，用来额外准备未来框、规划、实例和 occupancy 所需索引；
        # 3) pre_pipeline/pipeline 执行 config 中的数据增强、图像归一化、LoadOccupancy 和 Collect。
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        if aug_param is not None:
            input_dict['aug_param'] = copy.deepcopy(aug_param)

        # only load current frame
        if occ_load_flag is not None:
            input_dict['occ_load_flag'] = occ_load_flag
        # 当前参考帧才会走下面这些重监督构造；历史/未来帧通常只需要图像和 meta。
        # LoadOccupancy 虽然在 pipeline 里执行，但它依赖这里写入的 scene_token_list、
        # lidar_token_list、egopose_list、ego2lidar_list 来读取并对齐整段 fine occupancy。
        # 训练时预处理：生成规划损失和碰撞指标使用的未来框及 BEV 占用 mask。
        #! 可改为从 info['gt_future_boxes']、info['segmentation_bev'] 直接读取。
        if occ_load_flag:   # only load current frame
            gt_future_boxes, segmentation_bev = self.get_future_bboxes(index)
            input_dict.update(
                gt_future_boxes=gt_future_boxes,
                segmentation_bev=segmentation_bev,
            )
        # 训练时预处理：在线生成自车规划标签、驾驶命令与候选轨迹。
        #! 可改为从 PKL 的 sdc_planning、sdc_planning_mask、command、sample_traj 字段直接读取。
        if occ_load_flag:
            # sdc_plan
            info = self.data_infos[index]
            sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(info['token'])
            sample_traj = self.get_trajectory_sampling(info, self.future_length+1)
            input_dict.update(
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command,
                sample_traj=sample_traj,
            )
        # 训练时预处理：聚合历史到未来窗口内的实例，并修补实例时序。
        #! inflated occupancy/flow 可直接读取离线 instance_dict；fine-grained 且不使用实例监督时可跳过此段。
        if occ_load_flag:    # only load current frame
            # 这里收集的窗口长度是 history + current + future：
            # [index - queue_length, ..., index, ..., index + future_length]
            # 这些 token/pose 会被 LoadOccupancy 用来把各时刻 occupancy 统一变换到当前参考帧。
            cur_index_list = list(range(index-self.queue_length, index + (self.future_length + 1)))
            self.scene_token = []
            self.lidar_token = []
            self.egopose_list = []
            self.ego2lidar_list = []
            self.visible_instance_set = set()
            self.instance_dict = {}
            instance_map = {}
            # load annotation to instance_dict
            for self.counter, index_t in enumerate(cur_index_list):
                instance_map = self.record_instance(index_t, instance_map)
            # fix missing
            for token in self.instance_dict.keys():
                self.instance_dict[token] = self.refine_instance_poly(self.instance_dict[token])
            # update input_dict
            input_dict.update(
                use_fine_occ=self.use_fine_occ,
                scene_token_list=self.scene_token,
                lidar_token_list=self.lidar_token,
                egopose_list=self.egopose_list,
                ego2lidar_list=self.ego2lidar_list,
                instance_dict=self.instance_dict,
                instance_map=instance_map,
            )

        self.pre_pipeline(input_dict)
        # config pipeline 在这里真正执行：
        # - LoadMultiViewImageFromFiles 读取 6 路图像；
        # - 训练阶段 PhotoMetric/CropResizeFlip 做图像增强，并把同一 aug_param 传给历史帧；
        # - Normalize/Pad 规范图像张量尺寸；
        # - LoadOccupancy 在当前参考帧读取并对齐 occupancy 序列；
        # - Format/Collect 把模型需要的 key 和 img_metas 打包成 DataContainer。
        example = self.pipeline(input_dict)
        return example

    def _prepare_data_info(self, index, rand_interval=None):
        """
        Modified from BEVFormer:CustomNuScenesDataset,
            BEVFormer logits: randomly select (queue_length-1) previous images.
            Modified logits: directly select (queue_length) previous images.
        """
        # 在线时序采样：随机选择帧间隔并组装历史/未来队列；这是数据增强的一部分。
        #! 通常不应固化成单一 PKL 结果，否则会丢失 rand_frame_interval 带来的随机时序增强。
        rand_interval = (
            rand_interval if rand_interval is not None else
            np.random.choice(self.rand_frame_interval, 1)[0]
        )

        #* ================== 2. 取一个训练/测试样本 ==================
        # 根据配置取 history/current/future：
        # - previous_queue = queue_length 帧历史 + 当前帧，用于图像输入和 BEV temporal memory；
        # - future_queue = 当前帧 + future_length 帧未来，用于 occupancy 监督和 action condition；
        # - 当前配置中 queue_length=2、future_length=4，即 2 帧历史 + 当前帧 + 4 帧未来。
        # 1. get previous camera information.
        previous_queue = [] # history_len*['img', 'points', 'aug_param']
        previous_index_list = list(range(
            index - self.queue_length * rand_interval, index, rand_interval))
        previous_index_list = sorted(previous_index_list)
        if rand_interval < 0:  # the inverse chain.
            previous_index_list = previous_index_list[::-1]
        previous_index_list.append(index)
        aug_param = None
        for i, idx in enumerate(previous_index_list):
            idx = min(max(0, idx), len(self.data_infos) - 1)

            occ_load_flag = True if i==self.queue_length else False     # only load current frame

            example = self._prepare_data_info_single(idx, occ_load_flag, aug_param=aug_param)

            aug_param = copy.deepcopy(example['aug_param']) if 'aug_param' in example else None
            if example is None:
                return None
            previous_queue.append(example)

        # 2. get future occ information.
        future_queue = []
        # Future: from current to future frames.
        # use current frame as the 0-th future.
        future_index_list = list(range(
            index, index + (self.future_length + 1) * rand_interval, rand_interval))
        future_index_list = sorted(future_index_list)
        if rand_interval < 0:  # the inverse chain.
            future_index_list = future_index_list[::-1]
        has_future = False
        for i, idx in enumerate(future_index_list):
            idx = min(max(0, idx), len(self.data_infos) - 1)

            occ_load_flag = False

            example = self._prepare_data_info_single(idx, occ_load_flag)
            if example is None and not has_future:
                return None
            future_queue.append(example)
            has_future = True
        return self.union2one(previous_queue, future_queue)

    def union2one(self, previous_queue, future_queue):
        pass

    def evaluate(self, results, logger=None, **kawrgs):
        '''
        Evaluate by IOU and VPQ metrics for model evaluation
        '''
        eval_results = {}

        ''' calculate IOU of current and future frames'''
        if 'hist_for_iou' in results.keys():
            IoU_results_current_future = {}
            hist_for_iou = sum(results['hist_for_iou'])
            ious = cm_to_ious(hist_for_iou)
            res_table, res_dic = format_iou_results(ious, return_dic=True)
            for key, val in res_dic.items():
                IoU_results_current_future['IOU_{}'.format(key)] = val
            if logger is not None:
                logger.info('IOU Evaluation of current and future frames:')
                logger.info(res_table)
            eval_results.update(IoU_of_Current_Future=IoU_results_current_future)

        ''' calculate IOU of current frame'''
        if 'hist_for_iou_current' in results.keys():
            IoU_results_current = {}
            hist_for_iou = sum(results['hist_for_iou_current'])
            ious = cm_to_ious(hist_for_iou)
            res_table, res_dic = format_iou_results(ious, return_dic=True)
            for key, val in res_dic.items():
                IoU_results_current['IOU_{}'.format(key)] = val
            if logger is not None:
                logger.info('IOU Evaluation of current frame:')
                logger.info(res_table)
            eval_results.update(IoU_of_Current=IoU_results_current)

        ''' calculate IOU of future frame'''
        if 'hist_for_iou_future' in results.keys():
            IoU_results_future = {}
            hist_for_iou = sum(results['hist_for_iou_future'])
            ious = cm_to_ious(hist_for_iou)
            res_table, res_dic = format_iou_results(ious, return_dic=True)
            for key, val in res_dic.items():
                IoU_results_future['IOU_{}'.format(key)] = val
            if logger is not None:
                logger.info('IOU Evaluation of future frames:')
                logger.info(res_table)
            eval_results.update(IoU_of_Future=IoU_results_future)

        ''' calculate IOU of future frame with time_weighting'''
        if 'hist_for_iou_future_time_weighting' in results.keys():
            IoU_results_future_time_weighting = {}
            hist_for_iou = sum(results['hist_for_iou_future_time_weighting'])
            ious = cm_to_ious(hist_for_iou)
            res_table, res_dic = format_iou_results(ious, return_dic=True)
            for key, val in res_dic.items():
                IoU_results_future_time_weighting['IOU_{}'.format(key)] = val
            if logger is not None:
                logger.info('IOU Evaluation of future frames with time weighting:')
                logger.info(res_table)
            eval_results.update(IoU_of_Future_with_Time_Weighting=IoU_results_future_time_weighting)

        ''' calculate VPQ '''
        if 'vpq_metric' in results.keys() and 'vpq_len' in results.keys():
            vpq_sum = sum(results['vpq_metric'])
            eval_results['VPQ'] = vpq_sum/results['vpq_len']

        ''' calculate plan_metric '''
        if 'plan_metric' in results.keys() and 'data_len' in results.keys():
            for key, value in results['plan_metric'].items():
                eval_results[key] = sum(value)/results['data_len']
            eval_results.update(avg_l2=(eval_results['plan_L2_1s']+eval_results['plan_L2_2s']+eval_results['plan_L2_3s'])/3)
            eval_results.update(avg_obj_col=(eval_results['plan_obj_col_1s']+eval_results['plan_obj_col_2s']+eval_results['plan_obj_col_3s'])/3)
            eval_results.update(avg_obj_box_col=(eval_results['plan_obj_box_col_1s']+eval_results['plan_obj_box_col_2s']+eval_results['plan_obj_box_col_3s'])/3)
            eval_results.update(avg_obj_box_col_single=(eval_results['plan_obj_box_col_1s_single']+eval_results['plan_obj_box_col_2s_single']+eval_results['plan_obj_box_col_3s_single'])/3)
            eval_results.update(avg_obj_col_single=(eval_results['plan_obj_col_1s_single']+eval_results['plan_obj_col_2s_single']+eval_results['plan_obj_col_3s_single'])/3)
            eval_results.update(avg_l2_single=(eval_results['plan_L2_1s_single']+eval_results['plan_L2_2s_single']+eval_results['plan_L2_3s_single'])/3)

        if 'planning_results_computed' in results.keys():
            planning_results_computed = results['planning_results_computed']
            num_frames = len(planning_results_computed['L2'])
            planning_tab = PrettyTable()
            planning_tab.field_names = [
                "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"][:num_frames+1]
            for key in planning_results_computed.keys():
                value = planning_results_computed[key]  # Lout
                row_value = []
                row_value.append(key)
                for i in range(len(value)):
                    row_value.append('%.4f' % float(value[i]))
                planning_tab.add_row(row_value)
            print(planning_tab)

        return eval_results

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        rand_interval = None
        while True:
            data = self._prepare_data_info(
                self.usable_index[idx], rand_interval=rand_interval)
            if data is None:
                if self.test_mode:
                    idx += 1
                else:
                    if rand_interval is None:
                        rand_interval = 1  # use rand_interval = 1 for the same sample again.
                    else:  # still None for rand_interval = 1, no enough future.
                        idx = self._rand_another(idx)
                        rand_interval = None
                continue
            assert data is not None
            return data

    def __len__(self):
        return len(self.usable_index)
