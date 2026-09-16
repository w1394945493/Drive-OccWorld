import copy
from .nuscenes_dataset import CustomNuScenesDataset
import mmcv
from mmdet.datasets import DATASETS
import numpy as np
import cv2
import torch
from pyquaternion import Quaternion
from projects.mmdet3d_plugin.datasets.formating import cm_to_ious, format_iou_results
from projects.mmdet3d_plugin.bevformer.dense_heads.plan_head import calculate_birds_eye_view_parameters
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from prettytable import PrettyTable


@DATASETS.register_module()
class NuScenesWorldDatasetTemplateOffline(CustomNuScenesDataset):
    r"""Offline-PKL version of world dataset for visual point cloud forecasting.

    整体数据链路：
    1. 父类从 ann_file 指向的 ``nuscenes_infos_temporal_*.pkl`` 读取 data_infos，
       其中包含 sample token、相机路径、标定、ego/lidar 位姿、3D 框、速度等基础信息。
    2. 本离线版要求 ann_file 已由 ``tools/gen_new_data.py`` 增强，逐帧写入
       scene/location、规划标签、候选轨迹和 sample_annotations。
    3. ``_prepare_data_info`` 按 queue_length/future_length 取历史帧、当前帧和未来帧；
       其中当前帧作为参考帧，负责触发 occupancy、规划、未来框和实例监督的准备。
    4. ``_prepare_data_info_single`` 先用 ``get_data_info`` 组装单帧基础字段，再调用
       config 中的 pipeline 读取图像、做图像增强/归一化/pad，并通过 ``LoadOccupancy``
       从 occ_path 读取 fine-grained occupancy 标签。
    5. 子类 ``NuScenesWorldDatasetV1.union2one`` 把多帧结果合并成模型输入：历史+当前图像、
       当前参考帧坐标系下的 occupancy 序列、future can_bus、规划标签和 action condition。

    注释约定：普通 ``#`` 表示当前在数据集初始化或取样阶段执行的预处理；
    ``#!`` 表示该结果已经在离线数据转换阶段写入 PKL，训练时直接读取。

    与 ``NuScenesWorldDatasetTemplate`` 的关键区别：
    - 不初始化 NuScenes；
    - 不初始化 NuScenesCanBus；
    - 不初始化 NuScenesTraj；
    - 规划标签、候选轨迹、逐帧 annotation 均直接从 PKL 读取。
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

        # todo ================== Offline 关键区别 1：不再初始化 3 个 nuScenes SDK 对象 ==================
        # 旧版 NuScenesWorldDatasetTemplate 会在这里实例化：
        #   self.nusc = NuScenes(...)
        #   self.nusc_can = NuScenesCanBus(...)
        #   self.traj_api = NuScenesTraj(...)
        # 离线版要求这些信息已由 tools/gen_new_data.py 写入 PKL，因此训练阶段不再依赖：
        #   - NuScenes：原来用于查 scene/log/sample/sample_annotation；
        #   - NuScenesCanBus：原来用于查 CAN bus pose/steering 并采样 sample_traj；
        #   - NuScenesTraj：原来用于在线生成 sdc_planning / mask / command。
        self.required_offline_keys = (
            'scene_name', 'location',
            'sdc_planning', 'sdc_planning_mask', 'command',
            'sample_traj', 'sample_annotations')
        self._check_offline_pkl_fields()

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

    def _check_offline_pkl_fields(self):
        """检查增强版 PKL 是否包含离线 Dataset 需要的字段。"""
        if len(self.data_infos) == 0:
            return
        missing_keys = [
            key for key in self.required_offline_keys
            if key not in self.data_infos[0]
        ]
        if missing_keys:
            raise KeyError(
                'NuScenesWorldDatasetTemplateOffline requires an augmented PKL '
                'generated by tools/gen_new_data.py. '
                f'Missing keys in the first info: {missing_keys}')

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
        raise RuntimeError(
            'NuScenesWorldDatasetTemplateOffline does not sample trajectories '
            'online. Please generate an augmented PKL with tools/gen_new_data.py '
            "and read candidate trajectories from info['sample_traj'].")

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

            离线版不会在线查 nuScenes SDK，而是直接读取每帧 PKL 中已经保存的
            rec['sample_annotations']。滑窗聚合仍然在线进行：
                当前参考帧 index
                    -> 收集历史/当前/未来帧各自的 sample_annotations
                    -> 动态构造当前样本的 instance_dict / instance_map
        """
        # 离线取样：从当前帧 PKL 的 sample_annotations 读取 3D annotation，
        # 筛选目标类别并建立跨帧实例 ID。
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

        # todo ================== Offline 关键区别 2：逐帧 annotation 直接从 PKL 读取 ==================
        # 旧版这里会在线调用：
        #   current_sample = self.nusc.get('sample', rec['token'])
        #   annotation = self.nusc.get('sample_annotation', annotation_token)
        # 离线版中 gen_new_data.py 已把当前帧自己的 nuScenes 3D annotation 写入 PKL，
        # 所以这里直接遍历 rec['sample_annotations']。
        # 滑窗逻辑仍然保留：Dataset 仍会按当前参考帧收集历史/当前/未来多帧，
        # 只是每一帧的目标信息不再在线查 SDK，而是从 PKL 取。
        for annotation in rec['sample_annotations']:
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
        # todo ================== Offline 关键区别 3：规划标签和候选轨迹直接从 PKL 读取 ==================
        # 旧版这里会在线调用：
        #   self.traj_api.get_sdc_planning_label(info['token'])
        #   self.get_trajectory_sampling(info, self.future_length + 1)
        # 离线版直接读取 tools/gen_new_data.py 写入 PKL 的字段：
        # - sdc_planning / sdc_planning_mask / command 来自 NuScenesTraj 离线预计算；
        # - sample_traj 来自 CAN bus pose/steering 离线采样。
        if occ_load_flag:
            info = self.data_infos[index]
            input_dict.update(
                sdc_planning=info['sdc_planning'],
                sdc_planning_mask=info['sdc_planning_mask'],
                command=info['command'],
                sample_traj=info['sample_traj'],
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
