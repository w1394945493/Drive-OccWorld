"""把 Drive-OccWorld 训练时在线查询的标签预先写入 nuScenes info PKL。

这个脚本的用途：
    读取已有 ``nuscenes_infos_temporal_*_new.pkl``，给每帧 info 额外补充
    Drive-OccWorld 数据集类当前需要在线调用 NuScenes/NuScenesCanBus/NuScenesTraj
    才能得到的字段。

会写入的字段：
    - location: scene 对应地图位置，原来从 NuScenes log 表查询。
    - scene_name: scene token 对应的 scene 名称。
    - sdc_planning / sdc_planning_mask / command:
      原来由 NuScenesTraj.get_sdc_planning_label() 在线生成。
    - sample_traj:
      原来由 NuScenesCanBus 的 pose/steering 消息在线采样生成。
    - sample_annotations:
      当前帧自己的 nuScenes 3D annotation 信息。后续 Dataset 做滑窗时，
      可以直接从历史/当前/未来帧的 info['sample_annotations'] 收集目标，
      不必再在线调用 NuScenes.get('sample') / get('sample_annotation')。

这样做的好处：
    后续可以把 Dataset 改成优先读取 ``info[...]``，字段缺失时才 fallback 到 SDK。
    这样训练阶段就不用每次初始化 NuScenes / NuScenesCanBus / NuScenesTraj 去查这些信息。

这里的 SDK 指 Software Development Kit，中文可理解为“官方软件开发工具包”。
在本脚本里它不是模型，也不是新数据格式，而是 nuScenes 官方封装好的 Python
读取/查询接口：帮我们从原始 json 表、CAN bus 文件和 token 关系里查到所需字段。
"""

import argparse
import copy
import os
import sys

import mmcv
import numpy as np
from mmdet3d.core.bbox import Box3DMode
from nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from tqdm import tqdm


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from projects.mmdet3d_plugin.datasets.samplers import sampler as trajectory_sampler  # noqa: E402
from projects.mmdet3d_plugin.datasets.trajectory_api import NuScenesTraj  # noqa: E402


# 类别列表需要和 projects/configs/fine_grained 中的 class_names 保持一致。
# NuScenesTraj 内部生成 SDC box/label 时会用到这个类别顺序。
CLASS_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'construction',
    'motorcycle', 'pedestrian', 'trafficcone', 'trailer',
    'truck', 'driveable_surface', 'other', 'sidewalk',
    'terrain', 'mannade', 'vegetation',
]


DEFAULT_TRAIN_ANN_FILE = (
    '/c20250502/wangyushen/Weights/drive-occworld/'
    'nuscenes_infos_temporal_train_new.pkl'
)
DEFAULT_VAL_ANN_FILE = (
    '/c20250502/wangyushen/Weights/drive-occworld/'
    'nuscenes_infos_temporal_val_new.pkl'
)
DEFAULT_NEW_TRAIN_ANN_FILE = (
    '/c20250502/wangyushen/Weights/drive-occworld/'
    'nuscenes_infos_temporal_train_new_v2.pkl'
)
DEFAULT_NEW_VAL_ANN_FILE = (
    '/c20250502/wangyushen/Weights/drive-occworld/'
    'nuscenes_infos_temporal_val_new_v2.pkl'
)
DEFAULT_NUSC_ROOT = '/c20250502/wangyushen/Datasets/NuScenes/v1.0-trainval/'
DEFAULT_CAN_BUS_ROOT = '/c20250502/wangyushen/Datasets/NuScenes'


def build_scene_maps(nusc):
    """一次性建立 scene 查表，替代 Dataset 初始化时反复遍历 nusc.scene。

    Returns:
        scene_token_to_name: scene_token -> scene_name
        scene_name_to_location: scene_name -> log['location']
    """
    scene_token_to_name = {}
    scene_name_to_location = {}
    for scene in nusc.scene:
        scene_token_to_name[scene['token']] = scene['name']
        log = nusc.get('log', scene['log_token'])
        scene_name_to_location[scene['name']] = log['location']
    return scene_token_to_name, scene_name_to_location


def get_sample_annotations(nusc, rec):
    """离线读取当前帧自己的 nuScenes sample_annotation。

    这一步对应 Dataset 中 ``record_instance`` 原本在线执行的两类查询：
        current_sample = self.nusc.get('sample', rec['token'])
        annotation = self.nusc.get('sample_annotation', annotation_token)

    注意这里刻意只保存“当前帧自己的 annotation”，不提前构造窗口级
    instance_dict。原因是 history/current/future 滑窗范围仍应由在线 Dataset
    根据 queue_length / future_length 动态决定：
        当前参考帧 index
            -> 收集多个 info['sample_annotations']
            -> 动态构造当前样本的 instance_dict / instance_map

    Returns:
        sample_annotations: list[dict]。每个元素是一条 3D annotation，字段尽量保持
        接近 nuScenes 原始 annotation，便于 Dataset 后续按类别、可见性和
        instance_token 继续处理。
    """
    current_sample = nusc.get('sample', rec['token'])
    sample_annotations = []
    for annotation_token in current_sample['anns']:
        annotation = nusc.get('sample_annotation', annotation_token)
        sample_annotations.append({
            # annotation token 本身，便于调试或和 nuScenes 原表对齐。
            'token': annotation['token'],
            # instance_token 用来跨历史/当前/未来帧关联同一个目标。
            'instance_token': annotation['instance_token'],
            # category_name 保留原始细分类别；Dataset 可根据 use_separate_classes
            # 决定是细分 semantic_id，还是统一合并成 1 类。
            'category_name': annotation['category_name'],
            # 3D box 几何信息。record_instance 后续会把它们聚合进 instance_dict。
            'translation': annotation['translation'],
            'rotation': annotation['rotation'],
            'size': annotation['size'],
            # 可见性标签；原 record_instance 中作为 attribute_label 使用。
            'visibility_token': annotation['visibility_token'],
        })
    return sample_annotations


def get_trajectory_sampling(nusc_can, scene_token_to_name,
                            scene_name_to_location, rec, future_length,
                            sample_interval=0.5):
    """离线版 get_trajectory_sampling。

    对应 Dataset 中的 ``NuScenesWorldDatasetTemplate.get_trajectory_sampling``：
    根据当前帧 timestamp 对齐 CAN bus 的 pose 和 steering，得到/估计当前
    自车运动状态，然后从这个“当前状态”出发，随机采样一批未来 ego trajectory
    候选。

    更具体地说：
      - CAN bus 的 pose 消息提供当前帧附近的自车速度 vel，这里只取
        pose_data['vel'][0] 作为纵向初速度 v0；
      - CAN bus 的 steeranglefeedback 消息提供当前帧附近的方向盘转角 steering；
      - 当前曲率 kappa 不是直接从 CAN bus 读出的字段，而是用 steering
        近似换算得到：kappa = 2 * steering / 2.588；
      - 未来加速度、未来目标速度也不是 CAN bus 直接给的 GT，而是在
        trajectory_sampler.sample 里随机采样，用来扩展出多种可能动作。

    这里的 sample_traj 不是 nuScenes 已经发生的未来 GT 轨迹，而是 planner
    用来评估的“候选动作/候选轨迹”：
      1. 先读取当前帧附近的 CAN bus 速度和方向盘转角；
      2. 由方向盘转角近似得到当前曲率 kappa；
      3. sampler 再随机采样加速度/目标速度，生成多条可能的运动曲线；
      4. 默认采样 1800 条候选轨迹，原始在线 Dataset 注释为：
         [720, 360, 720] = Left / Straight / Right；
      5. 每条候选轨迹最后表示为 future_length 个 0.5s 相邻位移 step。

    Args:
        rec: 当前帧 info。
        future_length: 输出的未来相邻位移 step 数。当前在线 Dataset 中常见
            配置 future_length=4，但调用本函数时传入 future_length+1，因此
            默认脚本用 5。也就是说默认会生成 5 个未来 0.5s step，覆盖
            2.5s 的候选轨迹。
        sample_interval: nuScenes 关键帧间隔，默认 0.5s。

    Returns:
        sampled: shape 约为 [1800, future_length, 3]。
            - 1800: 候选轨迹条数；
            - future_length: 未来 step 数，默认 5；
            - 3: 每个 step 的相邻增量，通常可理解为 dx/dy/dyaw。
    """
    try:
        # 通过当前帧 scene_token 找到 scene_name，再从 CAN bus 中取该 scene 的消息。
        # CAN bus 是车载控制器局域网数据；这里不是读未来 GT，而是读当前帧附近的
        # 车辆状态/控制反馈，用作候选轨迹采样的初始条件。
        scene_name = scene_token_to_name[rec['scene_token']]
        # pose 消息中有自车状态，这里主要使用 vel[0] 作为当前纵向速度。
        pose_msgs = nusc_can.get_messages(scene_name, 'pose')
        pose_uts = [msg['utime'] for msg in pose_msgs]
        # steeranglefeedback 消息中有方向盘转角反馈，用于近似当前曲率。
        steer_msgs = nusc_can.get_messages(scene_name, 'steeranglefeedback')
        steer_uts = [msg['utime'] for msg in steer_msgs]

        ref_utime = rec['timestamp']
        # 找到距离当前 sample timestamp 最近的 CAN bus pose/steering 消息。
        # “最近”表示时间戳最接近当前 keyframe；因此它表示当前帧附近状态，
        # 不是过去一段历史序列，也不是未来实际发生的动作序列。
        pose_index = trajectory_sampler.locate_message(pose_uts, ref_utime)
        pose_data = pose_msgs[pose_index]
        steer_index = trajectory_sampler.locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]

        # 当前纵向初速度 v0，单位 m/s；这是 CAN bus pose 直接提供的当前速度分量。
        v0 = pose_data['vel'][0]

        # 当前方向盘转角 steering，来自 CAN bus steeranglefeedback。
        steering = steer_data['value']
        location = scene_name_to_location[scene_name]
        # 新加坡是左侧通行，原代码会翻转 steering 符号，以统一左右转方向定义。
        if location.startswith('singapore'):
            steering *= -1
        # 当前曲率 kappa 不是 CAN bus 直接提供的 GT 字段，而是由方向盘转角近似换算。
        # 未来加速度/未来速度也不会在这里读取，而是在 sampler 中随机采样。
        kappa = 2 * steering / 2.588
    except Exception:
        # 某些 scene 没有 CAN bus 数据，保持和在线 Dataset 一致的 fallback。
        v0 = 6.6
        kappa = 0

    # T0/N0 定义采样轨迹的初始局部坐标方向：
    # - t0: tangent/front，车辆前向；这里 y 轴是前方；
    # - n0: normal/side，车辆侧向；根据曲率符号决定左右侧法向。
    t0 = np.array([0.0, 1.0])
    n0 = np.array([1.0, 0.0]) if kappa <= 0 else np.array([-1.0, 0.0])

    # 构造细粒度采样时间轴。
    # 默认 future_length=5、sample_interval=0.5 时：
    # - t_end = 5 * 0.5 = 2.5s；
    # - t_interval = 0.5 / 10 = 0.05s；
    # - ts = [0.00, 0.05, ..., 2.50]，共 51 个细粒度时间点。
    t_start = 0
    t_end = future_length * sample_interval
    t_interval = sample_interval / 10
    ts = np.arange(t_start, t_end + t_interval, t_interval)

    # *=======================================#
    #* 根据当前速度/曲率采样 1800 条候选轨迹。
    # trajectory_sampler.sample 内部会混合三类轨迹：
    # - straight line：直行候选；
    # - circle：近似恒曲率转弯候选；
    # - clothoid：曲率渐变的更平滑转弯候选。
    # 其中第 6 个参数 M=1800 表示候选轨迹总数；按默认概率 [0.4, 0.2, 0.4]
    # 可理解为约 720 条左转、360 条直行、720 条右转候选。
    sampled_fine = trajectory_sampler.sample(v0, kappa, t0, n0, ts, 1800)

    # sampled_fine 的时间分辨率是 0.05s，shape 约为 [1800, 51, 3]。
    # 每 10 个点取一次，相当于恢复到 nuScenes 关键帧 0.5s 间隔：
    # 默认得到 t=[0, 0.5, 1.0, 1.5, 2.0, 2.5]，也就是 6 个位置点。
    sampled = sampled_fine[:, ::10]

    #* 把“绝对/累计轨迹点”转成“相邻 step 位移”。
    # 6 个位置点做差后变成 5 段位移：
    # [p0->p1, p1->p2, p2->p3, p3->p4, p4->p5]。
    # 因此默认输出 shape 为 [1800, 5, 3]；如果 --planning-steps=N，
    # 则输出 shape 为 [1800, N, 3]。
    sampled = sampled[:, 1:] - sampled[:, :-1]
    return sampled


def augment_infos(ann_file, out_file, nusc, nusc_can, traj_api,
                  planning_steps, overwrite=False):
    """读取一个 PKL，逐帧补充离线预计算字段，再保存到新 PKL。

    本函数会给每个 info 补充以下字段：
        - scene_name: 当前帧所属 scene 的名称。
        - location: 当前 scene 对应地图位置，替代 Dataset 初始化时遍历 nusc.scene/log。
        - sdc_planning: 未来 SDC GT 轨迹，shape=[planning_steps, 3]，为 x/y/delta_yaw。
        - sdc_planning_mask: 未来 SDC GT 有效位，shape=[planning_steps, 2]。
        - command: 根据未来 GT 轨迹离散得到的 left/right/forward 命令。
        - sample_traj: 基于当前速度和方向盘转角采样的候选 ego 轨迹，不是真实未来 GT。
        - sample_annotations: 当前帧所有 nuScenes 3D annotation 的精简原始信息，
          用于后续 Dataset 在线滑窗组合 instance_dict，替代训练时 self.nusc.get(...)
          查询 sample / sample_annotation。

    这些字段对应替代 Dataset 中对 NuScenes、NuScenesCanBus、NuScenesTraj 的在线查询。
    """
    print(f'Loading PKL: {ann_file}')
    data = mmcv.load(ann_file)
    infos = data['infos']
    print(f'Loaded {len(infos)} frames. Output will be saved to: {out_file}')

    # 这两个 map 用来替代 Dataset 里 self.scene2map 的在线构建。
    scene_token_to_name, scene_name_to_location = build_scene_maps(nusc)
    new_infos = []

    # tqdm 进度条会显示当前 split/pkl、处理速度和预计剩余时间。
    progress_desc = f'augment {os.path.basename(ann_file)}'
    for info in tqdm(infos, desc=progress_desc, dynamic_ncols=True):
        # deepcopy 避免修改 mmcv.load 出来的原始对象引用。
        info = copy.deepcopy(info)
        scene_name = scene_token_to_name[info['scene_token']]

        # 写入 scene_name/location，后续 Dataset 可直接从 info 读取。
        if overwrite or 'scene_name' not in info:
            info['scene_name'] = scene_name
        if overwrite or 'location' not in info:
            info['location'] = scene_name_to_location[scene_name]

        # *==========================================================================#
        # 写入当前帧自己的 sample_annotations。
        # 这不是窗口级 instance_dict，而是“单帧原始/半原始 annotation 缓存”：
        # - gen_new_data.py 离线查 nuScenes SDK，把当前帧所有 3D annotation 写入 PKL；
        # - Dataset 在线阶段仍按当前 index 做滑窗；
        # - 滑窗中的每一帧直接读取 info['sample_annotations']；
        # - 再动态构造当前参考帧的 instance_dict / instance_map。
        # 这样既能避免训练时实例化/调用 self.nusc，又不把 queue_length/future_length 写死。
        if overwrite or 'sample_annotations' not in info:
            info['sample_annotations'] = get_sample_annotations(nusc, info)

        # *==========================================================================#
        # 写入 SDC 未来轨迹、有效 mask 和 high-level command。
        # 这对应原来的 self.traj_api.get_sdc_planning_label(info['token'])。
        # 与下面 sample_traj 不同，这里是 nuScenes 数据集中真实发生的未来自车轨迹 GT：
        #   - get_sdc_planning_label 会沿当前 sample 的 next 指针查未来关键帧；
        #   - 读取未来 ego_pose，并把未来自车 box/pose 转到“当前参考帧 LiDAR 坐标系”；
        #   - 最终得到 sdc_planning: [planning_steps, 3]，每行是 x / y / delta_yaw；
        #   - sdc_planning_mask: [planning_steps, 2]，表示对应未来 step 是否有效；
        #   - command: [planning_steps]，根据未来横向位移粗略离散成 right / left / forward。
        # 如果当前帧临近 scene 末尾，未来帧不足，mask 会标出哪些 step 无效。
        if overwrite or not all(k in info for k in (
                'sdc_planning', 'sdc_planning_mask', 'command')):
            sdc_planning, sdc_planning_mask, command = (
                traj_api.get_sdc_planning_label(info['token']))
            info['sdc_planning'] = sdc_planning
            info['sdc_planning_mask'] = sdc_planning_mask
            info['command'] = command

        # *==========================================================================#
        # 写入候选轨迹 sample_traj，替代训练阶段在线读取 CAN bus pose/steering 再采样。
        # 注意 sample_traj 不是 nuScenes 中真实发生的未来轨迹；
        # nuScenes 的真实未来自车轨迹已经写在上面的 sdc_planning 中。
        # sample_traj 是根据“当前速度 v0 + 当前方向盘转角 steering/曲率 kappa”
        # 通过 trajectory_sampler.sample() 采样出来的一组可能 ego motion proposals。
        # 之所以叫“候选轨迹”，是因为 planner 可以在这些不同候选动作/轨迹上评估代价，
        # 或将其作为规划/动作条件相关分支的输入。
        if overwrite or 'sample_traj' not in info:
            info['sample_traj'] = get_trajectory_sampling(
                nusc_can=nusc_can,
                scene_token_to_name=scene_token_to_name,
                scene_name_to_location=scene_name_to_location,
                rec=info,
                future_length=planning_steps)

        new_infos.append(info)

    out_data = dict(data)
    out_data['infos'] = new_infos
    mmcv.mkdir_or_exist(os.path.dirname(out_file))
    mmcv.dump(out_data, out_file)
    print(f'Saved augmented PKL to: {out_file}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute Drive-OccWorld planning/action fields into PKL.')
    # 输入是当前 config 正在使用的 train/val pkl。
    parser.add_argument('--train-ann-file', default=DEFAULT_TRAIN_ANN_FILE)
    parser.add_argument('--val-ann-file', default=DEFAULT_VAL_ANN_FILE)
    # 输出建议另存为 v2，避免覆盖原始 pkl。
    parser.add_argument('--out-train-ann-file', default=DEFAULT_NEW_TRAIN_ANN_FILE)
    parser.add_argument('--out-val-ann-file', default=DEFAULT_NEW_VAL_ANN_FILE)
    # nusc-root 是 v1.0-trainval 目录；can-bus-root 是 NuScenes 根目录。
    parser.add_argument('--nusc-root', default=DEFAULT_NUSC_ROOT)
    parser.add_argument('--can-bus-root', default=DEFAULT_CAN_BUS_ROOT)
    parser.add_argument('--version', default='v1.0-trainval')
    parser.add_argument(
        '--planning-steps',
        type=int,
        default=5,
        help='future_length + 1 used by NuScenesTraj/sample_traj. '
             'For current configs future_length=4, so default is 5.')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        # 默认不覆盖已有字段，方便重复运行；加 --overwrite 会强制重新计算。
        help='Recompute fields even if they already exist in the input PKL.')
    return parser.parse_args()


def main():
    args = parse_args()

    # 只在脚本启动时初始化一次 SDK，避免 Dataset 每次构造都做这些在线查询。
    # SDK = Software Development Kit，即 nuScenes 官方提供的 Python 工具包/查询接口；
    # 它负责帮我们查原始 json 表、CAN bus 消息和 token 之间的关联关系。
    # 本脚本的目标就是把这些 SDK 查询结果提前固化进 PKL，减少训练阶段在线查询。
    #
    # nusc: nuScenes 主数据库接口，负责提供「静态/标注/时序 meta」：
    #   - scene -> log -> location，用来判断地图位置，例如 singapore 场景；
    #   - sample/sample_data/ego_pose/calibrated_sensor，用来沿着当前帧 token 查询未来帧；
    #   - sample_annotation 等标注信息，可扩展用于未来框、实例时序等预处理。
    #   这里既会查当前帧信息，也会沿 sample['next'] 查未来帧信息。
    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=True)
    #
    # nusc_can: nuScenes CAN bus 接口，负责提供「与当前 sample 时间戳对齐的自车状态」。
    #   CAN bus = Controller Area Network bus，即车辆内部控制器局域网总线。
    #   在自动驾驶数据集中，它记录的是自车底盘/定位相关信号，而不是相机图像或目标标注；
    #   例如 pose、速度、加速度、角速度、方向盘转角等车辆自身运动状态。
    #   这里“自车状态”不是笼统地全部写入 PKL，而是用于 sample_traj 采样的两个量：
    #   1) 当前纵向速度 v0；
    #   2) 当前方向盘转角 steering。
    #   CAN bus pose 消息本身还包含位置/姿态/速度等信息，但本脚本当前只取 pose_data['vel'][0]；
    #   自车未来位置/偏航监督则由下面的 traj_api 从 nuScenes sample/ego_pose 链生成。
    #   - pose 消息：从 CAN bus 高频 pose 序列中，取与当前帧 timestamp 对齐/最近的一条，
    #     读取其中的当前纵向速度 vel，作为 sample_traj 的初始速度；
    #   - steeranglefeedback 消息：同样按当前帧 timestamp 对齐/最近，读取当前方向盘转角；
    #   - 这些值会被当作“当前帧”的初始运动状态，用于离线生成 sample_traj 候选轨迹。
    #   注意：这里不是历史序列，也不是未来序列；只是把 CAN bus 高频消息对齐到当前 sample。
    nusc_can = NuScenesCanBus(dataroot=args.can_bus_root)
    #
    # traj_api: 基于 nusc 的未来 sample 链生成「未来 SDC 监督标签」。
    #   SDC = Self-Driving Car，即自车/ego vehicle。
    #   - sdc_planning: 未来自车在当前参考帧 LiDAR 坐标系下的 x/y/yaw；
    #   - sdc_planning_mask: 哪些未来 step 有效；
    #   - command: 根据未来自车位移粗略离散成 left/right/forward。
    #   未来时刻数量由 args.planning_steps 控制，并传给 NuScenesTraj(planning_steps=...)。
    #   当前配置 future_length=4，而 Dataset 原来使用 future_length+1。
    #   原因是 Dataset/union2one 中会把“当前帧位置”拼到 sdc_planning 前面，
    #   再做相邻时刻差分，得到每个 0.5s step 的位移/action：
    #       [current, t+1, t+2, ...] -> diff -> [current->t+1, t+1->t+2, ...]
    #   因此要得到 future_length 个相邻位移，需要 future_length+1 个未来/参考位置点。
    #   所以脚本默认 planning_steps=5，表示最多生成 5 个未来关键帧的 SDC 标签。
    #   因此它提供的是未来 GT planning/action label，而不是模型预测出来的计划。
    traj_api = NuScenesTraj(
        nusc=nusc,
        CLASSES=CLASS_NAMES,
        box_mode_3d=Box3DMode.LIDAR,
        planning_steps=args.planning_steps)

    augment_infos(
        ann_file=args.train_ann_file,
        out_file=args.out_train_ann_file,
        nusc=nusc,
        nusc_can=nusc_can,
        traj_api=traj_api,
        planning_steps=args.planning_steps,
        overwrite=args.overwrite)
    augment_infos(
        ann_file=args.val_ann_file,
        out_file=args.out_val_ann_file,
        nusc=nusc,
        nusc_can=nusc_can,
        traj_api=traj_api,
        planning_steps=args.planning_steps,
        overwrite=args.overwrite)


if __name__ == '__main__':
    main()
