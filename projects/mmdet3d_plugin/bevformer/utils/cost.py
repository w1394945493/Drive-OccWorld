import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.draw import polygon

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx

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


class Cost_Function(nn.Module):
    """候选自车轨迹总代价函数。

    这个模块用于 PlanHead_v1 中的 cost-based planning：
    - 输入一批候选轨迹 trajs；
    - 根据预测/GT occupancy、可行驶区域和 BEV cost volume 计算每条轨迹的代价；
    - 输出每条候选轨迹的总代价，后续 select() 会选择代价最小的轨迹。

    当前实际启用的代价包括：
    1. SafetyCost：候选轨迹是否和动态障碍物占用区域重叠；
    2. HeadwayCost：候选轨迹前方一定距离内是否存在障碍物；
    3. Rule：候选轨迹是否离开可行驶区域；
    4. Cost_Volume：从网络预测的 cost_volume 上采样得到的可学习代价。

    注释掉的 LR_divider / Comfort / Progress 是预留项，当前不会参与最终 cost。
    """

    def __init__(self, cfg):
        super(Cost_Function, self).__init__()

        #* ================== 1. 基于 occupancy / map rule 的手工代价 ==================
        # SafetyCost: 计算自车 footprint 沿候选轨迹是否压到 instance_occupancy。
        # instance_occupancy 来自语义 occupancy 中的动态目标类别。
        self.safetycost = SafetyCost(cfg)

        # HeadwayCost: 计算自车前方安全距离区域是否存在障碍物。
        # 它比 SafetyCost 更关注“前方一段距离”的潜在碰撞风险。
        self.headwaycost = HeadwayCost(cfg)

        # 下面几个 cost 是预留/未启用项：
        # - LR_divider: 可用于惩罚靠近/跨越车道线；
        # - Comfort: 可用于惩罚过大加速度、横向加速度、jerk；
        # - Progress: 可用于鼓励向前推进或靠近目标。
        # self.lrdividercost = LR_divider(cfg)
        # self.comfortcost = Comfort(cfg)
        # self.progresscost = Progress(cfg)

        # Rule: 计算候选轨迹是否驶出 drivable_area。
        # drivable_area 来自语义 occupancy 中的 driveable_surface 类别。
        self.rulecost = Rule(cfg)

        #* ================== 2. 网络预测的可学习代价 ==================
        # Cost_Volume: 在 PlanHead_v1 中由 costvolume_head(bev_feats) 预测得到。
        # 这里沿候选轨迹在 cost_volume 上采样，把 BEV 特征学习到的风险/偏好转成轨迹代价。
        self.costvolume = Cost_Volume(cfg)

    def forward(self, cost_volume, trajs, instance_occupancy, drivable_area):
        """计算每条候选轨迹的总代价。

        Args:
            cost_volume: torch.Tensor, shape (B, H, W)。
                PlanHead_v1 从 BEV 特征预测出的单通道代价图。
            trajs: torch.Tensor, shape (B, N, 2)。
                N 条候选轨迹在 BEV/自车坐标系下的位置点，这里只使用 x/y。
                在 PlanHead_v1 中传入的是 trajs[:, :, :2]。
            instance_occupancy: torch.Tensor, shape (B, H, W)。
                动态障碍物占用图；有动态目标的位置为 1。
            drivable_area: torch.Tensor, shape (B, H, W)。
                可行驶区域图；可行驶区域为 1。

        Returns:
            cost_fo: torch.Tensor, shape 通常为 (B, N)。
                每条候选轨迹的总代价。值越小表示轨迹越优。
        """
        # * 总体代价 = safe cost + headway cost + costvolume + rule cost
        # * 1. safety cost: 是否压倒障碍物，把自车矩形 footprint 放到候选轨迹位置上，然后在 instance_occupancy 上采样。如果候选轨迹对应的自车车身区域覆盖到了动态障碍物，占用越多，代价越高。
        #* Safety cost：惩罚候选轨迹与动态障碍物 occupancy 重叠。
        # clamp 到 [0, 100] 是为了避免某一项代价数值过大，压制其他代价项。
        safetycost = torch.clamp(self.safetycost(trajs, instance_occupancy), 0, 100)                 # penalize overlap with instance_occupancy

        #* Headway cost：惩罚候选轨迹前方安全距离内存在障碍物。
        # 和 safety cost 相比，它更强调前方行驶空间是否安全。
        headwaycost = torch.clamp(self.headwaycost(trajs, instance_occupancy, drivable_area), 0, 100)# penalize overlap with front instance (10m)

        # 未启用的代价项，保留作后续扩展。
        # lrdividercost = torch.clamp(self.lrdividercost(trajs, lane_divider), 0, 100)               # penalize distance with lane
        # comfortcost = torch.clamp(self.comfortcost(trajs), 0, 100)                                   # penalize high accelerations (lateral, longitudinal, jerk)
        # progresscost = torch.clamp(self.progresscost(trajs), -100, 100)                              # L2 loss

        #* Rule cost：惩罚驶出可行驶区域。
        # 如果候选轨迹 footprint 落在 drivable_area=0 的区域，代价会增大。
        rulecost = torch.clamp(self.rulecost(trajs, drivable_area), 0, 100)                          # penalize overlap with out of drivable_area

        #* Learned cost volume：从网络预测的 cost_volume 上按轨迹位置采样。
        # 这一项让 planner 不只依赖手工规则，也能利用 BEV 特征学习到的数据驱动代价。
        costvolume = torch.clamp(self.costvolume(trajs, cost_volume), 0, 100)                        # sample on costvolume

        #* 最终总代价：当前实现直接等权相加。
        # 后续 PlanHead_v1.select() 会对 cost_fo 做 topk(largest=False)，选择代价最小的候选轨迹。
        cost_fo = safetycost + headwaycost + costvolume + rulecost
        # cost_fc = progresscost

        return cost_fo



class BaseCost(nn.Module):
    """所有 cost 子项共享的 BEV 几何工具类。

    这里主要做两件事：
    1. 根据 plan_grid_conf 建立真实坐标 <-> BEV 栅格坐标的映射；
    2. 根据自车尺寸构造 ego footprint，并把它平移到候选轨迹位置上，
       之后在 occupancy / drivable_area / cost_volume 上采样得到代价。
    """

    def __init__(self, grid_conf):
        super(BaseCost, self).__init__()
        self.grid_conf = grid_conf

        dx, bx, _ = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        dx, bx = dx[:2], bx[:2]
        # dx: BEV 栅格分辨率，例如 [0.5, 0.5] m/grid。
        # bx: BEV 第一个栅格中心对应的真实坐标。
        # 注册为不可学习 Parameter 是原实现写法，本质上是几何常量。
        self.dx = nn.Parameter(dx,requires_grad=False)
        self.bx = nn.Parameter(bx,requires_grad=False)

        # bev_dimension: BEV 网格尺寸，例如 [200, 200, 1]。
        _,_, self.bev_dimension = calculate_birds_eye_view_parameters(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
        )

        # 自车矩形 footprint 尺寸，单位 m。
        # W: 自车宽度；H: 自车长度。
        # 后续 get_origin_points() 会用它构造自车在 BEV 上占据的矩形区域。
        self.W = 1.85
        self.H = 4.084

    def get_origin_points(self, lambda_=0):
        """构造位于原点附近的自车矩形 footprint，并转成 BEV 像素集合。

        Args:
            lambda_: 额外膨胀量。lambda_ 越大，自车 footprint 越保守，
                会覆盖更大的 BEV 区域。

        Returns:
            rc: Tensor, shape [M, 2]。
                自车 footprint 内部所有 BEV 像素坐标，格式近似为 [row, col]。
        """
        W = self.W
        H = self.H
        # 四个角点定义的是自车矩形轮廓。
        # 这里的 0.5 是原 ST-P3/UniAD 风格实现中的纵向偏置，
        # 可理解为让矩形 footprint 与自车参考点/车体中心对齐。
        pts = np.array([
            [-H / 2. + 0.5 - lambda_, W / 2. + lambda_],
            [H / 2. + 0.5 + lambda_, W / 2. + lambda_],
            [H / 2. + 0.5 + lambda_, -W / 2. - lambda_],
            [-H / 2. + 0.5 - lambda_, -W / 2. - lambda_],
        ])  # [lidar_y, lidar_x]
        # 真实坐标 -> BEV 连续坐标。
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())   # [bev_w, bev_h]
        # pts[:, [0, 1]] = pts[:, [1, 0]] # [bev_h, bev_w]
        # polygon 将矩形四边形内部填充成 BEV 像素点集合。
        rr , cc = polygon(pts[:,1], pts[:,0])   # [bev_h, bev_w]
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)  # [bev_h, bev_w]
        return torch.from_numpy(rc).to(device=self.bx.device) # (27,2)

    def get_points(self, trajs, lambda_=0):
        '''
        trajs: torch.Tensor<float> (B, N, 2)
        return:
        List[ torch.Tensor<int> (B, N), torch.Tensor<int> (B, N)]
        '''
        # rc 是自车矩形 footprint 在原点处覆盖的 BEV 像素集合，shape [M, 2]。
        rc = self.get_origin_points(lambda_)    # [bev_h, bev_w]
        B, N, _ = trajs.shape         # delta_[lidar_x, lidar_y]

        # 将候选轨迹真实位移 [m] 转为 BEV 栅格位移 [grid]，
        # 再把原点处自车 footprint 平移到每条候选轨迹的位置。
        # 最终 trajs shape 约为 [B, N, M, 2]，
        # 表示每条候选轨迹对应的自车矩形 footprint 覆盖哪些 BEV 像素。
        trajs = trajs.view(B, N, 1, 2) / self.dx  # delta_[bev_h, bev_w]
        # trajs[:,:,:,:,[0,1]] = trajs[:,:,:,:,[1,0]]
        trajs = trajs + rc  # [bev_h, bev_w]

        # 将连续坐标转为整数 BEV index，并 clamp 到有效边界内。
        rr = trajs[:,:,:,0].long()
        rr = torch.clamp(rr, 0, self.bev_dimension[0] - 1)

        cc = trajs[:,:,:,1].long()
        cc = torch.clamp(cc, 0, self.bev_dimension[1] - 1)

        return rr, cc

    def compute_area(self, instance_occupancy, trajs, ego_velocity=None, _lambda=0):
        '''
        instance_occupancy: torch.Tensor<float> (B, 200, 200)
        trajs: torch.Tensor<float> (B, N, 2)
        ego_velocity: torch.Tensor<float> (B, N)
        '''
        # _lambda 以米为单位传入，这里换算成 BEV grid 数。
        # 例如 _lambda=1m, dx=0.5m/grid，则 footprint 向外膨胀约 2 个 grid。
        _lambda = int(_lambda / self.dx[0])

        # rr/cc: 每条候选轨迹处，自车 footprint 覆盖的 BEV 像素集合。
        # shape 约为 [B, N, M]。
        rr, cc = self.get_points(trajs, _lambda)    # [bev_h, bev_w]
        B, N, _ = trajs.shape

        if ego_velocity is None:
            ego_velocity = torch.ones((B,N), device=trajs.device)

        # ii 用于 batch 维索引；必须和 trajs/occupancy 同设备。
        ii = torch.arange(B, device=trajs.device)

        # 在 occupancy / dangerous_area 上采样自车 footprint 覆盖的所有像素，
        # 并对 footprint 内像素求和：
        # - 若输入是 instance_occupancy，则表示车身压到多少动态障碍物像素；
        # - 若输入是 dangerous_area，则表示车身压到多少不可行驶区域像素。
        subcost = instance_occupancy[ii[:, None, None], rr, cc].sum(dim=-1)

        # 可选速度权重：速度越大，在相同碰撞/占用面积下代价越高。
        subcost = subcost * ego_velocity

        return subcost

    def discretize(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        '''
        B, N,  _ = trajs.shape # delta_[lidar_x, lidar_y]

        xx, yy = trajs[:,:,0], trajs[:,:,1] # delta_[lidar_x, lidar_y]

        # 将候选轨迹点从真实坐标离散为 BEV index。
        # 注意：这只取候选轨迹中心点，不考虑自车矩形 footprint。
        # Cost_Volume 使用这个函数在 learned cost map 上采样中心点代价。
        xi = ((xx - self.bx[0]) / self.dx[0]).long()
        xi = torch.clamp(xi, 0, self.bev_dimension[0]-1)    # bev_h

        yi = ((yy - self.bx[1]) / self.dx[1]).long()
        yi = torch.clamp(yi,0, self.bev_dimension[1]-1)     # bev_w

        return xi, yi

    def evaluate(self, trajs, C):
        '''
            trajs: torch.Tensor<float> (B, N, 2)   N: sample number
            C: torch.Tensor<float> (B, 200, 200)
        '''
        B, N, _ = trajs.shape

        # ii 用于 batch 维索引；必须和 trajs/C 同设备。
        ii = torch.arange(B, device=trajs.device)

        # Syi/Sxi 是每条候选轨迹中心点对应的 BEV index。
        Syi, Sxi = self.discretize(trajs)

        # 在代价图 C 上采样候选轨迹中心点处的代价。
        CS = C[ii, Syi, Sxi]
        return CS

class Cost_Volume(BaseCost):
    def __init__(self, cfg):
        super(Cost_Volume, self).__init__(cfg)

        self.factor = 100.

    def forward(self, trajs, cost_volume):
        '''
        cost_volume: torch.Tensor<float> (B, 200, 200)
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        '''

        # cost_volume 是网络从 BEV feature 预测出的可学习代价图。
        # 先限制数值范围，再沿候选轨迹中心点采样。
        cost_volume = torch.clamp(cost_volume, 0, 1000)

        # factor=100 放大 learned cost 的量级，使其能和 rule/safety/headway cost 相加。
        return self.evaluate(trajs, cost_volume) * self.factor

class Rule(BaseCost):
    def __init__(self, cfg):
        super(Rule, self).__init__(cfg)

        self.factor = 5

    def forward(self, trajs, drivable_area):
        '''
            trajs: torch.Tensor<float> (B, N, 2)   N: sample number
            drivable_area: torch.Tensor<float> (B, 200, 200)
        '''
        B, _,  _ = trajs.shape

        # drivable_area=1 表示可行驶。
        # dangerous_area=1 表示不可行驶区域。
        dangerous_area = torch.logical_not(drivable_area).float()
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(dangerous_area[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        # 计算自车 footprint 压到不可行驶区域的面积，面积越大 rule cost 越高。
        subcost = self.compute_area(dangerous_area, trajs)

        return subcost * self.factor


class SafetyCost(BaseCost):
    def __init__(self, cfg):
        super(SafetyCost, self).__init__(cfg)
        self.w = nn.Parameter(torch.tensor([1.,1.]),requires_grad=False)

        self._lambda = 1.
        self.factor = 0.1

    def forward(self, trajs, instance_occupancy):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        instance_occupancy: torch.Tensor<float> (B, 200, 200)
        '''
        # * SafetyCost：是否压到动态障碍物：它会把自车矩形 footprint 放到候选轨迹位置上，然后在 instance_occupancy 上采样。
        # * 如果候选轨迹对应的自车车身区域覆盖到了动态障碍物，占用越多，代价越高。
        B, N, _ = trajs.shape
        # 根据候选位移近似速度，nuScenes keyframe 间隔为 0.5s。
        # 速度用于让高速状态下的潜在碰撞更重。
        ego_velocity = torch.sqrt((trajs ** 2).sum(axis=-1)) / 0.5  # B,N

        # subcost1: 原始自车 footprint 与动态障碍物 occupancy 的重叠面积。
        # 对应 o_c(tau, t, 0)。
        subcost1 = self.compute_area(instance_occupancy, trajs)

        # subcost2: 膨胀后的自车 footprint 与动态障碍物 occupancy 的重叠面积，
        # 并乘以 ego_velocity。_lambda=1m 会扩大安全边界，更保守。
        # 对应 o_c(tau, t, lambda) x v(tau, t)。
        subcost2 = self.compute_area(instance_occupancy, trajs, ego_velocity, self._lambda)

        # 两个安全项等权相加，再乘 factor 缩放量级。
        subcost = subcost1 * self.w[0] + subcost2 * self.w[1]

        return subcost * self.factor


class HeadwayCost(BaseCost):
    def __init__(self, cfg):
        super(HeadwayCost, self).__init__(cfg)
        self.L = 10  # Longitudinal distance keep 10m
        self.factor = 1.

    def forward(self, trajs, instance_occupancy, drivable_area):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        instance_occupancy: torch.Tensor<float> (B, 200, 200)
        drivable_area: torch.Tensor<float> (B, 200, 200)
        '''
        B, N, _ = trajs.shape
        # 只关注可行驶区域上的动态障碍物。
        # 例如道路上的车辆会保留，非可行驶区域上的物体对 headway 影响较小。
        instance_occupancy_ = instance_occupancy * drivable_area  # B,H,W
        # breakpoint()
        # import matplotlib.pyplot as plt
        # plt.imshow(instance_occupancy_[0].detach().cpu().numpy())
        # plt.show()
        # breakpoint()
        tmp_trajs = trajs.clone()
        # 将候选轨迹点沿前向 y 方向平移 10m，
        # 用于检查“候选位置前方 10m”是否有动态障碍物。
        tmp_trajs[:,:,1] = tmp_trajs[:,:,1]+self.L

        # 如果前方 10m 的自车 footprint 覆盖到障碍物，则 headway cost 增大。
        subcost = self.compute_area(instance_occupancy_, tmp_trajs)

        return subcost * self.factor

class LR_divider(BaseCost):
    def __init__(self, cfg):
        super(LR_divider, self).__init__(cfg)
        self.L = 1 # Keep a distance of 2m from the lane line
        self.factor = 10.

    def forward(self, trajs, lane_divider):
        '''
        trajs: torch.Tensor<float> (B, N, 2)   N: sample number
        lane_divider: torch.Tensor<float> (B, 200, 200)
        '''
        B, N, _ = trajs.shape

        xx, yy = self.discretize(trajs) # [bev_h, bev_w]
        xy = torch.stack([xx,yy],dim=-1) # (B, N, 2)  [bev_h, bev_w]

        # lane divider
        res1 = []
        for i in range(B):
            index = torch.nonzero(lane_divider[i]) # (n, 2)
            if len(index) != 0:
                xy_batch = xy[i].view(N, 1, 2)
                distance = torch.sqrt((((xy_batch - index) * reversed(self.dx))**2).sum(dim=-1)) # (N, n)
                distance,_ = distance.min(dim=-1) # (N)
                index = distance > self.L
                distance = (self.L - distance) ** 2
                distance[index] = 0
            else:
                distance = torch.zeros((N),device=trajs.device)
            res1.append(distance)
        res1 = torch.stack(res1, dim=0)

        return res1 * self.factor


class Comfort(BaseCost):
    def __init__(self, cfg):
        super(Comfort, self).__init__(cfg)

        self.c_lat_acc = 3 # m/s2
        self.c_lon_acc = 3 # m/s2
        self.c_jerk = 1 # m/s3

        self.factor = 0.1

    def forward(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, 2)
        '''
        B, N, _ = trajs.shape
        lateral_velocity = trajs[:,:,0] / 0.5
        longitudinal_velocity = trajs[:,:,1] / 0.5
        lateral_acc = lateral_velocity / 0.5    # B,N
        longitudinal_acc = longitudinal_velocity / 0.5  # B,N

        # jerk
        ego_velocity = torch.sqrt((trajs ** 2).sum(dim=-1)) / 0.5
        ego_acc = ego_velocity / 0.5
        ego_jerk = ego_acc / 0.5    # B,N

        subcost = torch.zeros((B, N), device=trajs.device)

        lateral_acc = torch.clamp(torch.abs(lateral_acc) - self.c_lat_acc, 0,30)
        subcost += lateral_acc ** 2
        longitudinal_acc = torch.clamp(torch.abs(longitudinal_acc) - self.c_lon_acc, 0, 30)
        subcost += longitudinal_acc ** 2
        ego_jerk = torch.clamp(torch.abs(ego_jerk) - self.c_jerk, 0, 20)
        subcost += ego_jerk ** 2

        return subcost * self.factor

class Progress(BaseCost):
    def __init__(self, cfg):
        super(Progress, self).__init__(cfg)
        self.factor = 0.5

    def forward(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, 2)
        target_points: torch.Tensor<float> (B, 2)
        '''
        target_points = torch.zeros_like(trajs[:, 0, :])    # B,2
        B, N,  _ = trajs.shape
        subcost1 = trajs[:,:,1]

        if target_points.sum() < 0.5:
            subcost2 = 0
        else:
            target_points = target_points.unsqueeze(1)
            subcost2 = ((trajs - target_points) ** 2).sum(dim=-1)

        return (subcost2 - subcost1) * self.factor
