"""位姿补偿与残余采样流解耦的体素状态更新；尚未接入训练配置。"""
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models.builder import NECKS


@NECKS.register_module()
class DecoupledVoxelDynamics(nn.Module):
    """输入上一时刻 state[B,C,X,Y,Z]，输出目标时刻同形状特征。

    借鉴动静态解耦思路，不是 DFIT-OccWorld 的完整复现：当前仅用单帧
    连续特征，预测目标网格上的反向采样残余流，不含历史编码或图像渲染损失。
    """

    def __init__(self, channels, pc_range, hidden_channels=64, refinement_layers=2):
        super().__init__()
        bounds = torch.tensor(pc_range, dtype=torch.float32)
        if bounds.shape != (6,) or not torch.isfinite(bounds).all() or not (bounds[3:] > bounds[:3]).all():
            raise ValueError('pc_range 必须为有限且递增的 (xmin,ymin,zmin,xmax,ymax,zmax)')
        if channels < 1 or hidden_channels < 1 or refinement_layers < 1:
            raise ValueError('通道数和 refinement_layers 必须为正整数')
        self.channels = channels
        self.register_buffer('bounds', bounds, persistent=False)

        #! 空间邻域编码：从位姿补偿后的特征预测流和门控，而非逐体素独立线性预测。
        self.motion_encoder = nn.Sequential(
            nn.Conv3d(channels, hidden_channels, 1), nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1), nn.GELU())
        self.flow_head = nn.Conv3d(hidden_channels, 3, 1)
        self.gate_head = nn.Conv3d(hidden_channels, 1, 1)
        layers = [nn.Conv3d(channels, hidden_channels, 1), nn.GELU()]
        for _ in range(refinement_layers):
            layers.extend([nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1), nn.GELU()])
        layers.append(nn.Conv3d(hidden_channels, channels, 1))
        self.refinement = nn.Sequential(*layers)

        #! 初始 flow=0、细化残差=0，输出等价于静态位姿重采样；gate=0.5 不阻断流梯度。
        for layer in (self.flow_head, self.gate_head, self.refinement[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)


    def reference_grid(self, state, target_to_memory):
        #* 目标网格中心 → 物理坐标 → 上一时刻 memory 坐标 → [0,1] 归一化坐标。
        shape = state.shape[2:]
        axes = [(torch.arange(n, device=state.device, dtype=state.dtype) + .5) / n for n in shape]
        centers = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)  # [X,Y,Z,3]。
        bounds = self.bounds.to(state)
        points = (centers * (bounds[3:] - bounds[:3]) + bounds[:3]).reshape(1, -1, 3)
        # target_to_memory=[B,4,4] 为列向量格式，平移在最后一列。
        pose = target_to_memory.to(state)
        points = points @ pose[:, :3, :3].transpose(-1, -2) + pose[:, None, :3, 3]
        return ((points - bounds[:3]) / (bounds[3:] - bounds[:3])).reshape(state.shape[0], *shape, 3)

    @staticmethod
    def sample(state, reference):
        #* state=[B,C,X,Y,Z] 对应 grid_sample 的 D/H/W；采样坐标必须换为 (Z,Y,X)。
        grid = reference[..., [2, 1, 0]] * 2 - 1  # [B,X,Y,Z,3]，越界部分零填充。
        return F.grid_sample(state, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    def forward(self, state, target_to_memory):
        #! ================== 1. 静态分支：仅补偿自车运动 ==================
        reference = self.reference_grid(state, target_to_memory)  # [B,X,Y,Z,3]。
        base = self.sample(state, reference)  # [B,C,X,Y,Z]，目标帧网格上的静态候选。

        #! ================== 2. 动态分支：学习额外的反向采样流 ==================
        hidden = self.motion_encoder(base)
        flow = self.flow_head(hidden)  # [B,3,X,Y,Z]，memory 的 (X,Y,Z) 体素格数，不是米。
        gate = self.gate_head(hidden).sigmoid()  # [B,1,X,Y,Z]，目标网格上的软门控。
        #* flow 表示目标位置应去 memory 哪里取值，不是物体正向运动速度。
        # 同坐标系物体向 +X 移动一格，目标位置需沿 -X 找来源，即 flow_x=-1。
        # gate 无运动真值监督时仅是分支混合权重，不等同于真实动态概率。
        scale = state.new_tensor(state.shape[2:])
        dynamic_reference = reference + flow.permute(0, 2, 3, 4, 1) / scale
        #* 直接采样原 state，避免对已插值的 base 再 warp，造成二次插值平滑。
        dynamic = self.sample(state, dynamic_reference)  # [B,C,X,Y,Z]。

        #! ================== 3. 软融合与局部残差细化 ==================
        coarse = (1 - gate) * base + gate * dynamic
        # 细化用于学习修正拖影/缺失，但不能保证恢复新显露内容或推断真实物体速度。
        # 不 detach；返回完整状态，外层不应再次加 state，可直接接共享占据解码器。
        return coarse + self.refinement(coarse)
