import torch
from torch import nn

class MS_CAM_3D_ONE_DIM(nn.Module):
    "From https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py"

    def __init__(
        self,
        input_channel=64,
        output_channel=64,
        r=4,
        global_attn_dim="wz",
        norm_type="batch",
        num_groups=16,
    ):
        super(MS_CAM_3D_ONE_DIM, self).__init__()
        inter_channels = int(input_channel // r)

        self.global_attn_dim = global_attn_dim

        def make_norm(channels):
            if norm_type == "batch":
                return nn.BatchNorm3d(channels)
            elif norm_type == "group":
                return nn.GroupNorm(num_groups, channels)
            else:
                raise ValueError(f"Unsupported normalization type: {norm_type}")

        self.local_att = nn.Sequential(
            nn.Conv3d(
                input_channel, inter_channels, kernel_size=1, stride=1, padding=0
            ),
            make_norm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                inter_channels, output_channel, kernel_size=1, stride=1, padding=0
            ),
            make_norm(output_channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv3d(
                input_channel, inter_channels, kernel_size=1, stride=1, padding=0
            ),
            make_norm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                inter_channels, output_channel, kernel_size=1, stride=1, padding=0
            ),
            make_norm(output_channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #* ================== 2.6.1 门控权重：局部分支 + 平面池化分支 ==================
        # x 是 coarse/refined 沿通道拼接的特征，默认 [B,256,X,Y,Z]；本模块输出 128 通道的融合权重。
        # local_att 的 1×1×1 卷积在各体素位置混合通道，不改变空间尺寸。
        xl = self.local_att(x)  # [B,128,X,Y,Z]，尚未 sigmoid 的门控分数。

        xg = 0
        #* global_att 将最后两个空间轴平均池化为 1，再经通道变换；通过 permute 选择池化平面。
        # 原命名 h/w/z 在这里对应实际张量的 X/Y/Z；并非将三维特征永久压成二维输出。
        if self.global_attn_dim == "wz":
            xg = self.global_att(x)  # 对 Y/Z 平面池化，保留 X：输出 [B,128,X,1,1]。
        elif self.global_attn_dim == "hw":
            xg = self.global_att(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)  # 对 X/Y 平面池化，保留 Z：[B,128,1,1,Z]。
        elif self.global_attn_dim == "hz":
            xg = self.global_att(x.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)  # 对 X/Z 平面池化，保留 Y：[B,128,1,Y,1]。
        else:
            raise ValueError(
                f"Unsupported global attention dimension: {self.global_attn_dim}"
            )
        xlg = xl + xg  # 广播平面统计到完整网格，与局部门控分数相加：[B,128,X,Y,Z]。

        #* 每个通道、每个体素得到一个 [0,1] 融合系数；不是类别概率，也没有在三个分支之间做 softmax。
        return self.sigmoid(xlg)

class DualFeatFusion_Tri(nn.Module):
    def __init__(self, input_channel, output_channel, norm_type="batch", num_groups=16):
        super(DualFeatFusion_Tri, self).__init__()
        self.ca_wz = MS_CAM_3D_ONE_DIM(
            input_channel * 2,
            output_channel,
            global_attn_dim="wz",
            norm_type=norm_type,
            num_groups=num_groups,
        )
        self.ca_hw = MS_CAM_3D_ONE_DIM(
            input_channel * 2,
            output_channel,
            global_attn_dim="hw",
            norm_type=norm_type,
            num_groups=num_groups,
        )
        self.ca_hz = MS_CAM_3D_ONE_DIM(
            input_channel * 2,
            output_channel,
            global_attn_dim="hz",
            norm_type=norm_type,
            num_groups=num_groups,
        )

        #* 原 forward 直接相加，未使用 self.fuse；不保留无梯度参数。

    def forward(self, x1, x2):
        #* ================== 2.6.2 根据两路特征预测三组门控权重 ==================
        # x1=coarse（LSS 粗特征），x2=refined（注意力细化特征）；均为 [B,C,X,Y,Z]。
        # 默认 C=128、X/Y/Z=128/128/16；拼接后 [B,256,128,128,16]，三组权重各为 [B,128,128,128,16]。
        # 三个独立 MS_CAM 分支分别利用 YZ/XY/XZ 平面池化统计，并结合局部分支生成权重。
        channel_factor_wz = self.ca_wz(torch.cat((x1, x2), 1))
        channel_factor_hw = self.ca_hw(torch.cat((x1, x2), 1))
        channel_factor_hz = self.ca_hz(torch.cat((x1, x2), 1))

        #* ================== 2.6.3 各分支分别融合 coarse 与 refined ==================
        # 每处融合为 α*coarse+(1-α)*refined：α 越大越偏向 coarse，越小越偏向 refined。
        # 逐元素加权，不拼接输出通道；每个分支仍为 [B,C,X,Y,Z]，两路输入及门控网络均可接收梯度。
        out_wz = channel_factor_wz * x1 + (1 - channel_factor_wz) * x2
        out_hw = channel_factor_hw * x1 + (1 - channel_factor_hw) * x2
        out_hz = channel_factor_hz * x1 + (1 - channel_factor_hz) * x2

        #* ================== 2.6.4 三分支求和并返回完整体素特征 ==================
        #* 沿用原实现直接相加，不除以 3，也不是再拼接；输出尚不是 occupancy 类别 logits。
        out = out_wz + out_hw + out_hz

        return out  # [B,C,X,Y,Z]，默认 [B,128,128,128,16]，继续送入后续 3D 编码器/占据头。
