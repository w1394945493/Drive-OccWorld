"""FoundationSSC DSGP：标定条件 context + 深度引导 DFormer + 视差通道混合。"""
import torch
from torch import nn
from torch.nn import functional as F
from .camera_condition import get_mlp_input
from .geometry import disparity_to_depth
from .modules.Context_Net_modules import ContextNet
from .modules.Depth_Net_modules import Disp2DepthChannelMixer, StereoVolumeEncoder
from .modules.DFormerv2 import dformerv2
from .modules.depth_utils import normalize_depth_to_255, normalize


class GeometryDepthNet(nn.Module):
    def __init__(self, input_channels=640, channels=128, disparity_channels=104,
                 depth_bound=(2., 58., .5), dformer_layers=4, mixer_layers=8):
        super().__init__()
        self.depth_bound = depth_bound
        self.disparity_channels = disparity_channels
        self.depth_channels = len(torch.arange(*depth_bound))
        if channels % 8 or self.depth_channels % 2:
            raise ValueError('context 通道需整除 8，深度通道需整除 2')
        self.context_net = ContextNet(input_channels, input_channels, channels, cam_channels=33)
        self.dformerv2 = dformerv2(embed_dims=[channels], depths=[dformer_layers], num_heads=[8], heads_ranges=[4])
        self.disp2depth_mlp = Disp2DepthChannelMixer(disparity_channels, 256, 512, self.depth_channels, mixer_layers)
        self.stereo_volume_encoder = StereoVolumeEncoder(self.depth_channels, self.depth_channels)

    def forward(self, features, disparity, cam_params, baseline):
        #* ================== 2.2.1 输入准备 ==================
        # features：左图融合特征，默认 [B,1,640,48,160]；下面的尺寸均以当前配置为例。
        # cam_params：左相机内外参、图像变换和公共 BDA；baseline：双目基线，单位米。
        #* disparity 来自冻结的 FoundationStereo，不是 GT；本模块内部网络仍可训练。
        b, n, c, h, w = features.shape
        if n != 1:
            raise ValueError('当前第三阶段仅处理一个左相机 context')
        prob, disp = [x.float() for x in disparity]  # prob：[B,104,96,320] 视差概率；disp：[B,1,384,1280] 左图视差，单位像素。
        if prob.shape[1] != self.disparity_channels:
            raise ValueError(f'视差通道 {prob.shape[1]} 与配置 {self.disparity_channels} 不符')
        #* ================== 2.2.2 预测视差 → 米制深度图 ==================
        #* geometry.py：按 Z=f*b/disparity 转换，使用与图像变换匹配的焦距，无效视差对应深度 0。
        # depth=[B,1,384,1280]，用于下面的深度引导，以及后续生成 proposal 几何候选。
        depth = disparity_to_depth(disp, baseline, cam_params)

        #* ================== 2.2.3 图像特征 + 标定 + 深度 → context ==================
        # get_mlp_input 将几何参数编码为 33 维标定输入；ContextNet 输出 [B,128,48,160]。
        context = self.context_net(features.reshape(b, c, h, w), get_mlp_input(*cam_params))
        #* 仅对深度引导副本截断、归一化，不改变最终返回的米制 depth，也不使用 GT 深度。
        guide = depth.clamp(0, self.depth_bound[1])
        # 全无效深度保留零输入；不伪造深度监督。
        guide = normalize(normalize_depth_to_255(guide), [.48], [.28])
        context = self.dformerv2(context, guide)[0]  # 深度引导的二维特征，保持 [B,128,48,160]。

        #* ================== 2.2.4 视差概率 → 可学习深度概率 ==================
        #* 这是可学习的视差概率→深度概率映射，不是简单按 d=fb/z 换通道。
        # 将 104 个视差通道转换为 112 个深度通道，对应 depth_bound=(2,58,0.5)。
        # 112=(58-2)/0.5：通道 k 对应相机 Z 方向的深度采样位置 2+0.5*k 米，
        # 即第 0/1/.../111 通道对应 2.0/2.5/.../57.5 米，不是到相机的欧氏距离。
        #* 通道对应的位置是米制的，但 logits 中的数值是未归一化分数，不是深度米数；
        #* 下面 softmax 后才得到各深度位置的概率权重，同一像素的 112 个概率之和约为 1。
        # 后续 LSS 按这些深度位置反投影，并用相应概率加权 context，而非直接回归一个深度值。
        logits = self.stereo_volume_encoder(self.disp2depth_mlp(prob))
        # 插值前：logits=[B,112,96,320]；softmax(1) 沿深度通道归一化，shape 不变。
        # 插值后：depth_prob=[B,112,48,160]，其中 (h,w) 来自输入 features，与 context 空间尺寸一致。
        # bilinear 仅将图像空间 H/W 下采样一半，不改变 112 个深度通道及其米制采样位置。
        # align_corners=True 表示输入、输出角点像素中心对齐；上述尺寸为当前配置示例。
        # 输出供 LSS 沿深度加权提升及后续细化使用，不是将深度图从米转换为其他单位。
        depth_prob = F.interpolate(logits.softmax(1), size=(h, w), mode='bilinear', align_corners=True)

        #* ================== 2.2.5 返回二维特征与两种深度信息 ==================
        # context[:,None]=[B,1,128,48,160]：补回左相机维；depth_prob=[B,112,48,160]。
        # depth=[B,1,384,1280]：米制深度图；它与 depth_prob 的概率分布含义不同。
        return context[:, None], depth_prob, depth
