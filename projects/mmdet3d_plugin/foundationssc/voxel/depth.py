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
        b, n, c, h, w = features.shape
        if n != 1:
            raise ValueError('当前第三阶段仅处理一个左相机 context')
        prob, disp = [x.float() for x in disparity]
        if prob.shape[1] != self.disparity_channels:
            raise ValueError(f'视差通道 {prob.shape[1]} 与配置 {self.disparity_channels} 不符')
        depth = disparity_to_depth(disp, baseline, cam_params)
        context = self.context_net(features.reshape(b, c, h, w), get_mlp_input(*cam_params))
        guide = depth.clamp(0, self.depth_bound[1])
        #* 全无效深度保留零输入；不伪造深度监督。
        guide = normalize(normalize_depth_to_255(guide), [.48], [.28])
        context = self.dformerv2(context, guide)[0]
        #* 这是可学习的视差概率→深度概率映射，不是简单按 d=fb/z 换通道。
        logits = self.stereo_volume_encoder(self.disp2depth_mlp(prob))
        depth_prob = F.interpolate(logits.softmax(1), size=(h, w), mode='bilinear', align_corners=True)
        return context[:, None], depth_prob, depth
