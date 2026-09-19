"""第三阶段总入口：DSGP → LSS / proposal+VoxFormer → 三平面门控融合。"""
import torch
from torch import nn
from .depth import GeometryDepthNet
from .geometry import VoxelGeometry
from .refiner import VoxelRefiner
from .fusion import DualFeatFusion_Tri


class FoundationVoxelEncoder(nn.Module):
    def __init__(self, point_cloud_range=(0, -25.6, -2, 51.2, 25.6, 4.4),
                 voxel_shape=(128, 128, 16), input_size=(384, 1280),
                 depth_bound=(2., 58., .5), input_channels=640, channels=128,
                 disparity_channels=104, downsample=8, pool_chunk=8,
                 depth_cfg=None, refiner_cfg=None, fusion_groups=16, ops_backend='pytorch'):
        super().__init__()
        self.depth_bound = depth_bound
        if ops_backend not in ('pytorch', 'cuda', 'auto'):
            raise ValueError('ops_backend 必须是 pytorch/cuda/auto')
        self.ops_backend = ops_backend
        self.geometry = VoxelGeometry(point_cloud_range, voxel_shape, depth_bound, input_size, downsample, pool_chunk, ops_backend)
        self.depth_net = GeometryDepthNet(input_channels, channels, disparity_channels, depth_bound, **(depth_cfg or {}))
        self.refiner = VoxelRefiner(voxel_shape, channels, ops_backend=ops_backend, **(refiner_cfg or {}))
        self.fusion = DualFeatFusion_Tri(channels, channels, norm_type='group', num_groups=fusion_groups)

    def forward(self, image_output, img_inputs, img_metas):
        features = image_output['img_feats'].float()
        device = features.device
        #* 双目只用于 stereo，后续 context/projection 只使用左相机；BDA 不带 camera 维。
        cam = [x[:, :1].to(device=device, dtype=torch.float32) for x in img_inputs[1:6]]
        cam.append(img_inputs[6].to(device=device, dtype=torch.float32))
        if features.shape[-2:] != tuple(x // self.geometry.downsample for x in self.geometry.input_size):
            raise ValueError('图像特征大小与 voxel 配置 input_size/downsample 不一致')
        context, prob, stereo_depth = self.depth_net(features, image_output['disparity'], cam, img_metas['baseline'])
        coarse, hits = self.geometry.lift_splat(context, prob, cam)
        proposal = self.geometry.proposal(stereo_depth, cam)
        refined, visible = self.refiner(context, prob, coarse, proposal, self.geometry, cam, self.depth_bound)
        voxel = self.fusion(coarse, refined)
        return dict(voxel_feats=voxel, coarse_voxel=coarse, refined_voxel=refined,
                    context=context, depth_prob=prob, stereo_depth=stereo_depth,
                    proposal=proposal, lifted_valid_points=hits, visible_voxels=visible)
