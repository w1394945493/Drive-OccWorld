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
        #*=============== 二、voxel 特征构建：2.1～2.6 ===============
        #** 2.1 输入准备：提取左图融合特征和几何标定
        # image_output 来自图像骨干/金字塔；默认 img_feats=[B,1,640,48,160]。
        # img_inputs 保存图像及标定，img_metas 提供双目基线；不读取辅助监督 GT。
        features = image_output['img_feats'].float()
        device = features.device
        #* cam 保存几何参数，不是图像或图像特征；双目用于 stereo，后续体素构建以左图为基准。
        #* 输入相机顺序为 [左, 右]；x[:, :1] 选左相机，并保留长度为 1 的相机维。
        #* img_inputs[1:6] 依次为 camera→LiDAR 旋转、平移、内参、图像变换旋转、平移。
        #* 选取后 shape 依次为 [B,1,3,3]、[B,1,3]、[B,1,4,4]、[B,1,3,3]、[B,1,3]；
        #* 后两项记录 resize/crop 等图像变换，确保投影坐标与预处理后的左图对应。
        cam = [x[:, :1].to(device=device, dtype=torch.float32) for x in img_inputs[1:6]]  # 左相机参数，与 features 同设备。
        cam.append(img_inputs[6].to(device=device, dtype=torch.float32))  # BDA=[B,4,4]：公共三维空间增强矩阵（当前为单位阵），无相机维，不切片。
        if features.shape[-2:] != tuple(x // self.geometry.downsample for x in self.geometry.input_size):
            raise ValueError('图像特征大小与 voxel 配置 input_size/downsample 不一致')

        #*=============== 2.2 二维 context 与深度信息 ===============
        #** 2.2 depth.py：图像特征 + 预测视差 + 标定 → context、depth_prob、stereo_depth
        # context=[B,1,128,48,160]：用于后续二维到三维的特征提取。
        # prob=[B,112,48,160]：可学习的深度概率，对应 depth_bound=(2,58,0.5)。
        # stereo_depth=[B,1,384,1280]：按 Z=f*b/disparity 转换的预测深度，单位米。
        # 两种深度用途不同：prob 用于概率加权；stereo_depth 用于生成几何候选。
        context, prob, stereo_depth = self.depth_net(features, image_output['disparity'], cam, img_metas['baseline'])

        #*=============== 2.3 LSS 粗体素分支 ===============
        #** 2.3 geometry.lift_splat：将 context 沿深度分布提升到三维并汇聚进体素
        # coarse=[B,128,128,128,16]，轴序为 [B,C,X,Y,Z]。
        # hits=[B]：落入空间范围的有效像素-深度采样点数，不是非空体素数量。
        coarse, hits = self.geometry.lift_splat(context, prob, cam)

        #*=============== 2.4 几何候选体素生成 ===============
        #** 2.4 geometry.proposal：将立体深度反投影到 LiDAR 空间，标记命中的体素
        # proposal=[B,1,128,128,16]：候选 mask，不是语义类别或最终占据预测。
        # 每个左图像素使用一个预测深度定位三维表面；命中的体素标为 1，未命中标为 0。
        #* 用深度先验筛选后续与图像特征做交叉注意力的候选，不使用 occupancy GT。
        #* 候选不等于所有视野内体素，也不是严格的真实可见性判断；深度误差会影响候选位置。
        # 0 仅表示未被预测深度命中，不代表已知空闲；此步只生成离散 mask，不计算梯度。
        proposal = self.geometry.proposal(stereo_depth, cam)

        #*=============== 2.5 候选体素细化分支 ===============
        #** 2.5 refiner.py：候选体素从图像取特征，再通过自注意力传播到三维网格
        # 结合 context/深度概率做交叉注意力；非候选体素使用 prior，随后自注意力细化。
        # refined 与 coarse 同形状；visible=[B] 为投影在图像视野内的体素数量，
        # 不是 mask，也不表示经过真实遮挡判断后的可见数量。
        refined, visible = self.refiner(context, prob, coarse, proposal, self.geometry, cam, self.depth_bound)

        #*=============== 2.6 双分支融合与输出 ===============
        #** 2.6 fusion.py：通过三平面门控融合 coarse 与 refined
        # voxel=[B,128,128,128,16]：最终体素特征，送给第三部分的 3D 编码器/占据头。
        # 此处尚未生成 [B,20,256,256,32] 的类别 logits；其余返回项供辅助损失和调试使用。
        voxel = self.fusion(coarse, refined)

        
        #* 输出字典：以下 shape 以当前默认配置为例；三维特征轴序均为 [B,C,X,Y,Z]。
        # voxel_feats 是后续占据预测的主输入；其余字段用于辅助监督、分支检查或可视化。
        return dict(
            voxel_feats=voxel,       # [B,128,128,128,16]：coarse/refined 经三平面门控融合的最终体素特征，不是类别 logits。
            coarse_voxel=coarse,     # [B,128,128,128,16]：context 按深度概率加权并经 LSS 汇聚得到的粗体素特征。
            refined_voxel=refined,   # [B,128,128,128,16]：候选图像交叉注意力 + 全体素自注意力得到的细化特征。
            context=context,        # [B,1,128,48,160]：深度引导的左图二维特征；1 为相机维，也供辅助二维语义头使用。
            depth_prob=prob,        # [B,112,48,160]：各米制深度位置的概率，沿 112 通道求和约为 1；供 LSS/DFA3D 及辅助深度损失使用。
            stereo_depth=stereo_depth,  # [B,1,384,1280]：预测视差换算的相机 Z 深度图（米），非 GT；供深度引导和 proposal 构建。
            proposal=proposal,      # [B,1,128,128,16]：浮点 0/1 候选 mask，无梯度；1 为预测深度命中，0 不代表已知空闲。
            lifted_valid_points=hits,  # [B]，int64：LSS 中落入体素范围的像素-深度采样点数，重复命中也计数，不是体素数。
            visible_voxels=visible, # [B]，int64：体素中心投影在左图视野内且相机深度为正的数量，非遮挡可见性判断，也不是 proposal 数量。
        )
