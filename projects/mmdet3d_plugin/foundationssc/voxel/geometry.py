"""列向量几何：增强像素 ↔ 相机 ↔ LiDAR，和 [X,Y,Z] 体素汇聚。"""
import math
import torch
from torch import nn
from ..ops import lift_pool, use_cuda


def unproject(uvd, cam_params):
    """uvd: [B,N,...,3]，第三维是相机 Z 深度（米），不是射线长度。"""
    rot, trans, intrinsic, post_rot, post_trans, bda = cam_params
    b, n = rot.shape[:2]
    shape = uvd.shape
    p = uvd.reshape(b, n, -1, 3)
    p = torch.linalg.solve(post_rot, (p - post_trans[:, :, None]).transpose(-1, -2)).transpose(-1, -2)
    p = torch.cat((p[..., :2] * p[..., 2:], p[..., 2:]), -1)
    #* 当前 PKL K4 第四列为零；也支持原 KITTI P2/P3 非零第四列。
    if intrinsic.shape[-1] == 4:
        p = p - intrinsic[:, :, None, :3, 3]
    p = torch.linalg.solve(intrinsic[..., :3, :3], p.transpose(-1, -2)).transpose(-1, -2)
    p = p @ rot.transpose(-1, -2) + trans[:, :, None]
    p = p @ bda[:, None, :3, :3].transpose(-1, -2)
    if bda.shape[-1] == 4:
        p = p + bda[:, None, None, :3, 3]
    return p.reshape(shape)


def project(points, cam_params):
    """[B,N,...,3] LiDAR/BDA 点 → 增强后的 [u,v,相机Z]；保留负深度供 mask。"""
    rot, trans, intrinsic, post_rot, post_trans, bda = cam_params
    shape = points.shape
    p = points.reshape(*rot.shape[:2], -1, 3)
    if bda.shape[-1] == 4:
        p = p - bda[:, None, None, :3, 3]
    p = torch.linalg.solve(bda[:, None, :3, :3], p.transpose(-1, -2)).transpose(-1, -2)
    p = torch.linalg.solve(rot, (p - trans[:, :, None]).transpose(-1, -2)).transpose(-1, -2)
    p = p @ intrinsic[..., :3, :3].transpose(-1, -2)
    if intrinsic.shape[-1] == 4:
        p = p + intrinsic[:, :, None, :3, 3]
    p = torch.cat((p[..., :2] / p[..., 2:].clamp_min(1e-6), p[..., 2:]), -1)
    p = p @ post_rot.transpose(-1, -2) + post_trans[:, :, None]
    return p.reshape(shape)


def disparity_to_depth(disparity, baseline, cam_params):
    """双目只支持相同水平缩放、无旋转/翻转；f 与视差必须同为增强后像素单位。"""
    _, _, intrinsic, post_rot, _, _ = cam_params
    focal = (post_rot @ intrinsic[..., :3, :3])[:, 0, 0, 0]
    baseline = baseline.to(disparity).reshape(-1)
    if (baseline <= 0).any() or (focal <= 0).any():
        raise ValueError('焦距与基线必须为正')
    valid = torch.isfinite(disparity) & (disparity > 1e-6)
    depth = focal[:, None, None, None] * baseline[:, None, None, None] / disparity.clamp_min(1e-6)
    return torch.where(valid, depth, torch.zeros_like(depth))


class VoxelGeometry(nn.Module):
    def __init__(self, point_cloud_range, voxel_shape, depth_bound, input_size, downsample=8, pool_chunk=8, ops_backend='pytorch'):
        super().__init__()
        self.voxel_shape = tuple(voxel_shape)  # X,Y,Z，而非 Conv3d 通常命名的 D,H,W。
        self.input_size = tuple(input_size)
        self.downsample = downsample
        self.pool_chunk = pool_chunk
        self.ops_backend = ops_backend
        if len(voxel_shape) != 3 or any(int(x) != x or x <= 0 for x in voxel_shape):
            raise ValueError('voxel_shape 必须为三个正整数')
        if pool_chunk < 1 or any(x % downsample for x in input_size):
            raise ValueError('pool_chunk 必须为正，图像尺寸必须整除下采样倍数')
        lower = torch.tensor(point_cloud_range[:3], dtype=torch.float32)
        extent = torch.tensor(point_cloud_range[3:], dtype=torch.float32) - lower
        if (extent <= 0).any() or depth_bound[2] <= 0 or depth_bound[1] <= depth_bound[0]:
            raise ValueError('空间范围/深度范围非法')
        self.register_buffer('lower', lower)
        self.register_buffer('voxel_size', extent / torch.tensor(voxel_shape))
        self.register_buffer('depth_bins', torch.arange(*depth_bound, dtype=torch.float32))
        ijk = torch.stack(torch.meshgrid(*(torch.arange(x) for x in voxel_shape), indexing='ij'), -1)
        self.register_buffer('centers', lower + (ijk.float() + .5) * self.voxel_size)

    def indices(self, points):
        #* 必须 floor，不能直接 long：-0.1 格不能被截断到第 0 格。
        idx = torch.floor((points - self.lower) / self.voxel_size).long()
        valid = torch.isfinite(points).all(-1) & (idx >= 0).all(-1)
        valid &= (idx < idx.new_tensor(self.voxel_shape)).all(-1)
        linear = (idx[..., 0] * self.voxel_shape[1] + idx[..., 1]) * self.voxel_shape[2] + idx[..., 2]
        return linear, valid

    def frustum(self, height, width, depths):
        ys = torch.linspace(0, self.input_size[0] - 1, height, device=depths.device)
        xs = torch.linspace(0, self.input_size[1] - 1, width, device=depths.device)
        dd, yy, xx = torch.meshgrid(depths, ys, xs, indexing='ij')
        return torch.stack((xx, yy, dd), -1)

    def lift_splat(self, context, depth_prob, cam_params):
        """LSS: context[B,1,C,h,w] × 深度概率[B,D,h,w] → [B,C,X,Y,Z]。"""
        b, n, c, h, w = context.shape
        if n != 1 or depth_prob.shape != (b, len(self.depth_bins), h, w):
            raise ValueError('LSS 当前支持单个左相机，深度与 context 尺寸必须一致')
        cuda = use_cuda(self.ops_backend, context)
        out = None if cuda else context.new_zeros((b * math.prod(self.voxel_shape), c))
        indices = []
        hits = torch.zeros(b, device=context.device, dtype=torch.long)
        #* 按深度分块 Lift，避免一次构造 [B,C,D,h,w]；index_add 等价于按 voxel 求和。
        for start in range(0, len(self.depth_bins), self.pool_chunk):
            ds = self.depth_bins[start:start + self.pool_chunk]
            uvd = self.frustum(h, w, ds)[None, None].expand(b, n, -1, -1, -1, -1)
            xyz = unproject(uvd, cam_params)
            idx, valid = self.indices(xyz)
            hits += valid.reshape(b, -1).sum(-1)
            if cuda:
                #* 先收集 voxel id；下方由原 FoundationSSC Lift→bev_pool 路径汇聚。
                indices.append(torch.where(valid, idx, -1).reshape(b, len(ds), h * w))
                continue
            idx = idx + torch.arange(b, device=idx.device).view(b, 1, 1, 1, 1) * math.prod(self.voxel_shape)
            lifted = (context.unsqueeze(3) * depth_prob[:, None, None, start:start + len(ds)]).permute(0, 1, 3, 4, 5, 2)
            out.index_add_(0, idx[valid], lifted[valid])
        if cuda:
            out = lift_pool(context[:, 0].reshape(b, c, h * w), depth_prob.flatten(2),
                            torch.cat(indices, 1), self.voxel_shape)
            return out.reshape(b, c, *self.voxel_shape), hits
        return out.reshape(b, *self.voxel_shape, c).permute(0, 4, 1, 2, 3).contiguous(), hits

    @torch.no_grad()
    def proposal(self, depth, cam_params):
        """真实立体预测深度反投影成二值候选，不用 GT，也不在空输入时制造随机点。"""
        b, _, h, w = depth.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing='ij')
        pixels = torch.stack((xx, yy), -1).to(depth)[None].expand(b, -1, -1, -1)
        xyz = unproject(torch.cat((pixels, depth.permute(0, 2, 3, 1)), -1)[:, None], cam_params)
        idx, valid = self.indices(xyz)
        valid &= depth[:, None, 0] > 0
        result = depth.new_zeros((b, math.prod(self.voxel_shape)))
        for i in range(b):
            result[i, idx[i][valid[i]]] = 1
        return result.reshape(b, 1, *self.voxel_shape)
