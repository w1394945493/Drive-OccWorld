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
        #* ================== 2.3.1 输入检查与汇聚容器 ==================
        #* 默认 C=128、D=112、h=48、w=160；N=1 表示只提升左图特征，非双目两路分别提升。
        #* X/Y/Z=128/128/16 为输出体素网格数，和深度采样数 D 是不同概念。
        b, n, c, h, w = context.shape
        if n != 1 or depth_prob.shape != (b, len(self.depth_bins), h, w):
            raise ValueError('LSS 当前支持单个左相机，深度与 context 尺寸必须一致')

        cuda = use_cuda(self.ops_backend, context)  # 根据后端配置及输入设备选择 CUDA / PyTorch 路径。
        out = None if cuda else context.new_zeros((b * math.prod(self.voxel_shape), c))  # PyTorch 累加缓冲区：[B*X*Y*Z,C]。
        indices = []  # CUDA 分支保存各深度块的体素索引，最后拼接为 [B,D,h*w]。
        hits = torch.zeros(b, device=context.device, dtype=torch.long)  # 每个样本的有效采样点数；多个点可落入同一体素。

        #* ================== 2.3.2 像素 + 深度 → 三维坐标 → 体素索引 ==================
        #* K=len(ds) 为当前深度块大小（默认最多 8）；所有块合起来仍覆盖完整的 D 个深度位置。
        #* 两种后端都分块计算几何；仅 PyTorch 分块 Lift，CUDA 后续 lift_pool 仍会构造完整 lifted 特征。
        for start in range(0, len(self.depth_bins), self.pool_chunk):
            ds = self.depth_bins[start:start + self.pool_chunk]  # [K]，相机 Z 深度，单位米。
            uvd = self.frustum(h, w, ds)[None, None].expand(b, n, -1, -1, -1, -1)  # [B,1,K,h,w,3]，末维为 (u,v,depth)。
            #* u/v 是预处理图像上的像素坐标，不是特征图下标；frustum 将 h*w 个位置铺到输入图像范围。
            xyz = unproject(uvd, cam_params)  # [B,1,K,h,w,3]：撤销图像变换 → 内参反投影 → camera→LiDAR → BDA。
            idx, valid = self.indices(xyz)  # 均为 [B,1,K,h,w]；idx=(ix*Y+iy)*Z+iz，valid 排除越界/非有限坐标。
            hits += valid.reshape(b, -1).sum(-1)  # 几何有效点数，不按深度概率筛选，也不是非空体素数。
            if cuda:
                #* CUDA 暂不生成加权特征，只保存每个采样点落入的体素；无效点标为 -1。
                indices.append(torch.where(valid, idx, -1).reshape(b, len(ds), h * w))
                continue

            # 2.3.3 PyTorch 备用路径：概率加权 Lift + 同体素求和。
            # 当前 ops_backend='cuda' 时，前面的 continue 会跳过本段；实际汇聚在循环后的 lift_pool 中执行。
            # 以下仅在 use_cuda 返回 False 时执行，保留用于 PyTorch 后端调试和对照。
            # 为不同 batch 添加体素编号偏移，避免不同样本被累加到同一位置。
            idx = idx + torch.arange(b, device=idx.device).view(b, 1, 1, 1, 1) * math.prod(self.voxel_shape)
            # [B,1,C,1,h,w] × [B,1,1,K,h,w] → [B,1,C,K,h,w] → [B,1,K,h,w,C]。
            # 每个像素的 C 维 context 被分配到 K 个深度位置，分别乘对应深度概率。
            lifted = (context.unsqueeze(3) * depth_prob[:, None, None, start:start + len(ds)]).permute(0, 1, 3, 4, 5, 2)
            out.index_add_(0, idx[valid], lifted[valid])  # Splat：有效点按体素编号累加 C 维特征，是求和而非平均。


        #* ================== 2.3.4 CUDA 汇聚与统一输出 ==================
        #* 与 2.3.3 数学等价：体素特征 = Σ(落入该体素的 context × 对应深度概率)，是求和而非平均。
        #* 区别：2.3.3 分块加权并用 index_add_ 累加；此处拼接全部索引，由 lift_pool 加权后调用 bev_pool。
        #* 两条路径二选一，不重复汇聚；均过滤无效点，并支持 context/depth_prob 的梯度回传。
        #* 浮点求和顺序不同，输出和梯度可能存在微小数值差异，不保证逐位一致。
        if cuda:
            #* ops/functional.py 的 lift_pool：context=[B,C,h*w]，概率/索引=[B,D,h*w]。
            #* 内部同样先做 context×概率，再过滤无效点并调用原 bev_pool 汇聚，输出 [B,C,X*Y*Z]。
            out = lift_pool(context[:, 0].reshape(b, c, h * w), depth_prob.flatten(2),
                            torch.cat(indices, 1), self.voxel_shape)
            return out.reshape(b, c, *self.voxel_shape), hits


        #* 两种后端均返回 coarse=[B,C,X,Y,Z] 与 hits=[B]；无点贡献的体素特征为零。
        #* 特征/深度概率可反传梯度；floor/整数索引决定的体素分配本身不可微。
        return out.reshape(b, *self.voxel_shape, c).permute(0, 4, 1, 2, 3).contiguous(), hits

    @torch.no_grad()
    def proposal(self, depth, cam_params):
        """真实立体预测深度反投影成二值候选，不用 GT，也不在空输入时制造随机点。"""
        #* ================== 2.4.1 预测深度与像素坐标准备 ==================
        #* 本方法不计算梯度：仅生成离散候选 mask，不学习体素特征，也不使用 GT 深度/占据标签。
        # depth 为左图预测的相机 Z 深度（米），默认 [B,1,384,1280]，不是 112 通道的深度概率。
        b, _, h, w = depth.shape
        yy, xx = torch.meshgrid(torch.arange(h, device=depth.device), torch.arange(w, device=depth.device), indexing='ij')  # 各为 [H,W]，对应预处理后左图的像素行/列。
        pixels = torch.stack((xx, yy), -1).to(depth)[None].expand(b, -1, -1, -1)  # [B,H,W,2]，末维为 (u,v)。

        #* ================== 2.4.2 每个像素反投影到一个三维位置 ==================
        # 拼接为 [B,1,H,W,3] 的 (u,v,depth)，其中长度 1 的维度表示左相机。
        # unproject：撤销图像变换 → 内参反投影 → camera→LiDAR → BDA；输出 xyz 同 shape，单位米。
        xyz = unproject(torch.cat((pixels, depth.permute(0, 2, 3, 1)), -1)[:, None], cam_params)
        idx, valid = self.indices(xyz)  # [B,1,H,W]：线性体素编号与有效 mask；过滤越界/非有限坐标。
        valid &= depth[:, None, 0] > 0  # [B,1,H,W]：额外排除无效视差对应的零深度。

        #* ================== 2.4.3 命中体素标为候选，供后续图像交叉注意力使用 ==================
        #* 与 LSS 不同：这里只标记位置，不做 context×概率加权，也不累计命中次数。
        result = depth.new_zeros((b, math.prod(self.voxel_shape)))  # [B,X*Y*Z]，浮点 0/1 mask，device/dtype 跟随 depth。
        for i in range(b):
            result[i, idx[i][valid[i]]] = 1  # 同一体素即使被多个像素命中也只标为 1；没有有效点则保持全零。
        #* 1 表示预测深度命中的候选；0 仅表示未命中，不能当作已知空闲区域或最终占据预测。
        return result.reshape(b, 1, *self.voxel_shape)  # 默认 [B,1,128,128,16]，轴序 [B,1,X,Y,Z]。
