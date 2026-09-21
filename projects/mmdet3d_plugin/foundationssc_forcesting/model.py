"""当前双目 + 已知未来自车位姿 → 各未来 LiDAR 坐标系的四步 occupancy。"""
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.models import DETECTORS
from ..foundationssc.image_model import FoundationSSCImageModel


class PoseVoxelAttention(nn.Module):
    #! 三维可变形交叉注意力：未来网格 query 从上一时刻 voxel memory 采样。
    def __init__(self, channels, pc_range, heads=4, points=4):
        super().__init__()
        if min(heads, points) < 1 or channels % heads:
            raise ValueError('channels 必须整除 heads；heads/points 必须为正')
        self.heads, self.points = heads, points
        self.register_buffer('bounds', torch.tensor(pc_range, dtype=torch.float32), persistent=False)
        self.position = nn.Linear(3, channels)
        self.value = nn.Conv3d(channels, channels, 1)
        self.offset = nn.Linear(channels, heads * points * 3)
        self.weight = nn.Linear(channels, heads * points)
        self.output = nn.Linear(channels, channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))
        self.norm = nn.LayerNorm(channels)
        nn.init.normal_(self.offset.weight, std=.01)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.ffn[-1].weight, std=1e-3)
        nn.init.zeros_(self.ffn[-1].bias)

    def reference_grid(self, state, target_to_memory):
        b, c, x, y, z = state.shape
        axes = [(torch.arange(n, device=state.device, dtype=state.dtype) + .5) / n for n in (x, y, z)]
        centers = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        bounds = self.bounds.to(state)
        metric = centers * (bounds[3:] - bounds[:3]) + bounds[:3]
        #! 标准列向量 T：p_memory=T @ p_target；这里坐标数组按行存放，所以乘 R.T。
        metric = metric[None] @ target_to_memory[:, :3, :3].transpose(1, 2) + target_to_memory[:, None, :3, 3]
        reference = (metric - bounds[:3]) / (bounds[3:] - bounds[:3])
        return centers, reference

    @staticmethod
    def sample(value, locations):
        #! volume=[B,C,X,Y,Z] 对应 grid_sample 的 [N,C,D,H,W]。
        #! grid 最后一维必须为 (Z,Y,X)，不是物理坐标 (X,Y,Z)；越界特征置零。
        grid = (locations[..., [2, 1, 0]] * 2 - 1).unsqueeze(3)
        return F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(-1)

    def forward(self, state, target_to_memory):
        b, c, x, y, z = state.shape
        centers, reference = self.reference_grid(state, target_to_memory)
        #! 一次处理全部 X*Y*Z 个 query，不分块；value 为上一时刻完整三维 memory。
        values = self.value(state).reshape(b * self.heads, c // self.heads, x, y, z)
        scale = state.new_tensor([x, y, z])
        qn = centers.shape[0]
        base = self.sample(state, reference[:, :, None]).squeeze(-1).transpose(1, 2)
        query = base + self.position(centers * 2 - 1)[None]
        #! offset 为 memory 三维体素单位；位姿先补偿自车运动，学习偏移再寻找场景信息。
        offset = self.offset(query).reshape(b, qn, self.heads, self.points, 3) / scale
        locations = reference[:, :, None, None] + offset
        locations = locations.permute(0, 2, 1, 3, 4).reshape(b * self.heads, qn, self.points, 3)
        sampled = self.sample(values, locations).reshape(b, self.heads, c // self.heads, qn, self.points)
        weights = self.weight(query).reshape(b, qn, self.heads, self.points).softmax(-1)
        update = (sampled * weights.permute(0, 2, 1, 3)[:, :, None]).sum(-1)
        update = update.permute(0, 3, 1, 2).reshape(b, qn, c)
        result = base + self.output(update)
        result = result + self.ffn(self.norm(result))
        return result.transpose(1, 2).reshape(b, c, x, y, z)


@DETECTORS.register_module()
class FoundationSSCForecastModel(FoundationSSCImageModel):
    #! 保持父类参数路径，load_from 可加载已训练的单帧 checkpoint；仅 dynamics 是新参数。
    def __init__(self, future_steps=4, attention_heads=4, sampling_points=4, **kwargs):
        super().__init__(**kwargs)
        if future_steps < 1:
            raise ValueError('future_steps 必须为正整数')
        if self.use_depth_loss or self.use_semantic_loss:
            raise ValueError('冻结前端的最小预测基线应关闭辅助监督')
        self.future_steps = int(future_steps)
        channels = kwargs['voxel_encoder'].get('channels', 128)
        #! 先冻结原模型，再创建预测器；解码器虽冻结，未来分支仍需对输入特征求梯度。
        self.requires_grad_(False)
        self.dynamics = PoseVoxelAttention(channels, kwargs['voxel_encoder']['point_cloud_range'],
                                          attention_heads, sampling_points)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for name, module in self.named_children():
            module.train(mode if name == 'dynamics' else False)
        return self

    def _features(self, img_inputs, img_metas):
        #! 感知前端只访问当前图像；未来真实位姿在后续 attention 中作为 oracle 条件。
        with torch.no_grad():
            image = self.extract_image_features(img_inputs, img_metas)
            return self.voxel_encoder(image, img_inputs, img_metas)['voxel_feats']

    def _decode(self, features):
        #! 不使用 no_grad：未来 loss 必须穿过冻结解码器反传到 dynamics。
        encoded = self.occ_encoder_neck(self.occ_encoder_backbone(features))
        return self.pts_bbox_head([encoded[0]])['output_voxels']

    def forward_train(self, img_inputs, img_metas, gt_occ, **kwargs):
        if gt_occ.ndim != 5 or gt_occ.shape[1] != self.future_steps + 1:
            raise ValueError('gt_occ 应为 [B,1+future_steps,X,Y,Z]')
        state = self._features(img_inputs, img_metas)
        # pipeline 已构造 [B,K,4,4]：每步目标→上一时刻，不是目标→初始当前帧。
        # 列向量格式，平移在最后一列；GT 保留各帧自身 LiDAR 坐标系。
        transforms = img_metas['forecast_target_to_previous'].to(state)  # 仅转换 device/dtype。
        losses = {}
        for step in range(1, self.future_steps + 1):
            #! 自回归四步，共享参数，默认不 detach；未来损失按步平均。
            state = self.dynamics(state, transforms[:, step - 1])
            step_losses = self.pts_bbox_head.loss(self._decode(state), gt_occ[:, step])
            losses.update({f'{key}_step_{step}': value / self.future_steps
                           for key, value in step_losses.items()})
        return losses

    def forward_test(self, img_inputs, img_metas, gt_occ=None, return_outputs=False, **kwargs):
        #! 当前+四步分别处于自己的 LiDAR 坐标系；直接与对应帧原始 occupancy 比较。
        state = self._features(img_inputs, img_metas)
        # 与训练一致：[B,K,4,4]，列向量格式的目标→上一时刻位姿，由 pipeline 构造。
        transforms = img_metas['forecast_target_to_previous'].to(state)  # 仅转换 device/dtype。
        predictions = []
        for step in range(self.future_steps + 1):
            if step:
                state = self.dynamics(state, transforms[:, step - 1])
            predictions.append(self._decode(state).argmax(1))
        pred = torch.stack(predictions, 1)
        if return_outputs or gt_occ is None:
            return dict(pred=pred)
        if gt_occ.shape != pred.shape:
            raise ValueError('未来预测与 GT 的时间/空间尺寸不一致')
        #! 复用父类逐样本统计和 token 去重协议，Dataset 汇总各步 IoU/mIoU。
        results = None
        for step in range(self.future_steps + 1):
            items = self.occupancy_results(pred[:, step], gt_occ[:, step], img_metas)
            if results is None:
                results = items
            else:
                for result, item in zip(results, items):
                    result['hist_for_iou_per_frame'].extend(item['hist_for_iou_per_frame'])
        return results
