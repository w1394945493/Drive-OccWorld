"""当前双目 + 已知未来自车位姿 → 各未来 LiDAR 坐标系的四步 occupancy。"""
import torch
from contextlib import nullcontext
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
        #* 小幅初始化输出残差，使初始状态接近位姿重采样；并非全部参数置零。
        nn.init.normal_(self.offset.weight, std=.01)
        nn.init.zeros_(self.offset.bias)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)
        nn.init.normal_(self.ffn[-1].weight, std=1e-3)
        nn.init.zeros_(self.ffn[-1].bias)

    def reference_grid(self, state, target_to_memory):
        #! 新增几何对齐：未来体素中心 → 上一时刻 memory 坐标，作为注意力采样参考点。
        b, c, x, y, z = state.shape
        axes = [(torch.arange(n, device=state.device, dtype=state.dtype) + .5) / n for n in (x, y, z)]
        centers = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        bounds = self.bounds.to(state)
        metric = centers * (bounds[3:] - bounds[:3]) + bounds[:3]
        # 标准列向量 T：p_memory=T @ p_target；这里坐标数组按行存放，所以乘 R.T。
        metric = metric[None] @ target_to_memory[:, :3, :3].transpose(1, 2) + target_to_memory[:, None, :3, 3]
        reference = (metric - bounds[:3]) / (bounds[3:] - bounds[:3])
        return centers, reference

    @staticmethod
    def sample(value, locations):
        # volume=[B,C,X,Y,Z] 对应 grid_sample 的 [N,C,D,H,W]。
        #* grid 最后一维必须为 (Z,Y,X)，不是物理坐标 (X,Y,Z)；越界特征置零。
        grid = (locations[..., [2, 1, 0]] * 2 - 1).unsqueeze(3)
        return F.grid_sample(value, grid, mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(-1)

    def forward(self, state, target_to_memory):
        b, c, x, y, z = state.shape
        centers, reference = self.reference_grid(state, target_to_memory)
        #* 一次处理全部 X*Y*Z 个 query，不分块；value 为上一时刻完整三维 memory。
        values = self.value(state).reshape(b * self.heads, c // self.heads, x, y, z)
        scale = state.new_tensor([x, y, z])
        qn = centers.shape[0]
        #! 以位姿重采样的 base + 目标位置编码构造未来 query，保留原场景信息。
        base = self.sample(state, reference[:, :, None]).squeeze(-1).transpose(1, 2)
        query = base + self.position(centers * 2 - 1)[None]
        #! offset 为 memory 三维体素单位；位姿先补偿自车运动，学习偏移再寻找场景信息。
        offset = self.offset(query).reshape(b, qn, self.heads, self.points, 3) / scale
        locations = reference[:, :, None, None] + offset
        locations = locations.permute(0, 2, 1, 3, 4).reshape(b * self.heads, qn, self.points, 3)
        #! 新增注意力汇聚：在学习得到的三维位置采样 memory，按各 head 的权重聚合。
        sampled = self.sample(values, locations).reshape(b, self.heads, c // self.heads, qn, self.points)
        weights = self.weight(query).reshape(b, qn, self.heads, self.points).softmax(-1)
        update = (sampled * weights.permute(0, 2, 1, 3)[:, :, None]).sum(-1)
        update = update.permute(0, 3, 1, 2).reshape(b, qn, c)
        #* 采样信息经残差和 FFN 更新，输出仍为三维连续特征，供下一步递推/解码。
        result = base + self.output(update)
        result = result + self.ffn(self.norm(result))
        return result.transpose(1, 2).reshape(b, c, x, y, z)


@DETECTORS.register_module()
class FoundationSSCForecastModel(FoundationSSCImageModel):
    #* 保持父类参数路径，load_from 可加载已训练的单帧 checkpoint；仅 dynamics 是新参数。
    def __init__(self, future_steps=4, attention_heads=4, sampling_points=4,
                 freeze_frontend=True, freeze_decoder=True, **kwargs):
        super().__init__(**kwargs)
        if future_steps < 1:
            raise ValueError('future_steps 必须为正整数')
        if self.use_depth_loss or self.use_semantic_loss:
            raise ValueError('当前 forecasting 损失接口尚未接入辅助监督，请关闭辅助损失')
        self.future_steps = int(future_steps)
        #* 默认复现原冻结策略；前端/解码器可分别解冻，FoundationStereo 始终冻结。
        self.freeze_frontend = freeze_frontend
        self.freeze_decoder = freeze_decoder
        channels = kwargs['voxel_encoder'].get('channels', 128)
        #* 先冻结原模型，再按开关解冻；解码器虽冻结，未来分支仍需对输入特征求梯度。
        self.requires_grad_(False)
        for name in ('image_pyramid', 'voxel_encoder'):
            getattr(self, name).requires_grad_(not freeze_frontend)
        for name in ('occ_encoder_backbone', 'occ_encoder_neck', 'pts_bbox_head'):
            getattr(self, name).requires_grad_(not freeze_decoder)
        #! 核心新增模块：未来各步共享这一套位姿条件三维注意力参数。
        self.dynamics = PoseVoxelAttention(channels, kwargs['voxel_encoder']['point_cloud_range'],
                                          attention_heads, sampling_points)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for name, module in self.named_children():
            #* 冻结模块保持 eval（包括 BN/Dropout），解冻模块跟随外层 train/eval。
            trainable = (name == 'dynamics'
                         or (name in ('image_pyramid', 'voxel_encoder') and not self.freeze_frontend)
                         or (name in ('occ_encoder_backbone', 'occ_encoder_neck', 'pts_bbox_head') and not self.freeze_decoder))
            module.train(mode and trainable)
        return self

    def _features(self, img_inputs, img_metas):
        #* 感知前端只访问当前图像；未来真实位姿在后续 attention 中作为 oracle 条件。
        #* 解冻时保留前端计算图；nullcontext 不会覆盖推理外层的 no_grad。
        # FoundationStereo 自身仍由父类 extract_image_features 内的 no_grad 冻结。
        with torch.no_grad() if self.freeze_frontend else nullcontext():
            image = self.extract_image_features(img_inputs, img_metas)
            return self.voxel_encoder(image, img_inputs, img_metas)['voxel_feats']

    def _decode(self, features):
        #* 复用原解码器，不使用 no_grad：未来 loss 必须穿过冻结解码器反传到 dynamics。
        encoded = self.occ_encoder_neck(self.occ_encoder_backbone(features))
        return self.pts_bbox_head([encoded[0]])['output_voxels']

    def forward_train(self, img_inputs, img_metas, gt_occ, **kwargs):
        #* ================== 1. 复用原单帧感知前端 ==================
        # gt_occ=[B,K+1,X,Y,Z]：第0项为当前标签，第1..K项为各未来帧原始标签。
        if gt_occ.ndim != 5 or gt_occ.shape[1] != self.future_steps + 1:
            raise ValueError('gt_occ 应为 [B,1+future_steps,X,Y,Z]')
        #* _features 复用 FoundationStereo → 图像金字塔 → voxel_encoder，仅运行一次。
        # state 为当前连续体素特征，默认 [B,128,128,128,16]；不是离散 occupancy。
        # 是否训练图像金字塔/voxel_encoder 由 freeze_frontend 控制，Stereo 始终冻结。
        state = self._features(img_inputs, img_metas)
        #! ================== 2. 新增未来位姿条件与自回归更新 ==================
        # pipeline 已构造 [B,K,4,4]：每步目标→上一时刻，不是目标→初始当前帧。
        # 列向量格式，平移在最后一列；GT 保留各帧自身 LiDAR 坐标系。
        transforms = img_metas['forecast_target_to_previous'].to(state)  # 仅转换 device/dtype。
        losses = {}
        for step in range(1, self.future_steps + 1):

            #! dynamics 是新增的三维位姿条件注意力，所有未来步调用同一个实例、共享参数。
            #* 输入为上一时刻特征，输出为本步目标坐标系特征；第二步起使用上一预测结果。
            #* 不重新提取未来图像特征，不 detach；后续步损失可反传到此前各次状态更新。
            state = self.dynamics(state, transforms[:, step - 1])

            #* ================== 3. 共享原占据解码器及损失 ==================
            #* _decode 复用同一套 3D ResNet + 3D FPN + OccHead，不为每个时间步复制网络。
            # logits 默认 [B,20,256,256,32]；复用原 CE/semantic scaling/geometric scaling 损失。
            #* freeze_decoder 仅控制解码器参数是否更新，不阻断 loss 到 dynamics 的梯度。
            #* 解码仅用于监督，下一步递推仍使用上面的 state，而不是 logits 或 argmax 标签。
            step_losses = self.pts_bbox_head.loss(self._decode(state), gt_occ[:, step])
            # 新增逐步损失命名并除以K；这里只监督未来1..K步，尚未计算当前帧损失。
            losses.update({f'{key}_step_{step}': value / self.future_steps
                           for key, value in step_losses.items()})
        return losses

    def forward_test(self, img_inputs, img_metas, gt_occ=None, return_outputs=False, **kwargs):
        #* ================== 1. 复用原单帧感知前端 ==================
        #* 当前+四步分别处于自己的 LiDAR 坐标系；直接与对应帧原始 occupancy 比较。
        # 与训练相同，只从当前双目提取一次 voxel 特征，不输入未来图像/occupancy GT。
        state = self._features(img_inputs, img_metas)
        # 与训练一致：[B,K,4,4]，列向量格式的目标→上一时刻位姿，由 pipeline 构造。
        transforms = img_metas['forecast_target_to_previous'].to(state)  # 仅转换 device/dtype。
        predictions = []
        #! ================== 2. 新增未来递推 + 共享原占据解码器 ==================
        for step in range(self.future_steps + 1):
            if step:
                #! 新增 dynamics：step=1..K 用同一套参数更新状态；step=0 保留原当前特征。
                state = self.dynamics(state, transforms[:, step - 1])
            #* 复用 _decode：当前和未来都共享同一套 3D ResNet + FPN + 占据头。
            # argmax 仅产生可视化/评估标签，不回灌到下一步 state。
            predictions.append(self._decode(state).argmax(1))
        pred = torch.stack(predictions, 1)  # [B,K+1,X,Y,Z]，顺序为当前、未来1..K。
        if return_outputs or gt_occ is None:
            return dict(pred=pred)
        if gt_occ.shape != pred.shape:
            raise ValueError('未来预测与 GT 的时间/空间尺寸不一致')
        #* ================== 3. 复用单帧评估统计，新增时间维汇总 ==================
        # 父类 occupancy_results 每次统计一个时间步、返回每个样本的混淆矩阵和 token。
        # 此处只将各步矩阵按时间拼入同一样本；跨卡收集/token去重仍由既有测试入口负责。
        # Dataset 复用已有逐步 IoU/mIoU 计算；GT 到这里才用于推理结果统计，不参与预测。
        results = None
        for step in range(self.future_steps + 1):
            items = self.occupancy_results(pred[:, step], gt_occ[:, step], img_metas)
            if results is None:
                results = items
            else:
                for result, item in zip(results, items):
                    result['hist_for_iou_per_frame'].extend(item['hist_for_iou_per_frame'])
        return results
