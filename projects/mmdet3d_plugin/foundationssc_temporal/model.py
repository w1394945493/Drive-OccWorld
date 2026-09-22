"""时序 SSC 扩展入口：当前先完整复用单帧前向，尚不融合历史。"""
import torch
from torch.utils.data._utils.collate import default_collate
from mmdet.models import DETECTORS
from ..foundationssc.image_model import FoundationSSCImageModel


@DETECTORS.register_module()
class FoundationSSCTemporalModel(FoundationSSCImageModel):
    #! 继承原初始化、冻结骨干策略及参数命名，可直接加载单帧 FoundationSSC 权重。
    #* 占据及辅助损失、混淆矩阵、train_step 均复用父类。
    # 两个前向入口都会调用下方 _forward_occupancy，训练与推理共用同一条特征路径。

    def extract_history_image_features(self, history_img_inputs, history_img_metas):
        """逐时刻复用双目骨干和图像金字塔，返回长度 T 的特征字典列表。"""
        b, steps = history_img_inputs[0].shape[:2]  # 图像：[B,T,2,3,H,W]；其余输入同样含 B,T。
        #! 正式 scatter：list[B][T] 的逐帧 dict；default_collate：list[T] 的批量 dict。
        if len(history_img_metas) and isinstance(history_img_metas[0], (list, tuple)):
            if len(history_img_metas) != b or any(len(row) != steps for row in history_img_metas):
                raise ValueError('历史 meta 应为 [B][T]，与历史图像维度一致')
            metas = [default_collate([row[t] for row in history_img_metas]) for t in range(steps)]
        else:
            metas = history_img_metas
        if len(metas) != steps:
            raise ValueError('历史 meta 的时刻数与历史图像不一致')

        #! 历史分支无梯度且不更新 BN：只临时切换共享骨干/FPN，随后精确恢复各子模块状态。
        # no_grad 本身不能阻止 BN 更新；也不能永久冻结 FPN，否则当前帧无法正常训练。
        modules = list(self.img_backbone.modules()) + list(self.image_pyramid.modules())
        modes = [module.training for module in modules]
        outputs = []
        self.img_backbone.eval()
        self.image_pyramid.eval()
        with torch.no_grad():
            for t in range(steps):
                inputs = tuple(value[:, t] for value in history_img_inputs)
                #! 历史左右图 → 视差概率/视差图 + 左图 DINO 特征 → 融合金字塔；全部共享原权重。
                # 补帧也正常提取；后续融合必须依据 temporal_metas.history_valid 区分真实历史。
                outputs.append(self.extract_image_features(inputs, metas[t]))
        for module, mode in zip(modules, modes):
            module.training = mode
        # 每项与当前帧一致：img_feats=[B,1,640,H/8,W/8]、disparity、dino_features、pyramid。
        # 保留逐帧列表，不沿时间重复 stack 大特征；此处尚未生成历史 voxel。
        return outputs

    def _forward_occupancy(self, img_inputs, img_metas, history_features=None):
        """当前双目 → 图像特征 → 体素特征 → 原占据解码器；与单帧前向等价。"""
        #* ================== 一、复用当前帧图像特征提取 ==================
        # 冻结 FoundationStereo + 可训练图像金字塔；历史图像已在外层 forward 单独提取。
        output = self.extract_image_features(img_inputs, img_metas)

        #* ================== 二、复用当前帧体素特征构建 ==================
        # 原 LSS、候选细化与双分支融合，默认 voxel_feats=[B,128,128,128,16]。
        output.update(self.voxel_encoder(output, img_inputs, img_metas))

        # 待实现时序融合：在此将历史体素对齐到当前坐标系，并融合到 output['voxel_feats']。




        # history_features 已传入此处待用；尚未构建历史体素或执行时序融合。
        # 当前历史特征不参与预测，不改变 voxel_feats；context/depth_prob 保留供当前帧辅助监督。

        #* ================== 三、复用原三维占据解码器 ==================
        encoded = self.occ_encoder_neck(self.occ_encoder_backbone(output['voxel_feats']))
        output.update(self.pts_bbox_head([encoded[0]]))  # 与原单帧一致，仅取 neck 最高分辨率输出。
        # output_voxels 保留梯度用于损失；离散预测仅用于评估/可视化。
        output['pred'] = output['output_voxels'].detach().argmax(dim=1)
        return output

    #* ================== 四、复用训练损失与推理评估入口 ==================
    def forward_train(self, img_inputs, img_metas, gt_occ, return_outputs=False,
                      gt_semantics=None, history_features=None, **kwargs):
        output = self._forward_occupancy(img_inputs, img_metas, history_features=history_features)
        losses = self.compute_losses(output, gt_occ, img_metas, gt_semantics)
        if return_outputs:
            output['losses'] = losses
            output['history_features'] = history_features
            return output
        return losses

    def forward_test(self, img_inputs, img_metas, gt_occ=None, return_outputs=False,
                     history_features=None, **kwargs):
        output = self._forward_occupancy(img_inputs, img_metas, history_features=history_features)
        if return_outputs or gt_occ is None:
            output['history_features'] = history_features
            return output
        return self.occupancy_results(output['pred'], gt_occ, img_metas)

    def forward(self, return_loss=False, history_img_inputs=None,
                history_img_metas=None, temporal_metas=None, **kwargs):
        #! 先提取历史图像特征；不缓存到 self，不改变原权重结构，暂不参与当前占据预测。
        if (history_img_inputs is None) != (history_img_metas is None):
            raise ValueError('历史图像和历史 meta 必须同时提供')
        history_features = self.extract_history_image_features(history_img_inputs, history_img_metas) if history_img_inputs is not None else []

        # 父类负责 img_metas 的 list→dict 整理，并分派到本类的 forward_train/forward_test：
        # 训练：当前占据损失 + 按配置启用的辅助深度/二维语义损失。
        # 推理：return_outputs=True 返回完整输出，否则有 GT 时返回逐样本混淆矩阵。
        return super().forward(return_loss=return_loss, history_features=history_features, **kwargs)
