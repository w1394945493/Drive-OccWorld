"""时序 SSC 扩展入口：当前先完整复用单帧前向，尚不融合历史。"""
from mmdet.models import DETECTORS
from ..foundationssc.image_model import FoundationSSCImageModel


@DETECTORS.register_module()
class FoundationSSCTemporalModel(FoundationSSCImageModel):
    #! 继承原初始化、冻结骨干策略及参数命名，可直接加载单帧 FoundationSSC 权重。
    #* forward_train/forward_test、占据及辅助损失、混淆矩阵、train_step 均复用父类。
    # 两个前向入口都会调用下方 _forward_occupancy，训练与推理共用同一条特征路径。

    def _forward_occupancy(self, img_inputs, img_metas):
        """当前双目 → 图像特征 → 体素特征 → 原占据解码器；与单帧前向等价。"""
        #* ================== 一、复用当前帧图像特征提取 ==================
        # 冻结 FoundationStereo + 可训练图像金字塔；不提取历史图像特征。
        output = self.extract_image_features(img_inputs, img_metas)

        #* ================== 二、复用当前帧体素特征构建 ==================
        # 原 LSS、候选细化与双分支融合，默认 voxel_feats=[B,128,128,128,16]。
        output.update(self.voxel_encoder(output, img_inputs, img_metas))

        # todo 待实现时序融合：在此将历史体素对齐到当前坐标系，并融合到 output['voxel_feats']。




        # todo 后续需从前向入口传入历史图像/标定和 temporal_metas，使用 history_valid 区分补帧。
        # 当前不读取历史输入，不改变 voxel_feats；context/depth_prob 保留供当前帧辅助监督。

        #* ================== 三、复用原三维占据解码器 ==================
        encoded = self.occ_encoder_neck(self.occ_encoder_backbone(output['voxel_feats']))
        output.update(self.pts_bbox_head([encoded[0]]))  # 与原单帧一致，仅取 neck 最高分辨率输出。
        # output_voxels 保留梯度用于损失；离散预测仅用于评估/可视化。
        output['pred'] = output['output_voxels'].detach().argmax(dim=1)
        return output

    #* ================== 四、复用训练损失与推理评估入口 ==================
    def forward(self, return_loss=False, history_img_inputs=None,
                history_img_metas=None, temporal_metas=None, **kwargs):
        #! 接收时序 pipeline 的额外字段，但本阶段明确不使用历史，仅验证完整当前帧流程。
        # 不缓存历史张量到 self，也不额外执行历史骨干，避免引入未实现的融合行为和计算开销。
        # 父类负责 img_metas 的 list→dict 整理，并分派到继承的 forward_train/forward_test：
        # 训练：当前占据损失 + 按配置启用的辅助深度/二维语义损失。
        # 推理：return_outputs=True 返回完整输出，否则有 GT 时返回逐样本混淆矩阵。
        return super().forward(return_loss=return_loss, **kwargs)
