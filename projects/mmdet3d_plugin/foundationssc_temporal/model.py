"""时序 SSC 的独立模型入口；当前阶段只验证数据接口。"""
from mmdet.models import DETECTORS
from ..foundationssc.image_model import FoundationSSCImageModel


@DETECTORS.register_module()
class FoundationSSCTemporalModel(FoundationSSCImageModel):
    #! 复用单帧模型参数结构；未来在此加入历史特征提取与融合，不复制原骨干。
    # 当前禁止静默忽略历史数据后按单帧训练，避免将接口验证误认为已实现时序融合。
    def forward(self, return_loss=False, history_img_inputs=None,
                history_img_metas=None, temporal_metas=None, **kwargs):
        raise NotImplementedError('当前仅完成时序数据接口，请运行 scripts/test_foundationssc_temporal.py；尚未实现融合前向')
