"""FoundationSSC Large：继承 Small 配置，仅替换骨干及其配套权重/通道。"""
_base_ = ['./foundationssc_semantic_kitti.py']

#* 参考 FoundationSSC/configs/customs/FoundationSSC-SemanticKITTI.py。
# Large 使用 23-51-11 的完整权重及 YAML，DINOv2 为 ViT-L/14、1024 通道。
# FoundationImagePyramid 随 backbone_channels 自动适配为 [256,512,1024,1024]，
# 融合输出仍为 4*160=640 通道；体素前端、占据头、两项辅助损失和训练参数不变。
model = dict(
    stereo_checkpoint='/c20250502/wangyushen/Weights/foundationssc/23-51-11/model_best_bp2.pth',
    stereo_config='/c20250502/wangyushen/Weights/foundationssc/23-51-11/cfg.yaml',
    backbone_channels=1024,
)

#* 仅隔离输出目录，避免自动续训时误加载 Small 的模型/优化器状态。
work_dir = 'out/foundationssc_large_semantic_kitti'
