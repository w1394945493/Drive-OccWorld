"""联合训练原可训练感知模块与未来预测器；FoundationStereo 仍冻结。"""
_base_ = './foundationssc_forecast.py'

freeze_frontend = False  #* 修改1：解冻图像金字塔与 voxel_encoder；FoundationStereo 仍冻结。
freeze_decoder = False  #* 修改2：解冻 3D ResNet、FPN 和占据头，与 dynamics 联合优化。
model = dict(freeze_frontend=freeze_frontend, freeze_decoder=freeze_decoder)  #* 显式覆盖 model，顶层变量不会自动更新父配置字典。
work_dir = 'out/foundationssc_forecast_no_freeze'  #* 修改3：独立保存日志及 checkpoint。

# 其余继承父配置；仅监督未来四步，无当前帧或辅助损失。解冻增加显存，建议先小规模验证。
