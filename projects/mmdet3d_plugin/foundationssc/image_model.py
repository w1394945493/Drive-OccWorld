"""当前帧 SSC：冻结 FoundationStereo → 图像/体素特征 → 占据预测与监督。"""
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data._utils.collate import default_collate
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        # [B,C,H,W] → [B,H,W,C]，在通道维做 LayerNorm，再恢复图像布局。
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

1
class FoundationImagePyramid(nn.Module):
    """对应原 SimpleFPN(layers=[4]) + SECONDFPN，不重复注册同名 neck。"""
    def __init__(self, backbone_channels=1024, out_channels=160):
        super().__init__()
        c = backbone_channels
        #* SimpleFPN 的实际实现没有额外 lateral conv：直接输出这四个尺度。
        self.fpn1 = nn.Sequential(nn.ConvTranspose2d(c, c // 2, 2, 2),
                                  LayerNorm2d(c // 2), nn.GELU(),
                                  nn.ConvTranspose2d(c // 2, c // 4, 2, 2))
        self.fpn2 = nn.ConvTranspose2d(c, c // 2, 2, 2)
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(2, 2)
        #* SECONDFPN: stride=[0.5,1,2,4]，统一空间尺寸后拼成 4*160=640 通道。
        self.deblocks = nn.ModuleList()
        for channels, stride in zip((c // 4, c // 2, c, c), (.5, 1, 2, 4)):
            conv = (nn.Conv2d(channels, out_channels, 2, 2, bias=False) if stride == .5
                    else nn.ConvTranspose2d(channels, out_channels, int(stride), int(stride), bias=False))
            self.deblocks.append(nn.Sequential(conv, nn.BatchNorm2d(out_channels, eps=1e-3, momentum=.01), nn.ReLU(inplace=True)))

    def forward(self, features):
        #* 图像金字塔：仅用第 4 组 DINO 特征生成四尺度，再对齐分辨率并拼接。
        # features 含四个 (feature_map, cls_token)；本模块不使用 cls_token。
        if len(features) != 4:
            raise ValueError('需要四层 DINOv2 特征，以第 4 层构建金字塔')
        x = features[3][0].float()  # FoundationStereo 可输出 FP16；可训练 FPN 使用 FP32。
        pyramid = [self.fpn1(x), self.fpn2(x), self.fpn3(x), self.fpn4(x)]
        outputs = [block(level) for block, level in zip(self.deblocks, pyramid)]
        if len({value.shape[-2:] for value in outputs}) != 1:
            raise ValueError('FPN 尺度不匹配，请检查 DINO 特征空间尺寸')
        return torch.cat(outputs, dim=1), pyramid  # fused 用于体素构建；pyramid 保留各尺度中间特征。


@DETECTORS.register_module()
class FoundationSSCImageModel(BaseModule):
    #* ================== 模型整体流程 ==================
    #* 一、双目图像 → 冻结 FoundationStereo/DINO → 可训练图像金字塔。
    #* 二、图像特征/深度/标定 → LSS 粗体素 + proposal 细化 → 融合 voxel 特征。
    #* 三、voxel 特征 → 3D ResNet/FPN → 当前帧 occupancy 分类与上采样。
    #* 四、训练监督与评估：占据/辅助损失，或混淆矩阵汇总；无未来预测模块。
    #* 测试入口据此选择逐样本 SSC 收集，不进入 forecasting 的按卡预求和分支。
    occupancy_eval_per_sample = True
    def __init__(self, stereo_checkpoint, stereo_config,
                 gru_iters=12, backbone_channels=1024, out_channels=160,
                 strict_load=True, voxel_encoder=None, occ_encoder_backbone=None,
                 occ_encoder_neck=None, pts_bbox_head=None, train_cfg=None, test_cfg=None,
                 use_depth_loss=False, use_semantic_loss=False,
                 loss_depth_weight=1., loss_seg_weight=1.):
        super().__init__()
        from omegaconf import OmegaConf
        #* ================== 初始化：加载并冻结立体骨干 ==================
        #* 所有骨干源码均在本包中；仅权重与 YAML 是外部数据文件。
        from .stereo.core.foundation_stereo import FoundationStereo
        for path in (stereo_checkpoint, stereo_config):
            if not Path(path).is_file():
                raise FileNotFoundError(f'缺少 FoundationStereo 配置/权重：{path}')
        args = OmegaConf.load(stereo_config)
        if 'vit_size' not in args:
            args.vit_size = 'vitl'
        self.gru_iters = gru_iters
        self.img_backbone = FoundationStereo(args)
        checkpoint = torch.load(stereo_checkpoint, map_location='cpu')
        state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        incompatible = self.img_backbone.load_state_dict(state, strict=strict_load)
        if incompatible.missing_keys:
            raise RuntimeError('冻结骨干不能缺失权重（含 EdgeNeXt/DINO）：' + str(incompatible.missing_keys))
        self.checkpoint_report = dict(missing=list(incompatible.missing_keys), unexpected=list(incompatible.unexpected_keys))
        self.img_backbone.requires_grad_(False)
        self.img_backbone.eval()
        #* ================== 初始化：可训练图像金字塔与体素前端 ==================
        self.image_pyramid = FoundationImagePyramid(backbone_channels, out_channels)
        self.voxel_encoder = None
        if voxel_encoder is not None:
            from .voxel.encoder import FoundationVoxelEncoder
            options = dict(voxel_encoder)
            options.setdefault('input_channels', 4 * out_channels)
            options.setdefault('disparity_channels', args.max_disp // 4)
            if options['disparity_channels'] != args.max_disp // 4:
                raise ValueError('voxel disparity_channels 必须等于 stereo YAML max_disp//4')
            self.voxel_encoder = FoundationVoxelEncoder(**options)

        #* ================== 初始化：三维编码器与占据分类头 ==================
        # 与原 FoundationSSC 一致：融合体素 → 3D ResNet → 3D FPN → OccHead。
        # 本地直接实例化，避免和 Drive-OccWorld 现有同名注册模块冲突。
        from .occupancy import CustomResNet3D, GeneralizedLSSFPN, OccHead
        modules = ((occ_encoder_backbone, CustomResNet3D, 'occ_encoder_backbone'),
                   (occ_encoder_neck, GeneralizedLSSFPN, 'occ_encoder_neck'),
                   (pts_bbox_head, OccHead, 'pts_bbox_head'))
        if self.voxel_encoder is None or any(config is None for config, _, _ in modules):
            raise ValueError('完整占据模型需配置 voxel_encoder、occ_encoder_backbone、occ_encoder_neck 和 pts_bbox_head')
        for config, cls, name in modules:
            options = dict(config)
            if options.pop('type', cls.__name__) != cls.__name__:
                raise ValueError(f'{name} 应使用本地 {cls.__name__}')
            setattr(self, name, cls(**options))
        #* ================== 初始化：可选辅助监督 ==================
        #* （FoundationSSC 辅助深度&语义损失) 构造参数默认关闭；是否启用以实际配置文件为准。
        # 深度损失直接监督 depth_prob；语义损失开启时才创建 plugin_head，不参与推理占据预测。
        self.use_depth_loss = use_depth_loss
        self.use_semantic_loss = use_semantic_loss
        self.loss_depth_weight = float(loss_depth_weight)
        self.loss_seg_weight = float(loss_seg_weight)
        if self.use_semantic_loss:
            from .auxiliary import PluginSegmentationHead
            self.plugin_head = PluginSegmentationHead(voxel_encoder.get('channels', 128), self.pts_bbox_head.out_channel)

    def train(self, mode=True):
        super().train(mode)
        #* 无论外层如何切换，冻结骨干始终 eval；FPN 正常切换 train/eval。
        self.img_backbone.eval()
        return self

    def init_weights(self):
        #* torch 模块已在构造时初始化，冻结骨干也已严格加载预训练权重。
        # train.py 会再次调用 init_weights；不能递归重置已加载的冻结骨干。
        self._is_init = True

    def train_step(self, data, optimizer=None):
        """MMCV Runner 接口：这里只算 loss，反传/裁剪/更新交给 OptimizerHook。"""
        #* 四、训练接口：调用下面的一～三部分及损失函数；参数更新由外部 Runner 完成。
        losses = self(return_loss=True, **data)
        values = {}
        for name, value in losses.items():
            if torch.is_tensor(value):
                values[name] = value.mean()
            elif isinstance(value, list) and all(torch.is_tensor(x) for x in value):
                values[name] = sum(x.mean() for x in value)
            else:
                raise TypeError(f'{name} 必须为 Tensor 或 Tensor 列表')
        loss = sum(value for name, value in values.items() if 'loss' in name)
        values['loss'] = loss
        log_vars = {}
        for name, value in values.items():
            #* 仅日志副本跨卡平均，不替换本卡带梯度的 loss；DDP 负责梯度同步。
            logged = value.detach().clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(logged)
                logged /= dist.get_world_size()
            log_vars[name] = logged.item()
        return dict(loss=loss, log_vars=log_vars, num_samples=data['gt_occ'].shape[0])

    #* ================== 一、图像特征提取 ==================
    #* 当前帧左右 RGB → 立体匹配结果 + DINO 特征 → 左图多尺度融合特征。
    def extract_image_features(self, img_inputs, img_metas):
        raw = img_metas['raw_img']
        # raw_img 是 pipeline 已 resize/crop 后、尚未标准化的左右 RGB 图，不是直接读取的原始尺寸图。
        if len(raw) != 2:
            raise ValueError('当前模型要求每帧一对左右图像')
        device = next(self.image_pyramid.parameters()).device
        left, right = [x.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).contiguous() for x in raw] # (1 3 384 1280)
        if left.shape != right.shape or left.shape[0] != img_inputs[0].shape[0]:
            raise ValueError('双目形状或 batch 不一致')
        #* 原图数值保持 0..255；FoundationStereo 内部 normalize_image 负责归一化。
        with torch.no_grad():
            #* 冻结骨干，无梯度；disparity 包含视差概率及视差图，不是 GT。
            # 输入 left/right=[B,3,H,W]，当前 H=384、W=1280。返回两部分：
            # disparity 是 [prob, disp_up] 列表，并非单个 Tensor：
            #   prob: [B,D_disp,H/4,W/4]，左图各像素的离散视差概率，通道维和为1；
            #         D_disp=stereo YAML 的 max_disp//4，max_disp=416 时为 [B,104,96,320]。
            #         它是初始代价体的分类概率，不是深度概率，后续由 depth_net 转换。
            #   disp_up: [B,1,H,W]=[B,1,384,1280]，迭代细化后的左图视差（输入图像像素单位）；
            #            后续结合焦距和基线按 Z=f*b/disparity 转成预测深度。
            # features 是四个 DINO 中间层的列表，每项为 (feature_map, cls_token)：
            #   feature_map: [2B,C,H/16,W/16]=[2B,C,24,80]；四层同尺寸，并非四尺度金字塔。
            #                DINO patch token 在 extractor 内已 reshape 并插值到此尺寸。
            #   cls_token: [2B,C]，对应各图像的全局 token，不是语义类别预测。
            #   C=384（Small）或1024（Large）；2B 的前B项为左图，后B项为右图。
            disparity, features = self.img_backbone(left, right, test_mode=True, iters=self.gru_iters)
        batch = left.shape[0]
        #* 后续体素构建仅使用左图特征，右图用于前面的立体匹配。
        # 仅截取左图：四组 ([B,C,24,80], [B,C])；image_pyramid 实际使用第4组 feature_map。
        left_features = [(feature[:batch], cls_token[:batch]) for feature, cls_token in features]
        # image_pyramid 用第4层左图 DINO 特征 [B,C,24,80] 构建四尺度特征：
        #   pyramid：四个 Tensor 的列表，尚未统一分辨率/通道；输入图像为384×1280时：
        #     Small: [B,96,96,320]、[B,192,48,160]、[B,384,24,80]、[B,384,12,40]；
        #     Large: [B,256,96,320]、[B,512,48,160]、[B,1024,24,80]、[B,1024,12,40]。
        #   fused：四路分别经卷积/转置卷积变为 [B,160,48,160]，再沿通道维拼接，
        #          得到 [B,640,48,160]（Small/Large 相同），并非四路求和。
        # fused 用于后续 voxel 构建；pyramid 保留作中间特征检查，当前不直接送入 voxel_encoder。
        fused, pyramid = self.image_pyramid(left_features)
        #* 图像金字塔训练时有梯度；默认 img_feats=[B,1,640,48,160]，1 表示左相机。
        return dict(img_feats=fused.unsqueeze(1), disparity=disparity,
                    dino_features=left_features, pyramid=pyramid)

    def _forward_occupancy(self, img_inputs, img_metas):
        """不读取未来信息或 GT：双目 → 图像特征 → 体素 → 当前帧语义 logits。"""
        #* ================== 一、图像特征提取 ==================
        output = self.extract_image_features(img_inputs, img_metas)

        #* ================== 二、voxel 特征构建 ==================
        # voxel/encoder.py：生成 context/深度概率，分别构建 LSS 粗体素和
        # proposal 引导的细化体素，再融合为 voxel_feats=[B,128,128,128,16]。
        #* 上述为当前配置尺寸；该部分可训练，使用预测深度而非辅助监督 GT 深度。
        output.update(self.voxel_encoder(output, img_inputs, img_metas))

        #* ================== 三、三维占据预测 ==================
        #* 融合 voxel → 多尺度 3D ResNet → 3D FPN → OccHead。
        encoded = self.occ_encoder_neck(self.occ_encoder_backbone(output['voxel_feats']))
        # 原 neck 返回两个融合尺度，原 detector 仅将最高分辨率尺度送给 OccHead。
        output.update(self.pts_bbox_head([encoded[0]]))
        # 默认 logits=[B,20,256,256,32]，用于训练损失；argmax 得到 pred=[B,256,256,32]。
        #* 仅离散 pred 分支 detach，用于评估/可视化；output_voxels 保留计算图供损失反传。
        output['pred'] = output['output_voxels'].detach().argmax(dim=1)
        return output

    #* ================== 四、训练监督与评估接口 ==================
    #* 以下 GT 只用于损失/指标；一～三部分的预测前向不依赖 GT。
    #* 四（1）、评估：逐样本混淆矩阵 → Dataset 汇总 IoU/mIoU。
    def occupancy_results(self, pred, gt_occ, img_metas):
        """每个样本返回 hist[gt, pred]；GT 仅用于统计，不参与预测。"""
        gt = gt_occ.to(device=pred.device)
        if gt.shape != pred.shape or gt.ndim != 4 or gt.is_floating_point():
            raise ValueError('pred/gt_occ 应为同形状 [B,X,Y,Z]，GT 必须是整数标签')
        classes = self.pts_bbox_head.out_channel
        tokens = img_metas['token']
        if isinstance(tokens, str):
            tokens = [tokens]
        if len(tokens) != pred.shape[0]:
            raise ValueError('token 数量与 batch 不一致')
        results = []
        for prediction, target, token in zip(pred, gt, tokens):
            # 排除 ignore 标签（默认 255）；empty 类保留在矩阵内，供 Dataset 计算空/非空 IoU。
            valid = target != self.pts_bbox_head.ignore_index
            target, prediction = target[valid].long(), prediction[valid].long()
            if ((target < 0) | (target >= classes) | (prediction < 0) | (prediction >= classes)).any():
                raise ValueError('评估标签超出类别范围')
            hist = torch.bincount(target * classes + prediction, minlength=classes ** 2).reshape(classes, classes)  # [20,20]，行=GT、列=预测。
            #* 只传小型 CPU 混淆矩阵；token 用于剔除分布式 sampler 补齐的重复样本。
            results.append(dict(sample_token=token, hist_for_iou_per_frame=[hist.cpu().numpy()]))
        return results

    def forward_test(self, img_inputs, img_metas, gt_occ=None, return_outputs=False, **kwargs):
        #* 推理复用一～三部分，不执行辅助二维分类头或训练损失。
        output = self._forward_occupancy(img_inputs, img_metas)
        #* 调试/可视化显式保留原始输出；正式评估默认返回 list[dict]。
        if return_outputs or gt_occ is None:
            return output
        return self.occupancy_results(output['pred'], gt_occ, img_metas)

    def compute_losses(self, output, gt_occ, img_metas, gt_semantics=None):
        #* 四（2）、训练监督：三项三维占据损失 + 可选深度 BCE/二维语义 CE。
        #* 深度监督约束第二部分 depth_prob；二维语义头约束第二部分左图 context。
        losses = self.pts_bbox_head.loss(output['output_voxels'], gt_occ)
        #! 拆出辅助损失供未来预测复用；单帧训练仍计算原有占据及辅助损失。
        losses.update(self.compute_auxiliary_losses(output, img_metas, gt_semantics))
        return losses

    def compute_auxiliary_losses(self, output, img_metas, gt_semantics=None):
        #! 仅监督当前输入帧的 depth_prob/context，不需要解码当前 occupancy。
        losses = {}
        #* （FoundationSSC 辅助深度&语义损失) 仅训练/显式损失检查访问 GT，推理不需要 LiDAR。
        if self.use_depth_loss or self.use_semantic_loss:
            if 'gt_depths' not in img_metas or 'projection_camera_indices' not in img_metas:
                raise ValueError('辅助监督已开启，请配置点云加载及 ProjectFoundationSSCLidar')
            cameras = img_metas['projection_camera_indices']
            left = cameras == 0
            if not (left.sum(1) == 1).all():
                raise ValueError('辅助监督每个样本必须且仅包含一个左相机标签')
            rows = torch.arange(left.shape[0], device=left.device)
            slots = left.long().argmax(1)
            # 根据相机编号找到左图标签所在槽位，不假设其一定是第 0 个；depth=[B,1,H,W] 为稀疏 GT 深度。
            depth = img_metas['gt_depths'][rows, slots][:, None]
            if self.use_depth_loss:
                from .auxiliary import depth_loss
                # 将稀疏 GT 深度转换为低分辨率深度分类目标，监督预测概率，不将 GT 喂入预测前向。
                losses['loss_depth'] = self.loss_depth_weight * depth_loss(output['depth_prob'], depth, self.voxel_encoder.depth_bound)
            if self.use_semantic_loss:
                if gt_semantics is None:
                    raise ValueError('use_semantic_loss=True 时必须提供 gt_semantics')
                target = gt_semantics[rows.to(gt_semantics.device), slots.to(gt_semantics.device)][:, None]
                logits = self.plugin_head(output['context'][:, 0].float())
                # 左图二维语义 logits=[B,20,H,W]，仅在有效深度/语义标签位置计算辅助 CE。
                losses['loss_seg_ce'] = self.loss_seg_weight * self.plugin_head.loss(logits, target, depth)
        return losses

    def forward_train(self, img_inputs, img_metas, gt_occ, return_outputs=False, gt_semantics=None, **kwargs):
        #* 四（3）、训练总入口：先完成一～三部分预测，再调用 compute_losses。
        output = self._forward_occupancy(img_inputs, img_metas)
        losses = self.compute_losses(output, gt_occ, img_metas, gt_semantics)
        #* 默认返回 loss 字典；调试脚本可取同一次前向的输出，避免重复运行骨干。
        if return_outputs:
            output['losses'] = losses
            return output
        return losses

    def forward(self, return_loss=False, **kwargs):
        #* 统一入口：return_loss=True 返回训练损失，False 返回评估统计或显式请求的原始预测。
        #* MMCV scatter 后为 list[每样本 meta]；独立脚本则已 default_collate 成 dict。
        if isinstance(kwargs.get('img_metas'), (list, tuple)):
            kwargs['img_metas'] = default_collate(kwargs['img_metas'])

        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
