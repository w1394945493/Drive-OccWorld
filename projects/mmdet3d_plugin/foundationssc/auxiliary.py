"""原 FoundationSSC BCE 深度监督与二维语义辅助头；不依赖外部仓库。"""
import torch
from torch import nn
from torch.nn import functional as F


def depth_loss(prob, target, bounds):
    #* （FoundationSSC 辅助深度&语义损失) 每个下采样块选最近有效深度，
    # 沿用原 DSGP 的半 bin 偏移、D+1 one-hot 去第0列和逐有效像素 BCE 求和。
    b, d, h, w = prob.shape
    if target.ndim != 4 or target.shape[:2] != (b, 1):
        raise ValueError('左图 gt_depths 应为 [B,1,H,W]')
    H, W = target.shape[-2:]
    scale = H // h
    if scale < 1 or H != h * scale or W != w * scale:
        raise ValueError('深度标签与概率图必须等比例整数下采样')
    if d != len(torch.arange(*bounds)):
        raise ValueError('深度概率通道数与 depth_bound 不一致')
    with torch.autocast(device_type=prob.device.type, enabled=False):
        prob = prob.float()
        target = target.to(prob).detach()
        if not torch.isfinite(target).all() or (target < 0).any():
            raise ValueError('深度标签应为非负有限值（米），0 表示无效')
        blocks = target.reshape(b, h, scale, w, scale).permute(0, 1, 3, 2, 4).reshape(-1, scale * scale)
        nearest = torch.where(blocks > 0, blocks, torch.full_like(blocks, 1e5)).min(-1).values
        bins = (nearest - (bounds[0] - bounds[2] / 2)) / bounds[2]
        bins = torch.where((bins >= 0) & (bins < d + 1), bins, torch.zeros_like(bins)).long()
        labels = F.one_hot(bins, d + 1)[:, 1:].float()
        valid = labels.max(-1).values > 0
        preds = prob.permute(0, 2, 3, 1).reshape(-1, d)
        if not valid.any():
            return prob.sum() * 0  # 全无效也保留计算图，避免 NaN/空均值。
        return F.binary_cross_entropy(preds[valid], labels[valid], reduction='sum') / valid.sum()


class PluginSegmentationHead(nn.Module):
    def __init__(self, in_channels=128, num_classes=20):
        super().__init__()
        #* （FoundationSSC 辅助深度&语义损失) 原三层 deconv+BN+ReLU，8倍上采样。
        blocks = []
        for channels in (128, 64, 32):
            blocks.append(nn.Sequential(nn.ConvTranspose2d(in_channels, channels, 2, 2, bias=False),
                                        nn.BatchNorm2d(channels, eps=1e-3, momentum=.01), nn.ReLU(inplace=True)))
            in_channels = channels
        self.deconv_blocks = nn.ModuleList(blocks)
        self.pred = nn.Conv2d(32, num_classes, 1)

    def forward(self, context):
        for block in self.deconv_blocks:
            context = block(context)
        return self.pred(context)

    def loss(self, logits, target, depth, ignore_index=255):
        #* （FoundationSSC 辅助深度&语义损失) 原实现逐样本平均 CE，再对 batch 平均；
        # 只监督有效深度像素，保留点级 unlabeled=0 的原映射，排除 ignore=255。
        if target.shape != depth.shape or target.shape != (logits.shape[0], 1, *logits.shape[-2:]):
            raise ValueError('左图语义/深度标签与二维预测形状不一致')
        if target.is_floating_point():
            raise ValueError('语义标签必须为整数')
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits = logits.float()
            target = target[:, 0].to(device=logits.device, dtype=torch.long)
            depth = depth[:, 0].to(logits)
            terms = []
            for prediction, label, z in zip(logits, target, depth):
                valid = torch.isfinite(z) & (z > 0) & (label != ignore_index)
                if not valid.any():
                    terms.append(prediction.sum() * 0)
                else:
                    terms.append(F.cross_entropy(prediction.permute(1, 2, 0)[valid], label[valid]))
            return torch.stack(terms).mean()
