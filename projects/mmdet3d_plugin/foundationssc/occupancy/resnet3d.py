"""移植 FoundationSSC resnet3d.py 的 3D 部分；不注册同名全局 backbone。"""
import torch.utils.checkpoint as checkpoint
from torch import nn
from mmcv.cnn import ConvModule


class BasicBlock3D(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        stride=1,
        kernel_size=3,
        padding=1,
        downsample=None,
    ):
        super(BasicBlock3D, self).__init__()
        self.conv1 = ConvModule(
            channels_in,
            channels_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            conv_cfg=dict(type="Conv3d"),
            norm_cfg=dict(
                type="BN3d",
            ),
            act_cfg=dict(type="ReLU", inplace=True),
        )
        self.conv2 = ConvModule(
            channels_out,
            channels_out,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
            conv_cfg=dict(type="Conv3d"),
            norm_cfg=dict(
                type="BN3d",
            ),
            act_cfg=None,
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)


class CustomResNet3D(nn.Module):

    def __init__(
        self,
        numC_input,
        num_layer=[2, 2, 2],
        num_channels=None,
        stride=[2, 2, 2],
        kernel_size=3,
        padding=1,
        backbone_output_ids=None,
        with_cp=False,
    ):
        super(CustomResNet3D, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = (
            [numC_input * 2 ** (i + 1) for i in range(len(num_layer))]
            if num_channels is None
            else num_channels
        )
        self.backbone_output_ids = (
            range(len(num_layer))
            if backbone_output_ids is None
            else backbone_output_ids
        )
        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [
                BasicBlock3D(
                    curr_numC,
                    num_channels[i],
                    stride=stride[i],
                    kernel_size=kernel_size,
                    padding=padding,
                    downsample=ConvModule(
                        curr_numC,
                        num_channels[i],
                        kernel_size=kernel_size,
                        stride=stride[i],
                        padding=padding,
                        bias=False,
                        conv_cfg=dict(type="Conv3d"),
                        norm_cfg=dict(
                            type="BN3d",
                        ),
                        act_cfg=None,
                    ),
                )
            ]
            curr_numC = num_channels[i]
            layer.extend(
                [
                    BasicBlock3D(
                        curr_numC, curr_numC, kernel_size=kernel_size, padding=padding
                    )
                    for _ in range(num_layer[i] - 1)
                ]
            )
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats
