# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


#* 仅保留前向所需函数，不导入可视化工具、不重置全局 logging。
import numpy as np

def freeze_model(model):
    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.buffers():
        p.requires_grad = False
    return model

def get_resize_keep_aspect_ratio(H, W, divider=16, max_H=1232, max_W=1232):
    assert max_H % divider == 0
    assert max_W % divider == 0

    def round_by_divider(x):
        return int(np.ceil(x / divider) * divider)

    H_resize = round_by_divider(H)  #!NOTE KITTI width=1242
    W_resize = round_by_divider(W)
    if H_resize > max_H or W_resize > max_W:
        if H_resize > W_resize:
            W_resize = round_by_divider(W_resize * max_H / H_resize)
            H_resize = max_H
        else:
            H_resize = round_by_divider(H_resize * max_W / W_resize)
            W_resize = max_W
    return int(H_resize), int(W_resize)
