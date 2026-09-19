"""FoundationSSC 的 33 维标定条件，采用本项目 PKL 相机外参。"""
import torch

def get_mlp_input(rot, tran, intrin, post_rot, post_tran, bda=None):
    B, N, _, _ = rot.shape

    if bda is None:
        bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)

    bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)

    if intrin.shape[-1] == 4:
        # for KITTI, the intrin matrix is 3x4
        mlp_input = torch.stack(
            [
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                intrin[:, :, 0, 3],
                intrin[:, :, 1, 3],
                intrin[:, :, 2, 3],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ],
            dim=-1,
        )

        if bda.shape[-1] == 4:
            mlp_input = torch.cat((mlp_input, bda[:, :, :3, -1]), dim=2)
    else:
        mlp_input = torch.stack(
            [
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda[:, :, 0, 0],
                bda[:, :, 0, 1],
                bda[:, :, 1, 0],
                bda[:, :, 1, 1],
                bda[:, :, 2, 2],
            ],
            dim=-1,
        )

    sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(
        B, N, -1
    )
    mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)

    return mlp_input
