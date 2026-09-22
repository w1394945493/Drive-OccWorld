#!/usr/bin/env python3
"""仅验证历史双目/标定/位姿与当前 SSC 数据，不构建模型、不加载权重、不使用 CUDA。"""
import argparse
import copy
import sys
from pathlib import Path
import torch
from torch.utils.data._utils.collate import default_collate


def check_temporal(batch):
    from test_foundationssc import check_batch
    check_batch(batch)  # 当前双目、投影、归一化、GT，以及训练样本的辅助监督。
    inputs, metas, temporal = batch['history_img_inputs'], batch['history_img_metas'], batch['temporal_metas']
    b, t = inputs[0].shape[:2]
    assert b == 1 and t == len(metas)  # 逐样本检查兼容变长点云辅助数据。
    assert inputs[0].shape[2:] == batch['img_inputs'][0].shape[1:]
    for slot in range(t):
        check_batch(dict(img_inputs=tuple(x[:, slot] for x in inputs),
                         img_metas=metas[slot], gt_occ=batch['gt_occ']))
        assert metas[slot]['scene_name'] == batch['img_metas']['scene_name']
        assert metas[slot]['token'] == temporal['history_tokens'][slot]
    assert temporal['current_token'] == batch['img_metas']['token']
    a, inverse = temporal['current_to_history'], temporal['history_to_current']
    assert a.shape == (b, t, 4, 4) and torch.isfinite(a).all()
    eye = torch.eye(4).expand_as(a)
    torch.testing.assert_close(a @ inverse, eye, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(a[..., 3, :], eye[..., 3, :])
    r = a[..., :3, :3]
    torch.testing.assert_close(r.transpose(-1, -2) @ r, eye[..., :3, :3], atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(b, t), atol=1e-4, rtol=1e-5)
    #! 独立按绝对位姿验证方向，不只是检查正逆矩阵互相抵消。
    expected = torch.linalg.inv(temporal['history_lidar_to_global']) @ temporal['current_lidar_to_global'][:, None]
    torch.testing.assert_close(a.double(), expected, atol=1e-5, rtol=1e-5)
    valid = temporal['history_valid']
    assert (temporal['frame_gaps'][valid] > 0).all()
    assert (temporal['time_offsets_seconds'][valid] < 0).all()
    assert (temporal['frame_gaps'][~valid] == 0).all()
    assert (temporal['time_offsets_seconds'][~valid] == 0).all()
    torch.testing.assert_close(a[~valid], eye[~valid], atol=1e-5, rtol=1e-5)
    for slot in range(t):
        if not valid[0, slot]:
            assert metas[slot]['token'] == batch['img_metas']['token']
            for historical, current in zip(inputs, batch['img_inputs']):
                torch.testing.assert_close(historical[:, slot], current)
    print('当前:', temporal['current_token'], '历史:', temporal['history_tokens'])
    print('历史双目:', tuple(inputs[0].shape), 'current→history:', tuple(a.shape))
    print('历史时间偏移（名义秒）:', temporal['time_offsets_seconds'].tolist())
    print('真实历史标记（False=当前帧补齐）:', valid.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/configs/foundationssc_temporal/foundationssc_temporal_semantic_kitti.py')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 10])
    parser.add_argument('--ann-file')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mmcv import Config
    from mmcv.parallel import collate
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin  # 注册数据接口；不导入 temporal 模型或调用 custom_imports。
    cfg = Config.fromfile(args.config)
    options = copy.deepcopy(cfg.data[args.split])
    if args.ann_file:
        options['ann_file'] = args.ann_file
    runner_dataset = build_dataset(options)
    options['pipeline'][-1]['runner_format'] = False
    dataset = build_dataset(options)
    for index in args.indices:
        if not 0 <= index < len(dataset):
            raise IndexError(f'{index} 不在 [0,{len(dataset)})')
        sample = dataset[index]
        check_temporal(default_collate([sample]))
        #* 真正执行 MMCV collate，检查 stack/pad_dims 接口；不宣称验证 DDP/CUDA scatter。
        packed = collate([runner_dataset[index], runner_dataset[index]], samples_per_gpu=2)
        expected = torch.stack([sample['history_img_inputs'][0]] * 2)
        torch.testing.assert_close(packed['history_img_inputs'][0].data[0], expected)
        assert len(packed['temporal_metas'].data[0]) == 2
        print(f'样本 {index}：数据与双样本 MMCV 打包通过。')
    print('仅数据接口验证完成；未执行特征提取、时序融合、损失或模型前向。')


if __name__ == '__main__':
    main()
