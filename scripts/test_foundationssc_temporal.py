#!/usr/bin/env python3
"""时序接口完整检查：数据 → 当前帧 SSC 前向 → 评估，可选损失反向；尚无历史融合。"""
import argparse
import copy
import importlib
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
        #! default_collate 对两种嵌套结构分别产生 list/tuple；统一容器类型，仍逐项检查帧名。
        assert tuple(metas[slot]['token']) == tuple(temporal['history_tokens'][slot])
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
    parser.add_argument('--split', choices=['train', 'val'], default=None,
                        help='默认推理用 val，--check-grad 用 train；辅助监督需训练 pipeline')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 10])
    parser.add_argument('--ann-file')
    parser.add_argument('--checkpoint', help='优先于配置 load_from；均未设置时仅验证初始化模型接口')
    parser.add_argument('--check-grad', action='store_true', help='检查真实损失与反向梯度，不更新参数')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    cfg = Config.fromfile(args.config)
    for name in cfg.get('custom_imports', {}).get('imports', []):
        importlib.import_module(name)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('完整模型验证需要 CUDA 环境及已编译的体素算子')
    torch.cuda.set_device(device)
    split = args.split or ('train' if args.check_grad else 'val')
    options = copy.deepcopy(cfg.data[split])
    if args.check_grad and (cfg.model.get('use_depth_loss') or cfg.model.get('use_semantic_loss')):
        if not any(step['type'] == 'ProjectFoundationSSCLidar' for step in options['pipeline']):
            parser.error('辅助损失检查需要投影标签，请使用 --split train')
    if args.ann_file:
        options['ann_file'] = args.ann_file
    runner_dataset = build_dataset(options)
    options['pipeline'][-1]['runner_format'] = False
    dataset = build_dataset(options)
    for index in args.indices:
        if not 0 <= index < len(dataset):
            raise IndexError(f'{index} 不在 [0,{len(dataset)})')
    model = build_detector(cfg.model).to(device)
    checkpoint = args.checkpoint or cfg.get('load_from')
    if checkpoint:
        #! 当前子类参数结构与单帧相同，严格加载，避免遗漏权重仍误报验证通过。
        load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
        print(f'加载完整模型权重：{checkpoint}')
    else:
        print('未加载完整 SSC 检查点（立体骨干仍加载配置权重）；指标仅用于接口检查。')
    print(f'数据划分：{split}；仅执行当前帧 SSC，历史输入暂不参与融合。')
    results = []
    torch.cuda.reset_peak_memory_stats(device)
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
        #! 使用正式 DataContainer→collate→scatter 路径，历史字段一并传入模型。
        batch = scatter(collate([runner_dataset[index]], samples_per_gpu=1),
                        [torch.cuda.current_device()])[0]
        model.eval()
        with torch.no_grad():
            output = model(return_loss=False, return_outputs=True, **batch)
            pred, logits = output['pred'], output['output_voxels']
            assert pred.shape == batch['gt_occ'].shape
            assert logits.shape == (pred.shape[0], model.pts_bbox_head.out_channel, *pred.shape[1:])
            assert torch.isfinite(logits).all() and torch.isfinite(output['voxel_feats']).all()
            print(f"voxel_feats={tuple(output['voxel_feats'].shape)}；logits={tuple(logits.shape)}；pred={tuple(pred.shape)}")
            del output, pred, logits
            # 单独验证正式评估入口，而非仅手工调用混淆矩阵函数。
            items = model(return_loss=False, **batch)
            assert len(items) == batch['gt_occ'].shape[0]
            assert all(len(item['hist_for_iou_per_frame']) == 1 for item in items)
            results.extend(items)
        if args.check_grad:
            model.train()
            model.zero_grad(set_to_none=True)
            losses = model(return_loss=True, **batch)
            assert losses and all(torch.isfinite(value).all() for value in losses.values())
            for enabled, key in ((model.use_depth_loss, 'loss_depth'), (model.use_semantic_loss, 'loss_seg_ce')):
                assert (key in losses) == enabled, key
            total = sum(value.mean() for key, value in losses.items() if 'loss' in key)
            total.backward()
            assert not model.img_backbone.training
            assert all(p.grad is None for p in model.img_backbone.parameters())
            names = ['image_pyramid', 'voxel_encoder', 'occ_encoder_backbone', 'occ_encoder_neck', 'pts_bbox_head']
            if model.use_semantic_loss:
                names.append('plugin_head')
            for name in names:
                grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads), f'{name} 梯度缺失或非有限'
                assert any(g.abs().sum() > 0 for g in grads), f'{name} 梯度全零'
            print('损失：' + ', '.join(f'{key}={value.mean().item():.6f}' for key, value in losses.items()))
            print(f'总损失={total.item():.6f}；可训练模块梯度有效，冻结骨干无梯度；未执行 optimizer.step。')
            model.zero_grad(set_to_none=True)
            del losses, total, grads
        del batch
    print('所选样本评估（不是完整验证集指标）：')
    dataset.evaluate(results)
    print(f'显存峰值：{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB')
    print('数据、当前帧前向和评估流程通过；尚未实现/验证历史时序融合或 DDP 训练。')


if __name__ == '__main__':
    main()
