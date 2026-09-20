#!/usr/bin/env python3
"""FoundationSSC 完整验证：数据 → 图像特征 → 体素 → 占据预测 → 真实占据损失。

固定执行当前已实现的完整流程；--check-grad 可额外检查反向传播。
--check-grad 对真实 GT 占据损失反传；不使用特征探针，尚未接入评估或优化器更新。
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


def check_batch(batch):
    #* （FoundationSSC 辅助深度&语义损失) 此脚本 batch=1，逐点数组维度 [1,N,...]。
    if 'aux_lidar' in batch:
        aux = batch['aux_lidar']
        points = aux['points']
        assert points.ndim == 3 and points.shape[0] == 1 and points.shape[-1] == 4
        for key in ('semantic_raw', 'semantic_labels', 'instance_ids'):
            assert aux[key].shape == points.shape[:2], key
        assert torch.isfinite(points).all()
        assert ((aux['semantic_labels'] >= 0) & (aux['semantic_labels'] < 20)).all()
        ids, counts = torch.unique(aux['semantic_labels'], return_counts=True)
        print(f"点云路径：{aux['lidar_path'][0]}；shape={tuple(points.shape)}")
        print(f"点级标签：{aux['pts_label_path'][0]}；点数/标签数一致")
        print(f'映射语义分布：{dict(zip(ids.tolist(), counts.tolist()))}')
        print(f"非零实例ID数量：{torch.unique(aux['instance_ids'][aux['instance_ids'] > 0]).numel()}")
    images, rots, trans, intrins, post_rots, post_trans, bda, c2l = batch['img_inputs']
    assert images.ndim == 5 and images.shape[:3] == (1, 2, 3)
    assert intrins.shape == (1, 2, 4, 4) and c2l.shape == (1, 2, 4, 4)
    assert bda.shape == (1, 4, 4)
    assert batch['gt_occ'].dtype == torch.int64 and batch['gt_occ'].ndim == 4
    for tensor in batch['img_inputs']:
        assert torch.isfinite(tensor).all()
    meta = batch['img_metas']
    assert (meta['baseline'] > 0).all()
    torch.testing.assert_close(c2l[..., :3, :3], rots)
    torch.testing.assert_close(c2l[..., :3, 3], trans)
    #* 检查投影：K4 @ lidar2camera 与 PKL 的 lidar2img 必须一致。
    projected = (intrins @ torch.linalg.inv(c2l))[..., :3, :]
    torch.testing.assert_close(projected, meta['lidar2img'], atol=1e-3, rtol=1e-5)
    for cam in range(2):
        raw = meta['raw_img'][cam]
        assert raw.dtype == torch.uint8 and raw.shape == (1, images.shape[-2], images.shape[-1], 3)
        rgb = raw.permute(0, 3, 1, 2).float() / 255
        expected = (rgb - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
        torch.testing.assert_close(images[:, cam], expected)
    #* 检查 post transform 正反变换；不改变相机内参以免增强被重复应用。
    point = torch.tensor([120., 90., 1.]).expand(1, 2, 3)
    transformed = (post_rots @ point[..., None]).squeeze(-1) + post_trans
    recovered = (torch.linalg.inv(post_rots) @ (transformed - post_trans)[..., None]).squeeze(-1)
    torch.testing.assert_close(point, recovered)
    #* （FoundationSSC 辅助深度&语义损失) 稀疏标签必须与变换后的输入图像同尺寸。
    if 'gt_semantics' in batch:
        depth, semantics = meta['gt_depths'], batch['gt_semantics']
        assert depth.shape == semantics.shape and depth.shape[-2:] == images.shape[-2:]
        assert torch.isfinite(depth).all() and (depth >= 0).all()
        assert (semantics[depth == 0] == 255).all()
        assert ((semantics[depth > 0] >= 0) & (semantics[depth > 0] < 20)).all()
        for slot, cam in enumerate(meta['projection_camera_indices'][0].tolist()):
            valid = depth[0, slot] > 0
            values = depth[0, slot][valid]
            limits = (float(values.min()), float(values.max())) if values.numel() else None
            print(f'相机 {cam}：有效投影像素={int(valid.sum())}，深度范围(m)={limits}')


def save_projection_overlays(batch, out_dir):
    #* （FoundationSSC 辅助深度&语义损失) 只画最终稀疏标签覆盖在 resize/crop 后 RGB 上，
    # 上排深度、下排语义；左右共用深度色标，保存图片不需要桌面。
    import os
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from vis_foundationssc import COLORS
    meta = batch['img_metas']
    cams = meta['projection_camera_indices'][0].tolist()
    depth, semantic = meta['gt_depths'][0].numpy(), batch['gt_semantics'][0].numpy()
    scene, token = str(meta['scene_name'][0]), str(meta['token'][0])
    frame = token.rsplit('_', 1)[-1]
    for name in (scene, frame):
        if name in ('', '.', '..') or '/' in name or '\\' in name:
            raise ValueError(f'非法保存目录名：{name!r}')
    folder = Path(out_dir) / scene / frame
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(cams), figsize=(10 * len(cams), 6), squeeze=False)
    finite = depth[depth > 0]
    maximum = max(1., float(finite.max())) if finite.size else 1.
    artist = None
    for slot, cam in enumerate(cams):
        raw = meta['raw_img'][cam][0].numpy()
        y, x = np.where(depth[slot] > 0)
        for ax in axes[:, slot]:
            ax.imshow(raw)
            ax.set_xlim(-.5, raw.shape[1] - .5)
            ax.set_ylim(raw.shape[0] - .5, -.5)
            ax.axis('off')
        artist = axes[0, slot].scatter(x, y, c=depth[slot, y, x], s=2, cmap='turbo', vmin=0, vmax=maximum)
        colors = COLORS[semantic[slot, y, x]].copy()
        colors[semantic[slot, y, x] == 0] = [128, 128, 128]  # 点级 unlabeled，非 empty。
        axes[1, slot].scatter(x, y, c=colors / 255., s=2)
        side = 'Left' if cam == 0 else 'Right'
        axes[0, slot].set_title(f'{side}: depth (m), {len(x)} pixels')
        axes[1, slot].set_title(f'{side}: semantics (gray = unlabeled)')
    fig.colorbar(artist, ax=axes[0].tolist(), fraction=.025, pad=.02)
    fig.savefig(folder / 'lidar_projection_overlay.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    np.savez_compressed(folder / 'lidar_projection.npz', gt_depths=depth,
                        gt_semantics=semantic, camera_indices=np.asarray(cams), token=np.asarray(token))
    print(f'投影叠加图和稀疏标签已保存：{folder}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='projects/configs/foundationssc/foundationssc_semantic_kitti.py')
    parser.add_argument('--stereo-checkpoint', help='覆盖 FoundationStereo 权重路径')
    parser.add_argument('--stereo-config', help='覆盖 FoundationStereo YAML 路径')
    parser.add_argument('--check-grad', action='store_true', help='对真实占据损失反传，检查可训练模块梯度及骨干冻结')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--ops-backend', choices=['cuda', 'pytorch', 'auto'], help='覆盖体素汇聚与注意力算子后端')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--ann-file', help='覆盖 PKL 路径；图像/标签路径仍取 PKL')
    parser.add_argument('--indices', nargs='+', type=int, default=[0])
    #* （FoundationSSC 辅助深度&语义损失) 可不运行模型，先验证点云/标签数据接口。
    parser.add_argument('--check-lidar-labels', action='store_true')
    parser.add_argument('--pts-label-root', help='点级标签根目录，下面为 <seq>/labels/*.label')
    parser.add_argument('--data-only', action='store_true', help='只运行数据检查，不加载模型/权重或使用 CUDA')
    #* （FoundationSSC 辅助深度&语义损失) 指定目录即启用左右投影，结果按场景/帧保存。
    parser.add_argument('--projection-out-dir', help='保存左右图的深度/语义叠加 PNG 与标签 NPZ')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin  # noqa: F401，注册 Dataset 与 pipeline
    cfg = Config.fromfile(args.config)
    dataset_cfg = cfg.data[args.split].copy()
    #* 独立脚本采用普通 DataLoader；正式 train.py 才使用 MMCV DataContainer。
    dataset_cfg['pipeline'] = [dict(step, runner_format=False) if step['type'] == 'PackFoundationSSCInputs'
                               else dict(step) for step in dataset_cfg['pipeline']]
    #* （FoundationSSC 辅助深度&语义损失) 命令行直接插入步骤，避免顶层配置变量
    # 被覆盖后不能重新生成 pipeline 的问题；显式根目录也会自动启用读取。
    #* （FoundationSSC 辅助深度&语义损失) 主脚本验证损失时也需 GT；正式 val pipeline 无需。
    check_aux_loss = not args.data_only and (cfg.model.get('use_depth_loss', False) or cfg.model.get('use_semantic_loss', False))
    if args.check_lidar_labels or args.pts_label_root or args.projection_out_dir or check_aux_loss:
        steps = dataset_cfg['pipeline']
        existing = next((step for step in steps if step['type'] == 'LoadSemanticKITTIPointsAndLabels'), None)
        if existing is None:
            existing = dict(type='LoadSemanticKITTIPointsAndLabels', pts_label_root=cfg.get('pts_label_root'))
            position = next(i for i, step in enumerate(steps) if step['type'] == 'PackFoundationSSCInputs')
            steps.insert(position, existing)
        if args.pts_label_root:
            existing['pts_label_root'] = args.pts_label_root
        #* （FoundationSSC 辅助深度&语义损失) 标签检查同时检查投影；可视化时取左右。
        projection = next((step for step in steps if step['type'] == 'ProjectFoundationSSCLidar'), None)
        if projection is None:
            projection = dict(type='ProjectFoundationSSCLidar', camera_indices=(0,))
            steps.insert(steps.index(existing) + 1, projection)
        if args.projection_out_dir:
            projection['camera_indices'] = (0, 1)
    if args.ann_file:
        dataset_cfg['ann_file'] = args.ann_file
    dataset = build_dataset(dataset_cfg)
    for index in args.indices:
        if not 0 <= index < len(dataset):
            raise IndexError(f'索引 {index} 越界，样本数 {len(dataset)}')
    loader = DataLoader(Subset(dataset, args.indices), batch_size=1, num_workers=0, shuffle=False)
    if args.data_only:
        for index, batch in zip(args.indices, loader):
            check_batch(batch)
            if args.projection_out_dir:
                save_projection_overlays(batch, args.projection_out_dir)
            print(f'样本 {index} 数据检查通过；未计算辅助损失或模型前向。')
        return
    from projects.mmdet3d_plugin.foundationssc import FoundationSSCImageModel
    options = dict(cfg.model)
    options.pop('type')
    if not options.get('voxel_encoder'):
        raise ValueError('完整验证需要配置 model.voxel_encoder')
    if args.ops_backend is not None:
        options['voxel_encoder'] = dict(options['voxel_encoder'], ops_backend=args.ops_backend)
    for key in ('stereo_checkpoint', 'stereo_config'):
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('完整 FoundationSSC 特征验证需要 CUDA 环境')
    torch.cuda.set_device(device)
    model = FoundationSSCImageModel(**options).to(device)
    model.train(args.check_grad)
    print(f'权重检查：{model.checkpoint_report}')
    print(f'冻结骨干参数量：{sum(p.numel() for p in model.img_backbone.parameters()):,}')
    print(f'可训练 FPN 参数量：{sum(p.numel() for p in model.image_pyramid.parameters()):,}')
    print(f'可训练体素前端参数量：{sum(p.numel() for p in model.voxel_encoder.parameters()):,}')
    print(f'体素算子后端：{model.voxel_encoder.ops_backend}')
    for name in ('occ_encoder_backbone', 'occ_encoder_neck', 'pts_bbox_head'):
        print(f'{name} 可训练参数量：{sum(p.numel() for p in getattr(model, name).parameters() if p.requires_grad):,}')
    eval_results = []
    for index, batch in zip(args.indices, loader):
        check_batch(batch)
        if args.projection_out_dir:
            save_projection_overlays(batch, args.projection_out_dir)
        print(f"样本 {index}，场景 {batch['img_metas']['scene_name'][0]}，帧 {batch['img_metas']['token'][0]}")
        names = ('归一化图像', 'camera→lidar旋转', 'camera→lidar平移', '4x4内参', '图像变换旋转', '图像变换平移', 'BDA单位阵', 'camera→lidar矩阵')
        for name, tensor in zip(names, batch['img_inputs']):
            print(f'  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}')
        print(f"  occupancy: {tuple(batch['gt_occ'].shape)}；基线(m): {batch['img_metas']['baseline'].item():.6f}")
        print('  投影、归一化、标定和 batch 检查通过。')
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        with torch.set_grad_enabled(args.check_grad):
            if args.check_grad:
                output = model(return_loss=True, return_outputs=True, **batch)
            else:
                output = model(return_loss=False, return_outputs=True, **batch)
                output['losses'] = model.compute_losses(output, batch['gt_occ'], batch['img_metas'], batch.get('gt_semantics'))
            feature = output['img_feats']
            assert feature.shape[:3] == (1, 1, 640), feature.shape
            assert feature.shape[-2:] == tuple(v // 8 for v in batch['img_inputs'][0].shape[-2:]), feature.shape
            tensors = [feature, *output['disparity'], *output['pyramid']]
            tensors += [value for pair in output['dino_features'] for value in pair]
            assert all(torch.isfinite(value).all() for value in tensors)
            assert not model.img_backbone.training
            from projects.mmdet3d_plugin.foundationssc.voxel.geometry import project, unproject
            geom = model.voxel_encoder.geometry
            expected = (1, cfg.model.voxel_encoder.channels, *geom.voxel_shape)
            assert output['voxel_feats'].shape == expected
            for key in ('voxel_feats', 'coarse_voxel', 'refined_voxel', 'context', 'depth_prob', 'stereo_depth', 'proposal'):
                assert torch.isfinite(output[key]).all(), key
                print(f'  {key}: {tuple(output[key].shape)}')
            torch.testing.assert_close(output['depth_prob'].sum(1), torch.ones_like(output['depth_prob'][:, 0]), atol=1e-5, rtol=1e-5)
            assert (output['lifted_valid_points'] > 0).all(), '所有 LSS 射线都在范围外，请检查标定/坐标轴'
            assert (output['visible_voxels'] > 0).all(), '没有可见体素，请检查相机外参'
            print(f"  有效 LSS 点: {output['lifted_valid_points'].tolist()}；可见体素: {output['visible_voxels'].tolist()}；候选数: {int(output['proposal'].sum())}")
            #* 用实际样本标定验证正反投影；不是单纯检查张量形状。
            cam = [x[:, :1].to(device).float() for x in batch['img_inputs'][1:6]] + [batch['img_inputs'][6].to(device).float()]
            uvd = torch.tensor([[[[100., 100., 10.], [500., 200., 20.]]]], device=device)
            torch.testing.assert_close(project(unproject(uvd, cam), cam), uvd, atol=1e-3, rtol=1e-4)
            torch.testing.assert_close(geom.lower.cpu(), batch['img_metas']['pc_range'][0, :3])
            upper = geom.lower + geom.voxel_size * geom.voxel_size.new_tensor(geom.voxel_shape)
            torch.testing.assert_close(upper.cpu(), batch['img_metas']['pc_range'][0, 3:])
            print('  深度概率、实际标定正反投影和三维空间覆盖检查通过。')
            logits, prediction = output['output_voxels'], output['pred']
            assert logits.shape == (batch['gt_occ'].shape[0], model.pts_bbox_head.out_channel, *batch['gt_occ'].shape[1:])
            assert prediction.shape == batch['gt_occ'].shape and prediction.dtype == torch.int64
            assert torch.isfinite(logits).all()
            torch.testing.assert_close(prediction, logits.detach().argmax(1))
            print(f'  占据 logits: {tuple(logits.shape)}；类别预测: {tuple(prediction.shape)}')
            print(f'  标签约定：empty={model.pts_bbox_head.empty_idx}，ignore={model.pts_bbox_head.ignore_index}')
            losses = output['losses']
            #* （FoundationSSC 辅助深度&语义损失) 按开关检查损失，允许旧三项/新五项。
            expected_losses = {'loss_voxel_ce', 'loss_voxel_sem_scal', 'loss_voxel_geo_scal'}
            if model.use_depth_loss:
                expected_losses.add('loss_depth')
            if model.use_semantic_loss:
                expected_losses.add('loss_seg_ce')
            assert set(losses) == expected_losses
            for name, loss in losses.items():
                assert loss.ndim == 0 and torch.isfinite(loss), name
                print(f'  {name}: {loss.item():.6f}')
            total_loss = sum(losses.values())
            print(f'  总损失: {total_loss.item():.6f}；深度监督={model.use_depth_loss}，二维语义监督={model.use_semantic_loss}')
            if args.check_grad:
                #* 真实 GT 的 CE + semantic/geometric scaling，替代旧体素特征平方均值探针。
                total_loss.backward()
                grads = [p.grad for p in model.image_pyramid.parameters() if p.requires_grad]
                assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
                assert any(g.abs().sum() > 0 for g in grads)
                assert all(not p.requires_grad and p.grad is None for p in model.img_backbone.parameters())
                print('  FPN 梯度有效；FoundationStereo 无梯度。')
                for name in ('depth_net', 'refiner', 'fusion'):
                    module = getattr(model.voxel_encoder, name)
                    values = [p.grad for p in module.parameters() if p.grad is not None]
                    assert values and all(torch.isfinite(x).all() for x in values), name
                    assert any(torch.count_nonzero(x) for x in values), name
                    print(f'  {name} 反向梯度检查通过。')
                del values, module
                for name in ('occ_encoder_backbone', 'occ_encoder_neck', 'pts_bbox_head') + (('plugin_head',) if model.use_semantic_loss else ()):
                    values = [p.grad for p in getattr(model, name).parameters() if p.requires_grad]
                    assert values and all(x is not None and torch.isfinite(x).all() for x in values), name
                    assert any(torch.count_nonzero(x) for x in values), name
                    print(f'  {name} 真实损失反向梯度检查通过。')
                del values
        eval_results.extend(model.occupancy_results(output['pred'], batch['gt_occ'], batch['img_metas']))
        torch.cuda.synchronize(device)
        print(f'  融合图像特征: {tuple(feature.shape)}, dtype={feature.dtype}')
        print(f"  DINO各层: {[tuple(pair[0].shape) for pair in output['dino_features']]}")
        print(f"  视差概率/视差图: {[tuple(x.shape) for x in output['disparity']]}")
        print(f'  耗时: {time.perf_counter()-start:.3f}s；显存峰值: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GiB')
        del output, feature, tensors, logits, prediction, losses, total_loss, loss
        if args.check_grad:
            del grads
        model.zero_grad(set_to_none=True)
    dataset.evaluate(eval_results)
    print('完整流程检查完成：数据 → 特征 → 占据预测/损失 → 当前帧 IoU/mIoU；尚未执行 optimizer.step。')


if __name__ == '__main__':
    main()
