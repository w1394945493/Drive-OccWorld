#!/usr/bin/env python3
"""Mayavi 有界面查看 occupancy.npz：pred_occ / gt_occ，支持当前帧和时序数组。"""
import argparse
import os
from pathlib import Path

import numpy as np

#* 与 FoundationSSC PNG 相同的 SemanticKITTI 配色。
from vis_foundationssc import COLORS


def load_voxels(path, key, step):
    with np.load(path, allow_pickle=False) as data:
        if key not in data:
            raise ValueError(f'NPZ 缺少 {key}；现有字段：{data.files}')
        voxels = data[key]
    if voxels.ndim == 4:  # 兼容 forecasting 导出的 [T,X,Y,Z]。
        if not 0 <= step < voxels.shape[0]:
            raise ValueError(f'step={step} 越界，时间步数为 {voxels.shape[0]}')
        voxels = voxels[step]
    elif voxels.ndim != 3 or step != 0:
        raise ValueError(f'{key} 应为 [X,Y,Z] 或 [T,X,Y,Z]；shape={voxels.shape}, step={step}')
    if not np.issubdtype(voxels.dtype, np.integer):
        raise ValueError(f'{key} 应为整数类别数组，不是 logits：{voxels.dtype}')
    if not np.isin(voxels, list(range(20)) + [255]).all():
        raise ValueError(f'{key} 含非 SemanticKITTI 0..19/255 标签')
    return voxels


def occupied_points(voxels, bounds, stride=1, classes=None):
    """按 [X,Y,Z] 轴序将体素中心映射到米；不交换轴、不改变坐标范围。"""
    low, high = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    size = (high - low) / np.asarray(voxels.shape)
    #* stride 仅稀疏显示体素，位置仍使用原网格坐标；不合并或放大体素。
    sparse = voxels[::stride, ::stride, ::stride]
    mask = (sparse != 0) & (sparse != 255)
    if classes is not None:
        mask &= np.isin(sparse, classes)
    indices = np.argwhere(mask)
    points = low + (indices * stride + .5) * size
    return points, sparse[mask], size


def draw(mlab, voxels, bounds, title, stride, classes):
    fig = mlab.figure(title, size=(850, 750), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    points, labels, size = occupied_points(voxels, bounds, stride, classes)
    if len(points):
        #* 标签只控制颜色，不能控制 cube 大小；每个 cube 对应真实体素边长。
        glyph = mlab.points3d(*points.T, labels.astype(float), mode='cube',
                              scale_factor=1., scale_mode='none', vmin=0, vmax=19,
                              figure=fig)
        glyph.glyph.glyph_source.glyph_source.x_length = float(size[0])
        glyph.glyph.glyph_source.glyph_source.y_length = float(size[1])
        glyph.glyph.glyph_source.glyph_source.z_length = float(size[2])
        glyph.module_manager.scalar_lut_manager.lut.table = np.column_stack(
            (COLORS, np.full(20, 255, dtype=np.uint8)))
    else:
        print(f'{title}：无满足条件的非空体素，仅显示空间边界。')
    extent = (bounds[0], bounds[3], bounds[1], bounds[4], bounds[2], bounds[5])
    mlab.outline(extent=extent, figure=fig)
    mlab.axes(extent=extent, ranges=extent, xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', figure=fig)
    mlab.title(title, size=.3, height=.95, figure=fig)
    center = (np.asarray(bounds[:3]) + np.asarray(bounds[3:])) / 2
    distance = np.linalg.norm(np.asarray(bounds[3:]) - np.asarray(bounds[:3])) * 1.2
    mlab.view(azimuth=220, elevation=65, distance=distance, focalpoint=center, figure=fig)
    print(f'{title}：显示 {len(points):,} 个体素，体素尺寸(m)={size.tolist()}')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz', type=Path, help='occupancy.npz 路径')
    parser.add_argument('--mode', choices=('both', 'pred', 'gt'), default='both')
    parser.add_argument('--step', type=int, default=0, help='时序 NPZ 的时间步；当前帧 NPZ 保持 0')
    parser.add_argument('--pc-range', nargs=6, type=float, default=[0, -25.6, -2, 51.2, 25.6, 4.4],
                        metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
                        help='必须与生成 NPZ 时配置一致，默认 SemanticKITTI 范围')
    parser.add_argument('--stride', type=int, default=1, help='显示抽样步长；卡顿时可设 2/3，不改变原 NPZ')
    parser.add_argument('--classes', nargs='+', type=int, help='仅显示指定非空类，例如 1 6 13')
    parser.add_argument('--mask-ignore', action='store_true', help='用 GT 的 255 区域屏蔽预测；默认显示原始预测')
    args = parser.parse_args()
    if args.stride < 1 or np.any(np.asarray(args.pc_range[3:]) <= np.asarray(args.pc_range[:3])) or not np.isfinite(args.pc_range).all():
        parser.error('stride 必须大于 0，pc-range 必须是有限且递增的空间边界')
    if args.classes is not None and any(c < 1 or c > 19 for c in args.classes):
        parser.error('--classes 仅接受 1..19；empty=0 和 ignore=255 默认不显示')
    pred = load_voxels(args.npz, 'pred_occ', args.step) if args.mode != 'gt' else None
    gt = load_voxels(args.npz, 'gt_occ', args.step) if args.mode != 'pred' or args.mask_ignore else None
    if pred is not None and gt is not None:
        if pred.shape != gt.shape:
            parser.error(f'pred/GT shape 不一致：{pred.shape} / {gt.shape}')
        if args.mask_ignore:
            pred = pred.copy()
            pred[gt == 255] = 255
    #* Mayavi 依赖图形桌面及 Qt/VTK；不使用 Agg/offscreen，不需要模型或 CUDA。
    if os.name != 'nt' and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('未检测到图形显示服务；请在桌面/WSLg 或已配置 X11 转发的环境运行')
    try:
        from mayavi import mlab
    except ImportError as exc:
        raise SystemExit(f'Mayavi/Qt/VTK 未就绪，请在图形环境安装相应依赖。原错误：{exc}') from exc
    figures = []
    for voxels, name in ((pred, 'Pred current' if args.step == 0 else f'Pred step {args.step}'),
                         (gt if args.mode != 'pred' else None, 'GT current' if args.step == 0 else f'GT step {args.step}')):
        if voxels is not None:
            figures.append(draw(mlab, voxels, args.pc_range, name, args.stride, args.classes))
    if len(figures) == 2:
        #* 两个独立交互窗口，可在桌面并排摆放；旋转/缩放同步，方便对比。
        mlab.sync_camera(figures[0], figures[1])
    mlab.show()


if __name__ == '__main__':
    main()
