# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import re
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from mmcv.utils import TORCH_VERSION, digit_version


def find_latest_checkpoint(work_dir):
    """Find latest checkpoint in work_dir for auto-resume.

    #! 自动断点续训逻辑：
    #! 1. 优先使用 MMCV CheckpointHook 通常维护的 latest.pth；
    #! 2. 如果 latest.pth 不存在，则在保存目录中查找 epoch_*.pth / iter_*.pth；
    #! 3. epoch/iter 数字越大，认为 checkpoint 越新；
    #! 4. 如果数字相同，使用文件修改时间更新的那个。
    #!
    #! 注意：这里返回的是用于 runner.resume() 的完整训练状态 checkpoint，
    #! 会恢复 model / optimizer / lr scheduler / epoch 或 iter。
    #! 这和 load_from 只加载模型权重不同。
    """
    if work_dir is None:
        return None

    work_dir = osp.abspath(work_dir)
    if not osp.isdir(work_dir):
        return None

    latest_path = osp.join(work_dir, 'latest.pth')
    if osp.isfile(latest_path):
        return latest_path

    checkpoint_pattern = re.compile(r'^(epoch|iter)_(\d+)\.pth$')
    candidates = []
    for filename in os.listdir(work_dir):
        match = checkpoint_pattern.match(filename)
        if match is None:
            continue
        checkpoint_path = osp.join(work_dir, filename)
        if not osp.isfile(checkpoint_path):
            continue
        checkpoint_type = match.group(1)
        checkpoint_step = int(match.group(2))
        checkpoint_mtime = osp.getmtime(checkpoint_path)
        # epoch checkpoint 和 iter checkpoint 通常不会混用；
        # 若混用，仅按数字和修改时间选最新，避免复杂推断。
        candidates.append(
            (checkpoint_step, checkpoint_mtime, checkpoint_type,
             checkpoint_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][-1]


def sync_total_epochs_to_runner(cfg, cfg_options=None):
    """Sync cfg.total_epochs to cfg.runner.max_epochs when appropriate.

    #! 修复原因：
    #! 当前 config 中同时保留了 OpenMMLab 常见的 total_epochs 字段和
    #! EpochBasedRunner 真正使用的 runner.max_epochs 字段。命令行调试时如果只写：
    #!   --cfg-options total_epochs=4
    #! runner.max_epochs 仍会保持配置文件中的默认值，导致需要每次同时写：
    #!   total_epochs=4 runner.max_epochs=4
    #!
    #! 这里在用户没有显式覆盖 runner.max_epochs 时，自动把 total_epochs 同步到
    #! runner.max_epochs，避免重复配置；如果用户显式写了 runner.max_epochs，
    #! 则尊重用户设置。
    """
    if cfg_options is not None and 'runner.max_epochs' in cfg_options:
        return
    if 'total_epochs' not in cfg:
        return
    if 'runner' not in cfg:
        return
    if cfg.runner.get('type', None) != 'EpochBasedRunner':
        return
    cfg.runner.max_epochs = cfg.total_epochs


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-auto-resume',
        action='store_true',
        help='disable automatically resuming from latest checkpoint in work_dir')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    sync_total_epochs_to_runner(cfg, args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            from projects.mmdet3d_plugin.bevformer.apis.train import custom_train_model
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    # meta['env_info'] = env_info
    # meta['config'] = cfg.pretty_text
    #! 日志精简：
    #! MMCV TextLoggerHook 会把 runner.meta 写入 .log.json 的第一行。
    #! 如果这里保存 env_info 和完整 cfg.pretty_text，json 文件开头会出现
    #! 一大段环境信息和完整配置，可读性很差且对训练曲线分析没有必要。
    #! 普通 .log 里上面已经 logger.info 打印了 Environment info，下面也会
    #! 打印 Config，因此这里不再把它们塞进 meta / .log.json。

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    #* ================== 自动断点续训 ==================
    # 如果用户没有通过 --resume-from 或 cfg.resume_from 显式指定断点，
    # 则自动从 work_dir 中寻找之前保存的 checkpoint。
    #
    #! 这样训练中断后，重新运行同一个 --work-dir 会优先恢复完整训练状态；
    #! 若找不到 checkpoint，则保持原逻辑，继续使用 cfg.load_from 加载预训练权重。
    auto_resume = cfg.get('auto_resume', True) and not args.no_auto_resume
    if args.resume_from is not None and not osp.isfile(args.resume_from):
        logger.warning(
            f'--resume-from is specified but file does not exist: '
            f'{args.resume_from}. Auto-resume will be skipped.')
    elif auto_resume and not cfg.get('resume_from', None):
        latest_checkpoint = find_latest_checkpoint(cfg.work_dir)
        if latest_checkpoint is not None:
            cfg.resume_from = latest_checkpoint
            logger.info(
                f'Auto-resume enabled: found checkpoint {latest_checkpoint}')
        else:
            logger.info(
                f'Auto-resume enabled, but no checkpoint found in '
                f'{osp.abspath(cfg.work_dir)}. Training will start normally.')
    elif not auto_resume:
        logger.info('Auto-resume disabled.')
    else:
        logger.info(f'Resume checkpoint explicitly set: {cfg.resume_from}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
