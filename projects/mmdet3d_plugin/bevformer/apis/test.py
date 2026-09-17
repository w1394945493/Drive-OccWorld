
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


import mmcv
import numpy as np
import pycocotools.mask as mask_util

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, show=False, out_dir=None):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """

    model.eval()

    # init predictions
    iou_metric = []
    iou_current_metric = []
    iou_future_metric = []
    iou_future_time_weighting_metric = []
    iou_per_frame_metric = []
    vpq_metric = []
    plan_metric = {
            'plan_L2_1s':[],
            'plan_L2_2s':[],
            'plan_L2_3s':[],
            'plan_obj_col_1s':[],
            'plan_obj_col_2s':[],
            'plan_obj_col_3s':[],
            'plan_obj_box_col_1s':[],
            'plan_obj_box_col_2s':[],
            'plan_obj_box_col_3s':[],
            'plan_L2_1s_single':[],
            'plan_L2_2s_single':[],
            'plan_L2_3s_single':[],
            'plan_obj_col_1s_single':[],
            'plan_obj_col_2s_single':[],
            'plan_obj_col_3s_single':[],
            'plan_obj_box_col_1s_single':[],
            'plan_obj_box_col_2s_single':[],
            'plan_obj_box_col_3s_single':[],
    }

    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    def _to_result_dict(result):
        """Normalize model test output to dict.

        #! 修复原因：
        #! 为了兼容 mmdet/apis/single_gpu_test()，Drive_OccWorld.forward_test()
        #! 现在返回 [test_output]；但本工程自定义的 custom_multi_gpu_test()
        #! 原来假设 result 一定是 dict，并直接调用 result.keys()。
        #! 多卡评估时因此会报：
        #! AttributeError: 'list' object has no attribute 'keys'
        #!
        #! 这里兼容两种格式：
        #! - dict: 原始 Drive-OccWorld 多卡评估返回；
        #! - [dict]: mmdet 标准 single_gpu_test 兼容返回。
        """
        if isinstance(result, list):
            assert len(result) == 1 and isinstance(result[0], dict), \
                'custom_multi_gpu_test expects result to be dict or [dict].'
            return result[0]
        assert isinstance(result, dict), \
            'custom_multi_gpu_test expects result to be dict or [dict].'
        return result

    def _sum_per_frame_hist(hist_list):
        """Sum per-frame confusion matrices collected on one rank."""
        if len(hist_list) == 0:
            return []
        num_frames = len(hist_list[0])
        return [
            sum(sample_hist[frame_idx] for sample_hist in hist_list)
            for frame_idx in range(num_frames)
        ]

    for i, data in enumerate(data_loader):

        with torch.no_grad():

            result = model(return_loss=False, rescale=True, **data)
            result = _to_result_dict(result)

            if 'hist_for_iou' in result.keys():
                iou_metric.append(result['hist_for_iou'])
            if 'hist_for_iou_current' in result.keys():
                iou_current_metric.append(result['hist_for_iou_current'])
            if 'hist_for_iou_future' in result.keys():
                iou_future_metric.append(result['hist_for_iou_future'])
            if 'hist_for_iou_future_time_weighting' in result.keys():
                iou_future_time_weighting_metric.append(result['hist_for_iou_future_time_weighting'])
            if 'hist_for_iou_per_frame' in result.keys():
                iou_per_frame_metric.append(result['hist_for_iou_per_frame'])
            if 'vpq' in result.keys():
                vpq_metric.append(result['vpq'])
            if 'plan_metric' in result.keys():
                for key in plan_metric.keys():
                    plan_metric[key].append(result['plan_metric'][key])

            batch_size = 1

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect lists from multi-GPUs
    res = {}

    if 'hist_for_iou' in result.keys():
        iou_metric = [sum(iou_metric)]
        iou_metric = collect_results_cpu(iou_metric, len(dataset), tmpdir)
        res['hist_for_iou'] = iou_metric

    if 'hist_for_iou_current' in result.keys():
        iou_current_metric = [sum(iou_current_metric)]
        iou_current_metric = collect_results_cpu(iou_current_metric, len(dataset), tmpdir)
        res['hist_for_iou_current'] = iou_current_metric

    if 'hist_for_iou_future' in result.keys():
        iou_future_metric = [sum(iou_future_metric)]
        iou_future_metric = collect_results_cpu(iou_future_metric, len(dataset), tmpdir)
        res['hist_for_iou_future'] = iou_future_metric

    if 'hist_for_iou_future_time_weighting' in result.keys():
        iou_future_time_weighting_metric = [sum(iou_future_time_weighting_metric)]
        iou_future_time_weighting_metric = collect_results_cpu(iou_future_time_weighting_metric, len(dataset), tmpdir)
        res['hist_for_iou_future_time_weighting'] = iou_future_time_weighting_metric

    # *============================================================#
    if 'hist_for_iou_per_frame' in result.keys():
        # 每个样本返回的是 [step0_hist, step1_hist, ...]；
        # 先在当前 rank 内按 step 累加，再交给 collect_results_cpu 汇总各 rank。
        iou_per_frame_metric = [_sum_per_frame_hist(iou_per_frame_metric)]
        iou_per_frame_metric = collect_results_cpu(
            iou_per_frame_metric, len(dataset), tmpdir)
        res['hist_for_iou_per_frame'] = iou_per_frame_metric

    if 'vpq' in result.keys():
        res['vpq_len'] = len(dataset)   # 5569
        vpq_metric = [sum(vpq_metric)]  # [一张卡上所有样本的和]
        vpq_metric = collect_results_cpu(vpq_metric, len(dataset), tmpdir)  # [每张 卡上所有样本的和]
        res['vpq_metric'] = vpq_metric

    if 'plan_metric' in result.keys():
        res['data_len'] = len(dataset)
        plan_metric = {key:[sum(plan_metric[key])] for key in plan_metric.keys()}
        plan_metric = {key:collect_results_cpu(plan_metric[key], len(dataset), tmpdir) for key in plan_metric.keys()}
        res['plan_metric'] = plan_metric
    # v2
    if model.module.turn_on_plan:
        planning_result = model.module.planning_metric_v2.compute()
        model.module.planning_metric_v2.reset()
        res['planning_results_computed'] = planning_result

    return res

def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]

        # remove tmp dir
        # shutil.rmtree(tmpdir)
        #! 修复原因：
        #! custom_multi_gpu_test() 会针对多个指标多次调用 collect_results_cpu()
        #! （hist_for_iou / hist_for_iou_current / hist_for_iou_future / vpq 等）。
        #! 当 EvalHook 传入固定 tmpdir，例如 work_dir/.eval_hook 时，如果第一次
        #! collect 后 rank0 立即删除整个 tmpdir，其他 rank 在下一次 collect 中刚写出
        #! 的 part_*.pkl 可能被异步删除，导致 rank0 读取 part_1.pkl 时报：
        #! FileNotFoundError: .../.eval_hook/part_1.pkl。
        #!
        #! 这里保留 part_*.pkl，不在每个指标收集后删除目录；后续同名文件会被覆盖，
        #! 文件很小，对磁盘影响可以忽略。若需要清理，可在整轮评估结束后统一清理。
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)
