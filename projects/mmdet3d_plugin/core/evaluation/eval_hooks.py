
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import os.path as osp

import mmcv
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import DistEvalHook


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None,  **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)
        # if dist.is_available() and dist.is_initialized():
        #     dist.barrier()
        # todo: 曾导致“评估表格已经打印，但迟迟不进入下一 epoch”的问题点：
        # todo: 在 before_train_epoch 里额外 barrier 太靠近 runner/hook 调度边界，
        # todo: 某些 rank 可能还没走到这里，另一些 rank 已经在等待，容易死锁。
        # todo: 因此不要在 epoch 开始前额外同步，只在 _do_evaluate() 结束处同步。
        #! 这里曾尝试在每个 epoch 开始前额外同步所有 rank，但实际多卡
        #! 训练中可能出现某些 rank 尚未进入 before_train_epoch，而另一些
        #! rank 已经在此处等待，从而导致评估指标已打印但迟迟不进入下一
        #! epoch 的死锁现象。当前只保留 _do_evaluate() 末尾的同步；
        #! epoch 开始前不再额外 barrier。

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test # to solve circlur  import

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)

            if getattr(self.dataloader.dataset,
                       'suppress_eval_log_buffer', False):
                # todo: 曾导致误判“第 2 个 epoch 卡住”的日志问题：
                # todo: Dataset.evaluate() 已经打印 compact table，但 TextLoggerHook
                # todo: 还会把 eval_results 作为普通训练日志再打印一次，显示成
                # todo: Epoch [1][5/10]、time=0、data_time=0、memory 异常等。
                # todo: 这不是新一轮训练，而是评估指标的二次日志输出。
                #! SemanticKITTIWorldDataset.evaluate() 已经主动打印了
                #! compact forecasting table。若继续保留 log_buffer.ready=True，
                #! MMCV TextLoggerHook 会把同一批 eval_results 再打印成
                #! “Epoch [1][5/10] time=0 data_time=0 memory=...” 形式，
                #! 既重复又容易误导为新一轮训练日志。这里清空 ready 状态，
                #! 保留 evaluate() 表格输出，跳过 TextLoggerHook 的二次打印。
                runner.log_buffer.clear()

        #! 修复原因：
        # todo: 必要同步点：
        # todo: custom_multi_gpu_test() 返回后，只有 rank0 会继续执行 evaluate()
        # todo: 和 compact table 打印；其他 rank 可能更早返回。这里保留一次
        # todo: eval 结束同步，避免 rank0 还在评估时其他 rank 已进入训练。
        #! 分布式评估时，custom_multi_gpu_test() 返回后只有 rank0 会继续执行
        #! dataset.evaluate()、打印表格和写 logger；其他 rank 会更早返回并可能进入
        #! 下一轮训练。若 rank0 仍在评估/写日志，而其他 rank 已经开始新的 DDP
        #! forward/backward，容易造成各 rank collective 调用顺序不一致，从而表现为
        #! “评估指标已打印，但训练长时间停住不继续”。
        #! 这里在评估 hook 末尾同步所有 rank，确保 rank0 完成评估日志后，
        #! 全部进程再一起进入下一个 epoch/iter。
        dist.barrier()
  
