
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import json
import os.path as osp
import re

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


class _ForecastEvalMixin:
    # todo: 单卡和多卡共用调度/JSON 格式化，避免只修复 DDP 而单卡仍走原生日志。

    def __init__(self, *args, dynamic_intervals=None,  **kwargs):
        super().__init__(*args, **kwargs)
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

    @staticmethod
    def _json_safe_value(value):
        """Convert common metric values to JSON-serializable Python scalars."""
        if hasattr(value, 'item'):
            try:
                return value.item()
            except ValueError:
                pass
        if isinstance(value, (list, tuple)):
            return [_ForecastEvalMixin._json_safe_value(v) for v in value]
        if isinstance(value, dict):
            return {
                k: _ForecastEvalMixin._json_safe_value(v)
                for k, v in value.items()
            }
        return value

    @staticmethod
    def _format_eval_time_label(step_idx, interval=0.5):
        """Format forecast step index as current / 0.5s / 1s / 2s.

        #* Dataset.evaluate() 返回的 compact 指标名称是 step_i_mIoU /
        #* step_i_IoU，其中：
        #*   step_0 表示当前参考帧 current；
        #*   step_1 表示第 1 个未来 occupancy 关键帧；
        #*   step_2 表示第 2 个未来 occupancy 关键帧；以此类推。
        #*
        #* SemanticKITTI 当前 occupancy 关键帧按约 2Hz 组织，
        #* 相邻关键帧间隔 interval=0.5s。因此保存 json 时将：
        #*   step_0 -> current
        #*   step_1 -> 0.5s
        #*   step_2 -> 1s
        #*   step_3 -> 1.5s
        #*   step_4 -> 2s
        #* 这样后续画曲线或对照论文表格时更直观。
        """
        if step_idx == 0:
            return 'current'
        seconds = step_idx * interval
        if abs(seconds - round(seconds)) < 1e-8:
            seconds = int(round(seconds))
        return f'{seconds}s'

    def _dump_eval_results_to_json(self, runner):
        """Append compact eval metrics to the same .log.json used by MMCV.

        #! 修复原因：
        #! SemanticKITTIWorldDataset.evaluate() 已经会打印 compact table。
        #! 为避免 TextLoggerHook 在终端二次打印 “Epoch [...], time=0,
        #! memory=...” 形式的误导日志，后面会 clear log_buffer。
        #! 但 clear 后 MMCV 的 json 日志也拿不到评估指标。
        #! 因此这里在 clear 前手动把当前 runner.log_buffer.output 中的
        #! 精简评估指标写入 {timestamp}.log.json，做到：
        #!   - 终端只保留 compact table；
        #!   - json 文件仍记录 current / 0.5s / 1s / ... / avg 的评估结果，
        #!     不再保存 eval_iter_num/time/data_time 等辅助字段，便于后续画曲线。
        """
        timestamp = getattr(runner, 'timestamp', None)
        if timestamp is None:
            return
        json_log_path = osp.join(runner.work_dir, f'{timestamp}.log.json')
        log_dict = {
            'mode': 'val',
            'epoch': runner.epoch + 1,
            #* after_train_epoch 时 iter 已递增；after_train_iter 时尚未递增。
            'iter': runner.iter if self.by_epoch else runner.iter + 1,
        }

        dataset = getattr(self.dataloader, 'dataset', None)
        #* 优先从 Dataset 读取预测时间间隔。
        #* SemanticKITTIWorldDataset 中定义 forecast_time_interval=0.5，
        #* 表示 step_1/step_2/... 分别对应 0.5s/1s/...。
        #* 如果其他数据集没有该字段，则默认按 0.5s 处理。
        time_interval = getattr(dataset, 'forecast_time_interval', 0.5)

        #* 只匹配 compact forecast 指标：
        #*   step_0_mIoU / step_0_IoU / step_1_mIoU / ...
        #* 不再保存 eval_iter_num、time、data_time 或逐类别 IoU，
        #* 保持 .log.json 中评估记录足够简洁。
        metric_pattern = re.compile(r'^step_(\d+)_(mIoU|IoU)$')
        grouped_metrics = {}

        for key, value in runner.log_buffer.output.items():
            match = metric_pattern.match(key)
            if match:
                step_idx = int(match.group(1))
                metric_name = match.group(2)
                #* 将 step_i 转成可读时间标签：
                #*   step_0 -> current，step_1 -> 0.5s，step_2 -> 1s。
                #* 同一时刻的 mIoU 和 IoU 先归组，再合并为一个 JSON 字段。
                time_label = self._format_eval_time_label(
                    step_idx, time_interval)
                grouped_metrics.setdefault(time_label, {})[metric_name] = (
                    self._json_safe_value(value))
            elif key in ('current_mIoU', 'current_IoU'):
                #* 当前帧 SSC 的精简指标也写入百分比 JSON。
                grouped_metrics.setdefault('current', {})[key[8:]] = self._json_safe_value(value)
            elif key in ('avg_mIoU', 'avg_IoU'):
                grouped_metrics.setdefault('avg', {})[key[4:]] = (
                    self._json_safe_value(value))

        #* 合并示例："current_mIoU/IoU": "12.30%/30.00%"。
        #* 斜杠两侧固定为 mIoU、IoU，将原始 0~1 指标乘以 100，保留两位小数。
        #* 含百分号和斜杠的值保存为 JSON 字符串；后续画曲线时可先用
        #* value.split('/') 拆分，去掉 % 后转为 float。缺失指标写 null 字样，
        #* 避免误记为 0；这里只改变日志格式，评估和最优 checkpoint 仍用数值。
        for time_label, metrics in grouped_metrics.items():
            values = [metrics.get(name) for name in ('mIoU', 'IoU')]
            log_dict[f'{time_label}_mIoU/IoU'] = '/'.join(
                'null' if value is None else f'{value * 100:.2f}%' for value in values)

        with open(json_log_path, 'a') as f:
            f.write(json.dumps(log_dict, ensure_ascii=False) + '\n')

    def _finish_compact_logging(self, runner):
        if getattr(self.dataloader.dataset, 'suppress_eval_log_buffer', False):
            self._dump_eval_results_to_json(runner)
            runner.log_buffer.clear()


class CustomEvalHook(_ForecastEvalMixin, BaseEvalHook):
    """单卡评估：复用百分比 JSON 保存和日志清理，不执行分布式通信。"""

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return
        from mmdet.apis import single_gpu_test

        was_training = runner.model.training
        try:
            model = getattr(runner.model, 'module', runner.model)
            if getattr(model, 'occupancy_eval_per_sample', False):
                #* FoundationSSC 没有检测框/传统 img 字段，使用自己的逐样本统计入口。
                from projects.mmdet3d_plugin.bevformer.apis.test import _test_current_occupancy
                results = _test_current_occupancy(runner.model, self.dataloader)
            else:
                results = single_gpu_test(runner.model, self.dataloader, show=False)
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)
            if self.save_best and key_score is not None:
                self._save_ckpt(runner, key_score)
            # todo: 原单卡 EvalHook 不会执行我们在多卡中添加的 JSON 保存/clear，
            # 因此曾重复打印 Epoch [1][5/20]，并将原始 step_i 指标写入日志。
            self._finish_compact_logging(runner)
        finally:
            runner.model.train(was_training)
        runner.logger.info('单卡评估结束：结果已记录，评估 Hook 已返回训练流程。')


class CustomDistEvalHook(_ForecastEvalMixin, BaseDistEvalHook):

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
                #! 但在清空前，先将精简评估指标写入 .log.json，避免 json
                #! 只记录训练 loss 而缺少每轮评估结果。
                self._finish_compact_logging(runner)

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
        # todo: 各 rank 一致清理日志状态，防止后续 LoggerHook 的 collective
        # 在不同 rank 上因 log_buffer 状态不同而执行不一致。
        if getattr(self.dataloader.dataset, 'suppress_eval_log_buffer', False):
            runner.log_buffer.clear()
        if runner.rank == 0:
            runner.logger.info('多卡评估结束：结果已记录，各 rank 已完成同步。')
  
