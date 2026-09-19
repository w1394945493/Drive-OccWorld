"""FoundationSSC 原始扩展的本地入口；CPU 导入不强制加载 CUDA/MMCV。"""
from .functional import lift_pool, deform_attention, prepare_attention, deform_attention_prepared, require_extension, use_cuda

__all__ = ['lift_pool', 'deform_attention', 'prepare_attention', 'deform_attention_prepared', 'require_extension', 'use_cuda']
