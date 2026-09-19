"""就地编译 FoundationSSC 原 bev_pool / DFA3D；2D attention 由 mmcv-full 提供。"""
import os
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  #* 无论从哪里执行 setup.py，扩展始终输出到本目录子包。
os.environ.setdefault('MAX_JOBS', '2')
csrc = ROOT / 'dfa3D/ops/csrc'
dfa_sources = sorted(str(p) for pattern in ('*.cpp', 'cpu/*.cpp', 'cuda/*.cu', 'cuda/*.cpp') for p in csrc.glob(pattern))

setup(
    name='foundationssc-local-ops',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='bev_pool.bev_pool_ext',
            sources=['bev_pool/src/bev_pool.cpp', 'bev_pool/src/bev_pool_cuda.cu'],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={'cxx': [], 'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__']},
        ),
        CUDAExtension(
            name='dfa3D._ext', sources=dfa_sources,
            include_dirs=[str(csrc / 'common'), str(csrc / 'common/cuda')],
            define_macros=[('WITH_CUDA', None)],
            extra_compile_args={'cxx': ['-std=c++17'], 'nvcc': ['-std=c++17']},
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
