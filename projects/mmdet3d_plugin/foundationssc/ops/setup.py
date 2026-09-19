"""在此目录执行 python setup.py build_ext --inplace，无需 pip 安装或 JIT 编译。"""
import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  #* 从任意 cwd 调用也将 .so 输出到 ops/，而不是项目根目录。
setup(
    name='foundationssc-local-ops',
    ext_modules=[CUDAExtension(
        name='_C', sources=['src/bindings.cpp', 'src/kernels.cu'],
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-lineinfo']})],
    cmdclass={'build_ext': BuildExtension},
)
