from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


# 增加中文注释：本扩展只有 CUDA 实现；提前检查 CUDA Toolkit，避免产生难以理解的编译错误。
if CUDA_HOME is None:
    raise RuntimeError("未找到 CUDA Toolkit，请确认 nvcc 可用且 CUDA_HOME 配置正确。")


# 增加中文注释：一次构建 dvxlr 和 dvxlr_v2 两个扩展。
# 使用短模块名可让 build_ext --inplace 将两个 .so 直接写入当前目录，随后由
# ``from third_lib.dvxlr import dvxlr, dvxlr_v2`` 按包内子模块方式加载。
ext_modules = [
    CUDAExtension(
        name="dvxlr",
        sources=["dvxlr.cpp", "dvxlr_cuda.cu"],
    ),
    CUDAExtension(
        name="dvxlr_v2",
        sources=["dvxlr_v2.cpp", "dvxlr_v2_cuda.cu"],
    ),
]


setup(
    name="drive-occworld-dvxlr",
    version="1.0.0",
    description="Drive-OccWorld differentiable voxel rendering CUDA extensions",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
