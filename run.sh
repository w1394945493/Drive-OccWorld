# * 1. 就地编译 DCNv3
cd projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3
python setup.py build_ext --inplace

# * 2. 就地编译 dvxlr 和 dvxlr_v2
cd third_lib/dvxlr
python setup.py build_ext --inplace
