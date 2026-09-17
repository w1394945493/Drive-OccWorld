# * 1. 就地编译 DCNv3
cd projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3
python setup.py build_ext --inplace

# * 2. 就地编译 dvxlr 和 dvxlr_v2
cd third_lib/dvxlr
python setup.py build_ext --inplace

# drive-occworld train
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)" \
python /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/train.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/projects/configs/fine_grained/action_condition_MMO_MSO_custom.py \
    --work-dir out/mmo_mso/train

# offline pkl generate 在原pkl文件基础上，生成离线版本的v2.pkl文件
python /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/gen_new_data.py

# use offline dataset 
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)" \
python /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/train.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/projects/configs/fine_grained/action_condition_MMO_MSO_custom_offline.py \
    --work-dir out/mmo_mso/train

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)" \
python /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/projects/configs/fine_grained/action_condition_MMO_MSO_custom_offline.py \
    --eval bbox

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(pwd)" \
python /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/test.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/projects/configs/fine_grained/action_condition_MMO_MSO_custom_offline_with_planning.py \
    --eval bbox

# ========================================================#
python /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/semantickitti_converter.py \
    --out-pkl out/semantic_kitti/mini.pkl

python /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/compare_semkitti_nuscenes_pkl.py \
    --semkitti-pkl /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/out/semantic_kitti/mini.pkl 