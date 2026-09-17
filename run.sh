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
# semantickitti
python /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/semantickitti_converter.py \
    --out-dir /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/data/semantic_kitti

python /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/compare_semkitti_nuscenes_pkl.py \
    --semkitti-pkl /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/out/semantic_kitti/semantickitti_infos_train.pkl

python /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/scripts/test_semantic_kitti_world_dataset.py

python scripts/test_semantic_kitti_drive_occworld_forward.py \
  --config projects/configs/kitti/semantic_kitti_drive_occworld.py \
  --index 0 \
  --train-iters 100


CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH="$(pwd)" \
python tools/train.py \
  projects/configs/kitti/semantic_kitti_drive_occworld.py \
  --work-dir out/semantic_kitti_drive_occworld_epoch_debug \
  --cfg-options \
  total_epochs=2 \
  runner.max_epochs=2 \
  data.train.max_samples=20 \
  data.val.max_samples=5 \
  data.workers_per_gpu=0 \
  checkpoint_config.interval=1 \
  evaluation.interval=1

CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH="$(pwd)" \
torchrun --nproc_per_node=2 --master_port=29501 \
  tools/train.py \
  projects/configs/kitti/semantic_kitti_drive_occworld.py \
  --launcher pytorch \
  --work-dir out/semantic_kitti_drive_occworld_epoch_debug_ddp \
  --cfg-options \
  total_epochs=4 \
  data.train.max_samples=20 \
  data.val.max_samples=10 \
  data.workers_per_gpu=0 \
  checkpoint_config.interval=1 \
  evaluation.interval=1



# =====================================================================#
#  火山服务器训练

PYTHONPATH="$(pwd)" \
python -m torch.distributed.launch \
    --nproc_per_node="${MLP_WORKER_GPU}" \
    --master_addr="${MLP_WORKER_0_HOST}" \
    --node_rank="${MLP_ROLE_INDEX}" \
    --master_port="${MLP_WORKER_0_PORT}" \
    --nnodes="${MLP_WORKER_NUM}" \
    /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/train.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/projects/configs/kitti/semantic_kitti_drive_occworld.py \
    --launcher pytorch \
    --work-dir /c20250502/wangyushen/Outputs/drive_occworld/semkitti/train

cd /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld
. /root/miniconda3/bin/activate
conda activate /vepfs-mlp2/c20250502/haoce/conda_env/wys_temp_2
bash sh/train_semkitti.sh