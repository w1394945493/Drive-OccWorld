PYTHONPATH="$(pwd)" \
torchrun \
    --nproc_per_node="${MLP_WORKER_GPU}" \
    --master_addr="${MLP_WORKER_0_HOST}" \
    --node_rank="${MLP_ROLE_INDEX}" \
    --master_port="${MLP_WORKER_0_PORT}" \
    --nnodes="${MLP_WORKER_NUM}" \
    /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/tools/train.py \
    /vepfs-mlp2/c20250502/haoce/wangyushen/Drive-OccWorld/projects/configs/kitti/semantic_kitti_drive_occworld.py \
    --launcher pytorch \
    --work-dir /c20250502/wangyushen/Outputs/drive_occworld/semkitti/train
