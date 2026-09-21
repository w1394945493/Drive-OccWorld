PYTHONPATH="$(pwd)" \
torchrun \
    --nproc_per_node="${MLP_WORKER_GPU}" \
    --master_addr="${MLP_WORKER_0_HOST}" \
    --node_rank="${MLP_ROLE_INDEX}" \
    --master_port="${MLP_WORKER_0_PORT}" \
    --nnodes="${MLP_WORKER_NUM}" \
    tools/train.py \
    projects/configs/foundationssc_forecasting/foundationssc_forecast_no_freeze.py \
    --launcher pytorch \
    --work-dir /c20250502/wangyushen/Outputs/drive_occworld/foundationssc_forecast_no_freeze/train