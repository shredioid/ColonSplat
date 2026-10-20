#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH='.'

# Shared params -these are our baseline values, the same should be in arguments/__init__.py
DEPTH_W=0.25
DCOL_W=0.005
KNN_W=(
    0.01
    )
ITERS=30000
COL_SMOOTH_W=0.0001

# Dataset list: explicit paths (relative to repo root, or absolute).
# Each entry must be a sequence folder containing e.g. transforms.json / images.
DATASETS=(
    # "/mnt/a-1/C3VDv2/c1_sigmoid1_t4_v4"
    "/mnt/a-1/C3VDv2/c1_transverse1_t1_v4"
    "/mnt/a-1/C3VDv2/c1_transverse1_t4_v4"
    # "/root/ColonSplat/data_custom/seq_3"
)

# Optional: override the dataset list from the command line, e.g.
#   ./run.sh /abs/path/to/seq_a /abs/path/to/seq_b
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
fi

for knn_weight in "${KNN_W[@]}" 
do
    for DATA_PATH in "${DATASETS[@]}" 
    do
        # strip trailing slash so basename is not empty
        DATA_PATH="${DATA_PATH%/}"

        if [ ! -d "$DATA_PATH" ]; then
            echo "===== SKIP (dataset path not found): $DATA_PATH ====="
            continue
        fi

        DATA_NAME=$(basename "$DATA_PATH")
        echo "===== Running $DATA_NAME ($DATA_PATH) ====="

        data_path="$DATA_PATH"
        exp_name="test/${DATA_NAME}_knn${knn_weight}_depth${DEPTH_W}_dcol${DCOL_W}"
        model_path="output/${exp_name}"

        # ---- TRAIN ----
        python train.py \
            -s "$data_path" \
            --port 6019 \
            --expname "$exp_name" \
            --depth_weight $DEPTH_W \
            --knn_weight $knn_weight \
            --dcol_weight $DCOL_W \
            --iterations $ITERS \
            --densify_from_iter 500000 \
            --pruning_from_iter 500000 \
            --densify_from_iter_fine 500000 \
            --pruning_from_iter_fine 500000 \
            --col_smooth_weight $COL_SMOOTH_W

        # ---- RENDER ----
        python render.py \
            -s "$data_path" \
            --model_path "$model_path" \
            --skip_video --skip_train --eval

        # ---- METRICS FOR EVERY N-TH TEST FRAME ----
        python metrics.py --model_path "$model_path"

        # ---- VISUALIZE LOOKAT CAMERAS ----
        python render_lookat_cameras.py \
            -s "$data_path" \
            --model_path "$model_path" \
            --skip_video --skip_train --eval

        # ---- CAMERA-TIMESTEP SHIFT ----
        python render_time_shift.py \
            -s "$data_path" \
            --model_path "$model_path" \
            --skip_video --skip_train --eval
    done
done
