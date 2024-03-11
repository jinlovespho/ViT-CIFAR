#!/bin/bash

# 0. Paths
DATA_PATH="/mnt/ssd2/dataset/CIFAR100"
SAVE_DIR="/mnt/ssd2/log/vit_cifar100"

# 1. Hyperparameters
MODEL_NAME="vit"        # ['vit', 'vit_parallel', 'vit_gyu']
DATASET="c100"
MAX_EPOCHS=200
BATCH_SIZE=128
LR=1e-3
NUM_LAYERS=24
HIDDEN=1024
MLP_HIDDEN=4096
HEAD=16
NUM_WORKERS=4

# 2. Logger & Project & Experiment Info
LOGGER="wandb"      # [ 'wandb', 'None' ]
PROJ_NAME="lignex1_vit_cifar100"
EXP_NAME="LOCAL_bs${BATCH_SIZE}_${MODEL_NAME}-Large_baseline"

# 3. run
CUDA_VISIBLE_DEVICES=0 python main.py   --model-name ${MODEL_NAME} \
                                        --dataset ${DATASET} \
                                        --data_path ${DATA_PATH} \
                                        --save_dir ${SAVE_DIR} \
                                        --num-classes 100 \
                                        --patch 8 \
                                        --batch-size ${BATCH_SIZE} \
                                        --eval-batch-size 1024 \
                                        --lr ${LR} \
                                        --max_epochs ${MAX_EPOCHS} \
                                        --dropout 0 \
                                        --head ${HEAD} \
                                        --num-layers ${NUM_LAYERS} \
                                        --hidden ${HIDDEN} \
                                        --mlp-hidden ${MLP_HIDDEN} \
                                        --project_name ${PROJ_NAME} \
                                        --experiment_memo ${EXP_NAME} \
                                        --num_workers ${NUM_WORKERS} \
                                        --logger ${LOGGER} \
                                        --api-key upJRJyzbQWeOazI7HlvvikhpG

echo "finished one experiment"