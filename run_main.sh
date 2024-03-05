#!/bin/bash

# 1. Hyperparameters
MODEL_NAME="vit"
DATASET="c100"
BATCH_SIZE=128
LR=1e-3
HEAD=8
NUM_LAYERS=8
HIDDEN=384
MLP_HIDDEN=1536
NUM_WORKERS=4
DATA_PATH="/mnt/ssd2/dataset/CIFAR100"
LOGGER="wandb"

# 2. Proj & Exp Info
PROJ_NAME="lignex1_vit_cifar100"
EXP_NAME="LOCAL_bs${BATCH_SIZE}_${MODEL_NAME}_baseline"

# 3. run
CUDA_VISIBLE_DEVICES=0 python main.py   --model-name ${MODEL_NAME} \
                                        --dataset ${DATASET} \
                                        --data_path ${DATA_PATH} \
                                        --num-classes 100 \
                                        --patch 8 \
                                        --batch-size ${BATCH_SIZE} \
                                        --eval-batch-size 1024 \
                                        --lr ${LR} \
                                        --max-epochs 200 \
                                        --dropout 0 \
                                        --head ${HEAD} \
                                        --num-layers ${NUM_LAYERS} \
                                        --hidden ${HIDDEN} \
                                        --mlp-hidden ${MLP_HIDDEN} \
                                        --project_name ${PROJ_NAME} \
                                        --experiment-memo ${EXP_NAME} \
                                        --num_workers ${NUM_WORKERS} \
                                        --logger ${LOGGER}
                                        # --api-key upJRJyzbQWeOazI7HlvvikhpG

echo "finished one experiment"