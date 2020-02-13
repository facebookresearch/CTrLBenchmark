#!/bin/zsh
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#source activate torch;
export MASTER_ADDR="learnfair125"
echo "Master is ${MASTER_ADDR}"

echo "I am a worker: ${SLURMD_NODENAME}"
echo "Connecting to ${MASTER_ADDR}"
#python -c "import ray; ray.init(redis_address='$MASTER_ADDR:6385'); print(ray.available_resources())"
ray start --redis-address $MASTER_ADDR:6385 --block

#echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES $SLURM_LOCALID
#echo $SLURM_PROCID
#echo $SLURM_NODELIST
#hostname
echo "okok ?"
