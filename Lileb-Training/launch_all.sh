#!/bin/zsh
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#source activate torch;

RAY_MEM="$((300*1000*1000*1000))"
echo "Assigning $RAY_MEM bytes for this node"

if [ $SLURMD_NODENAME = $HEAD_NAME ]
then
    echo "I am head node: ${SLURMD_NODENAME}"
    echo "Starting head node ..."
    ray start --head --redis-port $HEAD_PORT --memory $RAY_MEM --temp-dir /tmp/ray_${USER}
    echo "Done"

    sleep 20
    EXP_CONF_FILE=configs/${DOMAIN}/test_large.yaml
    echo "Launching exp with config file: ${EXP_CONF_FILE}:"
    cmd="python run.py with $EXP_CONF_FILE experiment.redis_address=localhost:$HEAD_PORT datasets.task_gen.samples_per_class=\"[${N_SAMPLES}, 200, 200]\""
    echo ${cmd}
    eval ${cmd}
    echo "Job done, stopping head node."
    ray stop
else
    sleep 10
    echo "I am the worker ${SLURMD_NODENAME}, connecting to $HEAD_ADDR"
    ray start --redis-address $HEAD_ADDR --memory $RAY_MEM --block --temp-dir /tmp/ray_veniat
fi
