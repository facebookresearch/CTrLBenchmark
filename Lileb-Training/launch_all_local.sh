#!/bin/zsh
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#source activate torch;
HEAD_PORT=7777
RAY_MEM="$((150*1000*1000*1000))"
echo "Assigning $RAY_MEM bytes for this node "

echo "Starting head node ..."
ray start --head --redis-port $HEAD_PORT --memory $RAY_MEM --temp-dir /tmp/ray_${USER}
echo "Done"

DOMAIN=cifarx
N_SAMPLES=10
EXP_CONF_FILE=configs/${DOMAIN}/test_large.yaml
echo "Launching exp with config file: ${EXP_CONF_FILE}:"
cmd="python run.py with $EXP_CONF_FILE experiment.redis_address=localhost:$HEAD_PORT datasets.task_gen.samples_per_class=\"[${N_SAMPLES}, 200, 200]\""
echo ${cmd}
eval ${cmd}
echo "Job done, stopping head node."
ray stop
