#!/bin/bash

#SBATCH --job-name=lileb
## %j is the job id, %u is the user id
#SBATCH --output=/checkpoint/%u/jobs/sample-%j.out
#SBATCH --error=/checkpoint/%u/jobs/sample-%j.err

#SBATCH --partition=uninterrupted
#SBATCH --nodes=2

#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task 80
#SBATCH --mem 300G
#SBATCH --time 4320


if [ -z "$1" ]
then
    echo "Missing mandatory arg1 (folder of the main config)."
    exit
else
    echo "Using config from $1."
    export DOMAIN=$1
fi

if [ -z "$2" ]
then
    echo "Missing mandatory arg2 (N train samples)."
    exit
else
    echo "Using $2 train samples."
    export N_SAMPLES=$2
fi

#export HEAD_NAME="${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}"
NODES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
echo "Running on nodes ${NODES[@]}"
export HEAD_NAME=${NODES[@]:0:1} #Get the first node
#export HEAD_NAME="${SLURM_NODELIST:0:12}"
export HEAD_PORT=${3:-6385}
export HEAD_ADDR="$HEAD_NAME:$HEAD_PORT"
echo "Head node: $HEAD_ADDR"

srun --label launch_all.sh
echo "Job finished"