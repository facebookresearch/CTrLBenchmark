#!/bin/bash

FOLDERS=( xyz )

NUM_SAMPLES=(1 3 5 10 100 300 500 1000 3000 )

N_RUNS=1

for folder in ${FOLDERS[@]}
do
    job_ids=()
    for N_SAMPLE in ${NUM_SAMPLES[@]}
    do
        command="sbatch --job-name $folder-$N_SAMPLE --parsable slurm_launcher_8.sh $folder $N_SAMPLE"
        echo ${command}
        for N in $(seq 1 ${N_RUNS})
        do
            job_id=$($command)
            echo "New job: $job_id"
            job_ids+=(${job_id})
        done
    done
    echo "Job ids for folder $folder: ${job_ids[@]}"
done
