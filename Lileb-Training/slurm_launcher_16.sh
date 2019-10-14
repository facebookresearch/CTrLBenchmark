#!/bin/bash

FOLDERS=( x y )

NUM_SAMPLES=( 1 3 5 10 50 100 500 1000 )

for folder in ${FOLDERS[@]}
do
    i=0
    job_ids=()
    for N in ${NUM_SAMPLES[@]}
    do
        command="echo $i"
        echo $command
        job_id=$($command)
        echo "New job: $job_id"
        job_ids+=($job_id)
        i=$(($i+1))
    done
    echo "Job ids for folder $folder: ${job_ids[@]}"
done
