#!/bin/bash

source .env
DOCKER_ID=$(docker ps | grep msnovelist6 | cut -d' ' -f1)
echo DOCKER $DOCKER_ID

case $1 in
    ## the options below are for building, running and debugging the docker image
    aptitude) ## install aptitude on the running docker container
        docker exec -uroot $DOCKER_ID apt-get update
        docker exec -uroot $DOCKER_ID apt-get install -y aptitude
        ;;
    root) ## enter the running docker container as root
        docker exec -uroot -it $DOCKER_ID bash
        ;;   
    build) ## build the Docker container
        docker build -f Dockerfile.cuda . -t stravsm/msnovelist6
        ;;
    push) ## push the Docker container
        docker push stravsm/msnovelist6
        ;;
    run) ## run the Docker container with webui and mounts from .env
        docker run -d \
            -v $DATA_LOC:/sirius6_db \
            -v $PWD:/msnovelist \
            -v $DATA_LOC:/target \
            stravsm/msnovelist6 \
            webui.sh
            ;;
    kill) ## kill the running docker container
        docker kill $DOCKER_ID
        ;;
    
    ## the options below are for running on SLURM and singularity

    singularity-build) ## build singularity container
        # requires a stravsm/msnovelist6 container on the docker registry,
        # as singularity doesn't build Dockerfiles by itself
        SINGULARITY_CACHEDIR=$SCRATCH_PATH/singularity_cache singularity build \
            $SCRATCH_PATH/MSNovelist-image/msnovelist.sif docker://stravsm/msnovelist6$
        ;;
    tail-train) ## find currently running training job, dial in and follow
        JOBID=$(squeue -o "%i %j" | grep -F "run_train.sh" | cut -f1 -d' ')
        srun --interactive --jobid $JOBID --pty bash -c "$PWD/admin.sh tail-gpu"
        ;;
    tail-eval) ## find currently running training job, dial in and follow
        JOBID=$(squeue -o "%i %j" | grep -F "run_evaluation.sh" | cut -f1 -d' ')
        srun --interactive --jobid $JOBID --pty bash -c "$PWD/admin.sh tail-cpu"
        ;;
    tail-gpu) ## find currently running training process and follow
        PYPID=$(nvidia-smi --query-compute-apps="pid" --format=csv,noheader)
        tail -f /proc/$PYPID/fd/1
        ;;
    tail-cpu)
        PYPID=$(ps -ef | grep python | grep evaluation  --query-compute-apps="pid" --format=csv,noheader)

esac