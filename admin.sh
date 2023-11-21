#!/bin/bash

source .env
DOCKER_ID=$(docker ps | grep msnovelist6 | cut -d' ' -f1)
echo DOCKER $DOCKER_ID

case $1 in
    aptitude)
        docker exec -uroot $DOCKER_ID apt-get update
        docker exec -uroot $DOCKER_ID apt-get install -y aptitude
        ;;
    root)
        docker exec -uroot -it $DOCKER_ID bash
        ;;   
    build)
        docker build -f Dockerfile.cuda . -t stravsm/msnovelist6
        ;;
    push)
        docker push stravsm/msnovelist6
        ;;
    run)
        docker run -d \
            -v $DATA_LOC:/sirius6_db \
            -v $PWD:/msnovelist \
            -v $DATA_LOC:/target \
            stravsm/msnovelist6 \
            webui.sh
            ;;
    kill)
        docker kill $DOCKER_ID
        ;;
    singularity-build)
        # requires a stravsm/msnovelist6 container on the docker registry,
        # as singularity doesn't build Dockerfiles by itself
        SINGULARITY_CACHEDIR=$SCRATCH_PATH/singularity_cache singularity build \
            $SCRATCH_PATH/MSNovelist-image/msnovelist.sif docker://stravsm/msnovelist6
esac