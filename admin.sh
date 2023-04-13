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
            -v $DATA_PATH:/sirius6_db \
            -v $PWD:/msnovelist \
            -v $DB_PATH:/target \
            stravsm/msnovelist6 \
            webui.sh
            ;;
    kill)
        docker kill $DOCKER_ID
        ;;
esac