#!/bin/bash

source .env
LOCAL_TARGET=$DATA_LOC

mkdir -p $LOCAL_TARGET

#aws s3 cp --recursive $S3_SOURCE $LOCAL_TARGET

ln -s $LOCAL_TARGET $LOCAL_TARGET/training
mkdir -p $LOCAL_TARGET/data
mkdir -p $LOCAL_TARGET/data/tensorboard
mkdir -p $LOCAL_TARGET/data/weights
mkdir -p $LOCAL_TARGET/data/results

