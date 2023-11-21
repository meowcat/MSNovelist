#!/bin/bash

cd /msnovelist
export COMPUTERNAME=DOCKER-LIGHT
export MSNOVELIST_BASE=/msnovelist

python train.py \
	-c /data/train/$SLURM_JOB_ID.yaml
