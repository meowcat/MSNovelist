#!/bin/bash

cd /msnovelist
export COMPUTERNAME=DOCKER-LIGHT
export MSNOVELIST_BASE=/msnovelist

python evaluation.py \
	-c /target/evaluation.yaml /data/evaluation/$SLURM_JOB_ID.yaml
