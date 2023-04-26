#!/bin/bash

cd /msnovelist
export COMPUTERNAME=DOCKER-LIGHT
export MSNOVELIST_BASE=/msnovelist

python training_subclass.py \
	-c /target/config.yaml /target/train/$SLURM_JOB_ID.yaml
