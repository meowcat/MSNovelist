#!/bin/bash

cd /msnovelist
export COMPUTERNAME=DOCKER-LIGHT
export MSNOVELIST_BASE=/msnovelist

python training_subclass.py -c /sirius6_db/config.yaml
