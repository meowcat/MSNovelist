#!/bin/bash


DB_STEP1=$(yq e '.db_step1' /home/stravsmi/msmsgym/MSNovelist-private/preprocessing_mist/log.yaml)
DB_STEP2=$(yq e '.db_step2' /home/stravsmi/msmsgym/MSNovelist-private/preprocessing_mist/log.yaml)
TARGET_DIR=/home/stravsmi/msmsgym/20240515_mistnovelist

cp $DB_STEP1 $TARGET_DIR
cp $DB_STEP2 $TARGET_DIR

