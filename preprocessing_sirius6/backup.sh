#!/bin/bash

DB_STEP1=$(yq e '.db_step1' /target/log.yaml)
DB_STEP2=$(yq e '.db_step2' /target/log.yaml)
BACKUP_TARGET=/sirius6_db/$(date +%s)

mkdir -p $BACKUP_TARGET
echo copying $DB_STEP1 to $BACKUP_TARGET
cp $DB_STEP1 $BACKUP_TARGET
echo copying $DB_STEP2 to $BACKUP_TARGET
cp $DB_STEP2 $BACKUP_TARGET

