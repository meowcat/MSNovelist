#!/bin/bash


DB_STEP1=$(yq e '.db_step1' /target/log.yaml)
DB_STEP2=$(yq e '.db_step2' /target/log.yaml)

sqlite3 $DB_STEP1 << EOF
    create index if not exists folds ON compounds (grp);
EOF
