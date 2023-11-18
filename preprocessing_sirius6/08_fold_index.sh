#!/bin/bash


DB_STEP1=$(yq e '.db_step1' /target/log.yaml)
DB_STEP2=$(yq e '.db_step2' /target/log.yaml)

sqlite3 $DB_STEP1 << EOF
    CREATE INDEX IF NOT EXISTS idx_folds ON compounds (grp);
    CREATE INDEX IF NOT EXISTS idx_inchikeys ON compounds (inchikey1);
EOF


sqlite3 $DB_STEP2 << EOF
    CREATE INDEX IF NOT EXISTS idx_folds ON compounds (grp);
    CREATE INDEX IF NOT EXISTS idx_inchikeys ON compounds (inchikey1);
EOF