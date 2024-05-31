#!/bin/bash


DB_STEP1=$(yq e '.db_step1' /home/stravsmi/msmsgym/MSNovelist-private/preprocessing_mist/log.yaml)
DB_STEP2=$(yq e '.db_step2' /home/stravsmi/msmsgym/MSNovelist-private/preprocessing_mist/log.yaml)

sqlite3 $DB_STEP1 << EOF
    CREATE INDEX IF NOT EXISTS idx_folds ON compounds (grp);
    CREATE INDEX IF NOT EXISTS idx_inchikeys ON compounds (inchikey1);
EOF


sqlite3 $DB_STEP2 << EOF
    CREATE INDEX IF NOT EXISTS idx_folds ON compounds (grp);
    CREATE INDEX IF NOT EXISTS idx_inchikeys ON compounds (inchikey1);
EOF