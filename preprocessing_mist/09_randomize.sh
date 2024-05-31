#!/bin/bash


DB_STEP1=$(yq e '.db_step1' /home/stravsmi/msmsgym/MSNovelist-private/preprocessing_mist/log.yaml)
DB_STEP2=$(yq e '.db_step2' /home/stravsmi/msmsgym/MSNovelist-private/preprocessing_mist/log.yaml)

sqlite3 $DB_STEP1 << EOF
    UPDATE compounds
    SET perm_order = random();
EOF

sqlite3 $DB_STEP2 << EOF
    UPDATE compounds
    SET perm_order = random();
EOF
