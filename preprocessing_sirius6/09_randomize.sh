#!/bin/bash

DB_STEP1=$(yq e '.db_step1' /target/log.yaml)
DB_STEP2=$(yq e '.db_step2' /target/log.yaml)

sqlite3 $DB_STEP1 << EOF
    UPDATE compounds
    SET perm_order = random();
EOF

sqlite3 $DB_STEP2 << EOF
    UPDATE compounds
    SET perm_order = random();
EOF
