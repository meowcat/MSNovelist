#!/bin/bash


DB_STEP1=$(yq e '.db_step1' /target/log.yaml)
DB_STEP2=$(yq e '.db_step2' /target/log.yaml)

sqlite3 $DB_STEP1 << EOF
    -- select inchikey, grp, id, 'fold' || (id % 10) FROM compounds LIMIT 300;
    update compounds set grp = 'fold' || (id % 10) WHERE grp <> 'invalid';
    --select * from compounds WHERE grp = 'invalid' LIMIT 10;
EOF
