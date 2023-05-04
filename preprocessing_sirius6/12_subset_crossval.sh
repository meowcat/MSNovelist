
#!/bin/bash

DB_STEP2=/target/sirius6-crossval-folds.db
DBNEW_UUID=$(python -c "import uuid; print(uuid.uuid4())")
COMPOUNDS_LIMIT=30000


sqlite3 $DB_STEP2 << EOF
    ATTACH DATABASE '/target/sirius6-$DBNEW_UUID.db' AS target;
    CREATE TABLE target.compounds AS
        SELECT * FROM compounds ORDER BY perm_order LIMIT $COMPOUNDS_LIMIT;
    CREATE INDEX IF NOT EXISTS idx_folds ON compounds (grp);
    CREATE INDEX IF NOT EXISTS idx_inchikeys ON compounds (inchikey1);

EOF

echo "created /target/sirius6-$DBNEW_UUID.db"
echo "db_step2: /target/sirius6-$DBNEW_UUID.db" >> /target/log-subset.yaml

BACKUP_TARGET=/sirius6_db/$(date +%s)
mkdir -p $BACKUP_TARGET
cp /target/sirius6-$DBNEW_UUID.db $BACKUP_TARGET