#!/bin/bash


DB_STEP1=$(yq e '.db_step1' /home/stravsmi/msmsgym/MSNovelist-private/preprocessing_mist/log.yaml)
DB_STEP2=$(yq e '.db_step2' /home/stravsmi/msmsgym/MSNovelist-private/preprocessing_mist/log.yaml)

DATE_NOW=$(date +%s)
echo $DATE_NOW

cp $DB_STEP1 $DB_STEP1.$DATE_NOW
cp $DB_STEP2 $DB_STEP2.$DATE_NOW

sqlite3 $DB_STEP1 << EOF
    -- select inchikey, grp, id, 'fold' || (id % 10) FROM compounds LIMIT 300;
    -- update compounds set grp = 'fold' || (id % 10) WHERE grp <> 'invalid';
    -- select * from compounds WHERE grp = 'invalid' LIMIT 10;
    drop table if exists inchikeys;
    drop index if exists inchikeys;
    create table inchikeys (inchikey1 CHAR(27), grp CHAR(128), id INTEGER);
    insert into inchikeys (inchikey1)
        select distinct inchikey1 from compounds;
    
    UPDATE inchikeys
        SET id = (
                SELECT MIN(id) FROM compounds
	            WHERE compounds.inchikey1 = inchikeys.inchikey1 
                );
    
    update inchikeys set grp = 'fold' || (id % 10) WHERE grp <> 'invalid';
    update compounds set grp = (
        SELECT grp from inchikeys 
        where inchikeys.inchikey1 = compounds.inchikey1
    )
    WHERE grp <> "invalid";

EOF

sqlite3 $DB_STEP2 << EOF

    drop table if exists inchikeys;
    drop index if exists inchikeys;
    create table inchikeys (inchikey1 CHAR(27), grp CHAR(128), id INTEGER);
    insert into inchikeys (inchikey1)
        select distinct inchikey1 from compounds;

    CREATE INDEX IF NOT EXISTS
	    index_compounds_inchikey1 on compounds (inchikey1);
    CREATE INDEX IF NOT EXISTS
	    index_inchikeys_inchikey1 on inchikeys (inchikey1);

    UPDATE inchikeys
        SET id = (
                SELECT MIN(id) FROM compounds
	            WHERE compounds.inchikey1 = inchikeys.inchikey1 
                );

    update inchikeys set grp = 'fold' || (id % 10) WHERE grp <> 'invalid';
    update compounds set grp = (
        SELECT grp from inchikeys 
        where inchikeys.inchikey1 = compounds.inchikey1
    )
    WHERE grp <> "invalid";

EOF


