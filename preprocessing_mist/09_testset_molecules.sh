#!/bin/bash


DB_STEP1=$(yq e '.db_step1' /home/stravsmi/msmsgym/MSNovelist-private/preprocessing_mist/log.yaml)
DB_STEP2=$(yq e '.db_step2' /home/stravsmi/msmsgym/MSNovelist-private/preprocessing_mist/log.yaml)

DATE_NOW=$(date +%s)
echo $DATE_NOW

# cp $DB_STEP1 $DB_STEP1.$DATE_NOW
# cp $DB_STEP2 $DB_STEP2.$DATE_NOW

sqlite3 $DB_STEP1 << EOF

    UPDATE inchikeys
        SET grp = 'train';

    UPDATE inchikeys
        SET grp = 'test'
        WHERE (id % 10) = 0;

    UPDATE compounds
        SET grp = (SELECT grp FROM inchikeys where compounds.inchikey1 = inchikeys.inchikey1)
        WHERE grp = 'train';

EOF
