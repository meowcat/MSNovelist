
import sys
sys.path.append('/msnovelist')

import infrastructure.preprocessing as pp
import sqlite3
import yaml
import fp_management.fp_database as db
import fp_management.fp_database_sqlite
from tqdm import tqdm

PROCESSING_BLOCK_MAX_COUNT=9999999999
#PROCESSING_BLOCK_MAX_COUNT=100000


config_ = "/target/log.yaml"
with open(config_, 'r') as f:
    config = yaml.safe_load(f)

databases = [config["db_step1"], config["db_step2"]]

for fp_db_ in databases:

    fp_db = db.FpDatabase.load_from_config(fp_db_)
    fp_db_invalid_ = fp_db_ + ".invalid_ids"

    with fp_db._db_con as con, open(fp_db_invalid_, 'r') as f:
        
        fp_db_invalid = [x.strip('\n') for x in f]

        q = "UPDATE compounds SET grp = 'invalid' WHERE id = ?"
        cur = con.cursor()

        for id in tqdm(fp_db_invalid):
            cur.execute(q, [id])
            #print(f"{fp_db_}: set invalid id {id}, rows: {cur.rowcount}")
    


