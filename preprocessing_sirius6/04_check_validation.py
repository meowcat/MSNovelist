

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

db_train = config["db_step2"]
fp_db = db.FpDatabase.load_from_config(db_train)
#fp_db.randomize()


q = f"SELECT id, smiles_canonical FROM compounds LIMIT {PROCESSING_BLOCK_MAX_COUNT}"
cur = fp_db._db_con.cursor()
res = cur.execute(q)
#data = res.fetchone()
id_invalid = []
for data in tqdm(res):
    is_ok = pp.substance_OK(data["smiles_canonical"])
    if is_ok != 0:
        id_invalid.append(data["id"])


invalid_file = db_train + ".invalid_ids"
with open(invalid_file, "w") as f:
    f.writelines([str(x) + "\n" for x in id_invalid])

print(f"Invalid entries: {len(id_invalid)}")



