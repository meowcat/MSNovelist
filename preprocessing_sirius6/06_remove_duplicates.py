import sqlite3
from iteration_utilities import duplicates
import yaml
from tqdm import tqdm

config_ = "/target/log.yaml"
with open(config_, 'r') as f:
    config = yaml.safe_load(f)

db_train = config["db_step1"]


con = sqlite3.connect(db_train)
with con:
    cur = con.cursor()

    q = "CREATE INDEX IF NOT EXISTS inchikeys ON compounds (inchikey1)"
    res = cur.execute(q)

    q = "SELECT inchikey1 FROM compounds"
    res = cur.execute(q)
    keys = res.fetchall()
    keys_keys = [x[0] for x in keys]
    keys_set = set(keys_keys)

    dups = list(duplicates(keys_keys))
    dups_killed = {}
    q = "UPDATE compounds SET grp = 'invalid' WHERE inchikey1 = ?"
    for dup in tqdm(dups):
        res = cur.execute(q, [dup])
        cnt = cur.rowcount
        dups_killed[dup] = cnt

