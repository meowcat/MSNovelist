import h5py
from tqdm import tqdm
import sqlite3
#import sqlite_utils as su

import sys
sys.path.append('/msnovelist')
import os

import fp_management.database as db
import itertools
import uuid
import fp_management.fingerprinting as fpr
import fp_management.fingerprint_map as fpm
import smiles_config as sc
import pickle
db_crossval = "/sirius6_db/canopus_crossval.hdf5"
db_train = "/sirius6_db/canopus_database.hdf5"

# db_old = "/sirius6_db/combined_0824_v44.db"
db_uuid = uuid.uuid4()
db_new = f"/target/sirius6-{db_uuid}.db"

h5_crossval = h5py.File(db_crossval, mode='r')
h5_train = h5py.File(db_train, mode='r')

PROCESSING_BLOCK_SIZE=40000
PROCESSING_BLOCK_MAX_COUNT=9999999999

selected_ = "/sirius6_db/smiles_sampled.pkl"
with open(selected_, 'rb') as f:
    selected = pickle.load(f)
selected.sort()

# inchikeys_crossval = h5_crossval["inchikeys"]
# inchikeys_train = h5_train["inchikey"]
#inchikeys_crossval_set = set(inchikeys_crossval)
#inchikeys_train_set = set(inchikeys_train)

#inchikeys_exclude = [k in inchikeys_crossval_set for k in tqdm(inchikeys_train)]

#con = sqlite3.connect(db_old)
#con_su = su.Database(db_old)

fp_map = fpm.FingerprintMap(sc.config['fp_map'])

fpr.Fingerprinter.init_instance(
    sc.config['normalizer_path'],
    sc.config['sirius_path'],
    fp_map,
    sc.config['fingerprinter_threads'],
    capture = True)
fingerprinter = fpr.Fingerprinter.get_instance()

def try_fp_item(smiles_generic, smiles_canonical, fp):
    try:
        item = db.FpItem.fromSiriusFp(
            smiles_generic = smiles_generic,
            smiles_canonical = smiles_canonical,
            fp = fp,
            source = "dataset",
            grp = "fold0",
            b64 = False
        )
    except:
        item = None
    return item

def db_item_block(block):
    smiles = [s_ for s_, fp_, sel in block if sel]
    fp = [fp_ for s_, fp_, sel in block if sel]
    smiles_proc = fingerprinter.process(smiles, calc_fingerprint=False)
    item = zip(smiles_proc, fp)
    fp_items_ = [try_fp_item(s['smiles_generic'], s['smiles_canonical'], fp)
                 for s, fp in item ]
    fp_items = [x for x in fp_items_ if x is not None]
    return fp_items


data_selected = [False] * len(h5_train["smiles"])
for i in selected:
    data_selected[i] = True
data_in = zip(
     h5_train["smiles"],
     h5_train["csiTruth"],
     data_selected
)


def take(n, iterable): 
    return list(itertools.islice(iterable, n))

print(f"database: {db_new}")

fp_db = db.FpDatabase.load_from_config(db_new)
block = take(PROCESSING_BLOCK_SIZE, data_in)

# smiles = [s_ for s_, fp_, sel in block if sel]
# smiles_proc = fingerprinter.process(smiles, calc_fingerprint=False)

processed_blocks = 0
while (len(block) > 0) and (processed_blocks < PROCESSING_BLOCK_MAX_COUNT):
    print(f"Processing block {processed_blocks}")
    selected_elements = sum(item[2] for item in block)
    data_proc = db_item_block(block)
    print(f"Loaded {len(block)} elements, "
          f"selected {selected_elements} elements, "
          f"successfully processed {len(data_proc)} elements")
    fp_db.insert_fp_multiple(data_proc)
    #print(f"last inserted id: {inserted_id}")
    block = take(PROCESSING_BLOCK_SIZE, data_in)
    processed_blocks = processed_blocks + 1


print(f"database: {db_new} written")

with open('/target/log.yaml', 'a+') as f:
    f.write(f'db_step1: {db_new}')

