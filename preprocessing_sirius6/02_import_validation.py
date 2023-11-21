import h5py
from tqdm import tqdm
import sqlite3
#import sqlite_utils as su

import sys
sys.path.append('/msnovelist')

import fp_management.database as db
import itertools
import uuid
import fp_management.fingerprinting as fpr
import fp_management.fingerprint_map as fpm

import smiles_config as sc

db_crossval = "/sirius6_db/canopus_crossval.hdf5"
db_train = "/sirius6_db/canopus_database.hdf5"

# db_old = "/sirius6_db/combined_0824_v44.db"
db_uuid = uuid.uuid4()
db_new = f"/target/sirius6-{db_uuid}.db"

h5_crossval = h5py.File(db_crossval, mode='r')
h5_train = h5py.File(db_train, mode='r')

PROCESSING_BLOCK_SIZE=40000

fp_map = fpm.FingerprintMap(sc.config['fp_map'])

fpr.Fingerprinter.init_instance(
    sc.config['normalizer_path'],
    sc.config['sirius_path'],
    fp_map,
    sc.config['fingerprinter_threads'],
    capture = True)
fingerprinter = fpr.Fingerprinter.get_instance()

def try_fp_item(smiles_generic, smiles_canonical, fp_true, fp_predicted):
    try:
        item = db.FpItem.fromSiriusFp(
            smiles_generic = smiles_generic,
            smiles_canonical = smiles_canonical,
            fp = fp_true,
            source = "dataset",
            grp = "fold0",
            b64 = False
        )
        item.fp_degraded = fp_predicted
    except:
        item = None
    return item

def db_item_block(block):
    smiles = [s_ for s_, fp_true_, fp_predicted_  in block]
    fp_true = [fp_true_.astype('uint8') for s_, fp_true_, fp_predicted_  in block]
    fp_predicted = [fp_predicted_ for s_, fp_true_, fp_predicted_  in block]

    smiles_proc = fingerprinter.process(smiles, calc_fingerprint=False)
    item = zip(smiles_proc, fp_true, fp_predicted)
    fp_items_ = [try_fp_item(s['smiles_generic'], s['smiles_canonical'], fp_true, fp_predicted)
                 for s, fp_true, fp_predicted in item ]
    fp_items = [x for x in fp_items_ if x is not None]
    return fp_items

data_in = zip(
     h5_crossval["smiles"],
     h5_crossval["csiTruth"],
     h5_crossval["csiPredicted"],
)


def take(n, iterable): 
    return list(itertools.islice(iterable, n))

print(f"database: {db_new}")

fp_db = db.FpDatabase.load_from_config(db_new)
block = take(PROCESSING_BLOCK_SIZE, data_in)
processed_blocks = 0

while len(block) > 0:
    print(f"Processing block {processed_blocks}")
    data_proc = db_item_block(block)
    fp_db.insert_fp_multiple(data_proc)
    #print(f"last inserted id: {inserted_id}")
    processed_blocks = processed_blocks + 1
    block = take(PROCESSING_BLOCK_SIZE, data_in)

print(f"database: {db_new} written")

with open('/target/log.yaml', "r+") as f:
    f.write(f'db_step2: {db_new}')
