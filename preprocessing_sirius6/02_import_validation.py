import h5py
from tqdm import tqdm
import sqlite3
#import sqlite_utils as su

import sys
sys.path.append('/msnovelist')

import fp_management.database as db
import itertools
import uuid
import fp_management.fingerprinting as fp
import smiles_config as sc

db_crossval = "/sirius6_db/canopus_crossval.hdf5"
db_train = "/sirius6_db/canopus_database.hdf5"

# db_old = "/sirius6_db/combined_0824_v44.db"
db_uuid = uuid.uuid4()
db_new = f"/target/sirius6-{db_uuid}.db"

h5_crossval = h5py.File(db_crossval, mode='r')
h5_train = h5py.File(db_train, mode='r')

PROCESSING_BLOCK_SIZE=40000


# inchikeys_crossval = h5_crossval["inchikeys"]
# inchikeys_train = h5_train["inchikey"]
#inchikeys_crossval_set = set(inchikeys_crossval)
#inchikeys_train_set = set(inchikeys_train)

#inchikeys_exclude = [k in inchikeys_crossval_set for k in tqdm(inchikeys_train)]

#con = sqlite3.connect(db_old)
#con_su = su.Database(db_old)


fingerprinter = fp.Fingerprinter(sc.config['fingerprinter_path'])

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
    smiles = [i[0] for i in block]
    fp_true = [i[1] for i in block]
    fp_predicted =  [i[2] for i in block]
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
while len(block) > 0:
    data_proc = db_item_block(block)
    fp_db.insert_fp_multiple(data_proc)
    #print(f"last inserted id: {inserted_id}")
    block = take(PROCESSING_BLOCK_SIZE, data_in)

print(f"database: {db_new} written")

with open('/target/log.yaml', "r+") as f:
    f.write(f'db_step2: {db_new}')
# data_proc = [db_item(smiles, fp) for smiles, fp in take(100, data_in)]
# fp_db.insert_fp_multiple(data_proc)






# def fp_iter(f):
#     with open(f) as h:
#         fp_reader = csv.reader(h, delimiter='\t')
#         for row in fp_reader:
#             try:
#                 fp = db.FpItem.fromSiriusFp(smiles_generic = row[0],
#                                             smiles_canonical = row[1],
#                                             fp = row[2])
#                 yield(fp)
#             except:
#                 pass


# INSERT_BLOCK = 5000

# import itertools
# def take(n, iterable): return list(itertools.islice(iterable, n))

# MOL_DB_PATH='/sirius6-db/sirius6.db'
# start = time.time()
# fp_db = db.FpDatabase.load_from_config(MOL_DB_PATH)
# fp_cnt = fp_count(MOL_FP_PATH)
# fp_it = fp_iter(MOL_FP_PATH)

# n_blocks = fp_cnt // INSERT_BLOCK
# logger.info(f"Inserting into DB, block size {INSERT_BLOCK}, block count {n_blocks}")

# insert_done = False
# for i in tqdm(range(n_blocks + 2)):
#         fp_block = take(INSERT_BLOCK, fp_it)
#         #print("Inserting...", flush=True)
#         fp_db.insert_fp_multiple(fp_block)
#         if len(fp_block) == 0:
#             insert_done = True
            
# logger.info(f"Insert finished, result: {insert_done}")

# dtime = time.time() - start
# logger.info(f"Done in {dtime} seconds")
# fp_db.close()


# schema_cpds = '''
# CREATE TABLE compounds (
#     id INTEGER PRIMARY KEY,
#     fingerprint BLOB, 
#     fingerprint_degraded BLOB, 
#     smiles_generic CHAR(128), 
#     smiles_canonical CHAR(128),
#     inchikey CHAR(27), 
#     inchikey1 CHAR(14), 
#     mol BLOB, 
#     mf BLOB, 
#     mf_text CHAR(128), 
#     source CHAR(128), 
#     grp CHAR(128), 
#     perm_order INT )
# '''