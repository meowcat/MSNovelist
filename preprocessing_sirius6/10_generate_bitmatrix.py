
import sys
sys.path.append('/msnovelist')

import h5py
import numpy as np
import humanize
import pandas as pd
from enum import IntEnum
import pickle
import sqlite3
import fp_management.fingerprinting as fpr
from tqdm import tqdm
import pathlib
import yaml
h5_crossval_ = "/sirius6_db/canopus_crossval.hdf5"
h5_crossval = h5py.File(h5_crossval_, mode='r')
csi_pred = h5_crossval["csiPredicted"]
csi_truth = h5_crossval["csiTruth"]
inchikeys = h5_crossval["inchikeys"]


config_ = "/target/log.yaml"
with open(config_, 'r') as f:
    config = yaml.safe_load(f)

db_crossval_ = config["db_step2"]
db_crossval = sqlite3.connect(db_crossval_)

def dict_factory(cursor, row):
    fields = [column[0] for column in cursor.description]
    return {key: value for key, value in zip(fields, row)}

db_crossval.row_factory = dict_factory

db_crossval_cur = db_crossval.cursor()
# include everything, also the invalid entries,
# as we later need to pick them by row id!
cpds_query = db_crossval_cur.execute("SELECT * FROM Compounds")
def read_fp(cpd_data):
    fp_truth = np.frombuffer(cpd_data["fingerprint"], dtype = "uint8")
    fp_pred = np.frombuffer(cpd_data["fingerprint_degraded"], dtype = "float32")
    fp_truth_int = np.rint(fp_truth).astype("int16")
    fp_pred_int = np.rint(fp_pred).astype("int16")
    return (fp_truth_int, fp_pred_int)

fp_data = [read_fp(cpd) for cpd in tqdm(cpds_query)]



csi_truth_int = np.stack([fp_truth for fp_truth, fp_pred in fp_data])
csi_pred_int = np.stack([fp_pred for fp_truth, fp_pred in fp_data])

csi_TP = csi_pred_int & csi_truth_int
csi_FP = csi_pred_int & np.logical_not(csi_truth_int)
csi_TN = np.logical_not(csi_pred_int) & np.logical_not(csi_truth_int)
csi_FN = np.logical_not(csi_pred_int) & csi_truth_int

class Bitmask(IntEnum):
    PREDICTED = 1
    TRUTH = 2
    TP = 4
    FP = 8
    TN = 16
    FN = 32


csi_multiplex = \
    Bitmask.PREDICTED * csi_pred_int + \
    Bitmask.TRUTH * csi_truth_int + \
    Bitmask.TP *  csi_TP + \
    Bitmask.FP * csi_FP + \
    Bitmask.TN * csi_TN + \
    Bitmask.FN * csi_FN
# create a bit-encoded matrix, where 
# * bit1 is predicted T/F 
# * bit2 is true T/F
humanize.naturalsize(sys.getsizeof(csi_multiplex))
csi_multiplex_min = csi_multiplex.astype('int8')
humanize.naturalsize(sys.getsizeof(csi_multiplex_min))

out_path = pathlib.Path(db_crossval_).with_suffix('.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(csi_multiplex_min, f)




