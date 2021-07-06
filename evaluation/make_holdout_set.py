# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 19:22:30 2020

@author: stravsm
"""


import os
import sys
sys.path.append(os.environ['MSNOVELIST_BASE'])
import csv
import smiles_config as sc



import fp_management.fingerprinting as fpr
import fp_management.fingerprint_map as fpm
import fp_management.database as db
import random

import pickle
import pandas as pd
import numpy as np
import re

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from tqdm import tqdm

fp_map = fpm.FingerprintMap(sc.config["fp_map"])

import time


fp_map = fpm.FingerprintMap(sc.config["fp_map"])
fpr.Fingerprinter.init_instance(sc.config['fingerprinter_path'],
                                  sc.config['fingerprinter_threads'],
                                  capture = False,
                                  cache = sc.config['fingerprinter_cache'])
fingerprinter = fpr.Fingerprinter.get_instance()


top_n = 10


# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("make_holdout_set startup")

n = 128
data_eval_ = sc.config["db_path_eval"]
db_eval = db.FpDatabase.load_from_config(data_eval_)


eval_id = str(int(time.time()))


sc.config.setdefault('cv_fold', 0)
cv_fold = sc.config["cv_fold"]


for cv_fold in range(sc.config["cv_folds"]):
    data_eval = []
    
    evaluation_sets = ['casmi', 'sirius']
    for evaluation_set_ in evaluation_sets:
        fold_set = f"fold{cv_fold}"
        evaluation_set = f"fold{cv_fold}-{evaluation_set_}"
        data_eval.extend(db_eval.get_grp(evaluation_set))
    
    inchi_reserve = set([x["inchikey1"] for x in data_eval])
    
    complement = db_eval.get_grp(fold_set)
    complement_filtered = [x for x in complement if x["inchikey1"] not in inchi_reserve]
    
    ids = [x["Index"] for x in complement_filtered]
    db_eval.set_grp(f"fold{cv_fold}-holdout", pd.DataFrame({'id': ids}))

picklepath_out_ = db_eval.config['path']
picklepath_out_ = os.path.splitext(picklepath_out_)[0]
picklepath_out = f"{picklepath_out_}_holdout_{eval_id}.pkl"
db_eval.dump_pickle(picklepath_out)

