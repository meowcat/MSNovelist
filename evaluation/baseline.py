# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:38:27 2020

@author: stravsm
"""

# =============================================================================
# Produce baseline results for evaluation.
# Baseline 1: 5 randomly drawn structures from the training set with same MF,
# Baseline 2: all (i.e. including the best) structures from the training set
# with the same MF, assigned a random score
# (which should, without rescoring, behave like 5 random picks)
# =============================================================================

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

random_seed = sc.config['random_seed_global']
if random_seed != '':
    random.seed(random_seed)
    np.random.seed(random_seed)
    # This never uses any TF randomness, so no need to initialize that

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from tqdm import tqdm

fp_map = fpm.FingerprintMap(sc.config["fp_map"])

import time


# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("evaluation_baseline startup")


eval_id = str(int(time.time()))
pickle_id = eval_id
if sc.config['eval_id'] != '':
    eval_id = sc.config['eval_id']
if sc.config['eval_counter'] != '':
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
picklepath_all = sc.config["eval_folder"] + "baseline_all_" + pickle_id + ".pkl"
picklepath_rand5 = sc.config["eval_folder"] + "baseline_rand5_" + pickle_id + ".pkl"
picklepath_rand1 = sc.config["eval_folder"] + "baseline_rand1_" + pickle_id + ".pkl"


# Get the query set, to find the corresponding MF candidates
n = sc.config["eval_n"]
n_total = sc.config["eval_n_total"]
n_total_ = n_total // n * n
data_eval_ = sc.config["db_path_eval"]

logger.info("Initializing fingerprinter")

fpr.Fingerprinter.init_instance(sc.config['fingerprinter_path'],
                                  sc.config['fingerprinter_threads'],
                                  capture = True)
fingerprinter = fpr.Fingerprinter.get_instance()


logger.info("Loading training set and challenge set")

db_eval = db.FpDatabase.load_from_config(data_eval_)
sc.config.setdefault('cv_fold', 0)
cv_fold = sc.config["cv_fold"]
evaluation_set_ = sc.config['evaluation_set']
evaluation_set = f"fold{cv_fold}-{evaluation_set_}"
training_set = f"fold[^{cv_fold}]"

dataset_val = db_eval.get_grp(evaluation_set)
if n_total != -1:
    dataset_val = dataset_val[:n_total]
else:
    n_total = len(dataset_val)

# Get the training set
data_train_ = sc.config["db_path_train"]
db_train = db.FpDatabase.load_from_config(data_train_)

logger.info("Extracting all MF candidates per challenge")

candidate_mf = map(
    lambda mol_m: rdMolDescriptors.CalcMolFormula(mol_m["mol"]),
    dataset_val)

mf_blocks = list(map(lambda mf:
                db_train.get_by_mf(mf, training_set),
                candidate_mf))
# https://stackoverflow.com/questions/3300464/how-can-i-get-dict-from-sqlite-query
def dict_from_row(row):
    return dict(zip(row.keys(), row)) 

#pd.DataFrame.from_records(map(dict_from_row, mf_blocks[1]))

# This makes a dataframe with the reference result for each challenge
reference_blocks = pd.DataFrame.from_records([
    {'nn': i, 
     'smiles': ref['smiles'], 
     'score': np.inf,
     'fingerprint': ref['fingerprint_degraded'][fp_map.positions]}
    for i, ref in enumerate(dataset_val)])

    
# This makes a dataframe with all MF candidates per challenge,
# with a random score ("decoder score") for each
result_blocks = pd.concat([
    pd.DataFrame.from_records(map(
        lambda x: dict(dict_from_row(x), **{'nn': i, 
                                            'score': np.random.uniform()}), 
        block))
    for i, block in enumerate(mf_blocks)])

result_blocks["smiles"] = result_blocks["smiles_generic"]

logger.info("Processing reference and candidates (fingerprinting etc)")
            


block = db.process_df(result_blocks, fingerprinter,
                      construct_from = "smiles")

ref = db.process_df(reference_blocks, fingerprinter,
                      construct_from = "smiles")

fingerprinter.process_df(ref, inplace=True, out_column = "fingerprint_ref_true")
fingerprinter.process_df(block, inplace=True)

block_merge = block.join(ref, on="nn",  rsuffix="_ref")

logger.info(f"Pickling baseline: all MF matches in training set")
pickle.dump(block_merge, open(picklepath_all, "wb"))

block_rand5 = block_merge.sample(frac=1).reset_index(drop=True)\
    .groupby("nn").head(5)

logger.info(f"Pickling baseline: 5 random MF matches in training set")
pickle.dump(block_rand5, open(picklepath_rand5, "wb"))

block_rand1 = block_merge.sample(frac=1).reset_index(drop=True)\
    .groupby("nn").head(1)
pickle.dump(block_rand1, open(picklepath_rand1, "wb"))
