# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 10:43:46 2020

@author: stravsm
"""

import sys
import os
sys.path.append(os.environ['MSNOVELIST_BASE'])

import infrastructure.generator as gen
from collections import Counter
import numpy as np

import smiles_config as sc

project_path = sc.config['sirius_project_input']
queries = [f for f in os.scandir(project_path) if f.is_dir()]
queries = [f for f in queries if "fingerprints" in os.listdir(f)]

import importlib
from importlib import reload
from tqdm import tqdm

import pandas as pd

from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm
from fp_management import database as db


import time
import pickle


from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import infrastructure.score as msc


# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("apply_baseline startup")


eval_id = str(int(time.time()))
pickle_id = eval_id
if sc.config['eval_id'] != '':
    eval_id = sc.config['eval_id']
if sc.config['eval_counter'] != '':
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
picklepath = sc.config["eval_folder"] + "db_base_" + pickle_id + ".pkl"
csv_path = sc.config["eval_folder"] + "db_base_" + pickle_id + ".csv"


k = sc.config["eval_k"]
kk = sc.config["eval_kk"]
steps = sc.config["eval_steps"]


TRAINING_SET = sc.config['training_set']
VALIDATION_SET = sc.config['validation_set']
pipeline_encoder = sc.config['pipeline_encoder']
pipeline_reference = sc.config['pipeline_reference']


# # Load dataset
# fp_db  = db.FpDatabase(sc.config['db_path'])
# fp_val = fp_db.get_grp(VALIDATION_SET)
fp_map = fpm.FingerprintMap(sc.config["fp_map"])
# fp_val = fp_val[:n_total_]

fpr.Fingerprinter.init_instance(sc.config['fingerprinter_path'],
                                  sc.config['fingerprinter_threads'],
                                  capture = False,
                                  cache = sc.config['fingerprinter_cache'])
fingerprinter = fpr.Fingerprinter.get_instance()

results = []

m = len(queries)
logger.info(f"Processing {m} queries")
for i, query in enumerate(tqdm(queries)):
    query_path = os.path.join(query.path, "fingerprints")
    query_name = query.name
    
    # Load fingerprints
    fingerprints_path = os.listdir(query_path)
    mf = [path.split("_")[0] for path in fingerprints_path]
    fp_path = [os.path.join(query_path, path) for path in fingerprints_path]
    fp = {mf: np.genfromtxt(open(path, "r")) for mf, path in zip(mf, fp_path)}
    
    # Load DB search results (top-32),
    # associate with the predicted fingerprint
    results_path_ = os.path.join(query.path, "fingerid")
    results_path = os.listdir(results_path_)
    results_path = [x for x in results_path if '.tsv' in x]
    results_mf = [path.split("_")[0] for path in results_path]
    rerank_n = sc.config['rerank_sirius_results_n']
    results_ref_ = [pd.read_csv(os.path.join(results_path_, x), sep="\t").assign(mf_text = mf) for x, mf in zip(results_path, results_mf)]
    results_ref_ = [x.loc[lambda row: row['rank'] <= rerank_n] for x in results_ref_]
    results_ref = pd.concat(results_ref_).assign(query = query_name)
    results_ref = db.process_df(results_ref, fingerprinter, construct_from="smiles")    
    results_ref = fingerprinter.process_df(results_ref, in_column = "smiles_canonical")
    
    results_ref["fingerprint_ref"] = [fp[x] for x in results_ref["mf_text"]]
    results_ref = results_ref.loc[results_ref["fingerprint"].notna()].copy()
    if(len(results_ref) > 0):
        results_ref = msc.compute_candidate_scores(results_ref, fp_map, additive_smoothing_n = 5000)
        results.append(results_ref)
    

export_columns = ["query", "mf_text", "mz", 
                  "score_lim_mod_platt", "score_mod_platt",
                   "inchikey1", "smiles"]

results_scores = pd.concat(results)
pickle.dump(results_scores, file=open(picklepath, 'wb'))

from rdkit.Chem import rdMolDescriptors
results_export = results_scores.copy()
#results_export["mf_text"] = [rdMolDescriptors.CalcMolFormula(m) for m in tqdm(results_export["mol"])]
# Note: this m/z is not always right since there may be adducts involved.
results_export["mz"] = [rdMolDescriptors.CalcExactMolWt(m) + 1.0072 for m in tqdm(results_export["mol"])]
results_export = results_export[export_columns]
results_export.to_csv(csv_path, index=False)
