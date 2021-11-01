# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:38:27 2020

@author: stravsm
"""

# =============================================================================
# Produce baseline results for evaluation.
# The top-n hits from SIRIUS
# variables: top_n gives the number of ranks extracted
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
import re

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from tqdm import tqdm

fp_map = fpm.FingerprintMap(sc.config["fp_map"])

import time


top_n = 10

all_evaluation_sets = sc.config['all_evaluation_sets']


# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("evaluation_baseline_sirius startup")


eval_id = str(int(time.time()))
pickle_id = eval_id
if sc.config['eval_id'] != '':
    eval_id = sc.config['eval_id']
if sc.config['eval_counter'] != '':
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
picklepath_all = sc.config["eval_folder"] + f"baseline_sirius_top{top_n}_" + pickle_id + ".pkl"
picklepath_coverage = sc.config["eval_folder"] + f"coverage_baseline_sirius_top{top_n}_" + pickle_id + ".pkl"



# Get the query set, to find the corresponding MF candidates
n = sc.config["eval_n"]
n_total = sc.config["eval_n_total"]
n_total_ = n_total // n * n
data_eval_ = sc.config["db_path_eval"]

logger.info("Initializing fingerprinter")

fpr.Fingerprinter.init_instance(sc.config['fingerprinter_path'],
                                  sc.config['fingerprinter_threads'],
                                  capture = True,
                                  cache = sc.config['fingerprinter_cache'])
fingerprinter = fpr.Fingerprinter.get_instance()


logger.info("Loading challenge set")

db_eval = db.FpDatabase.load_from_config(data_eval_)
# sc.config.setdefault('cv_fold', 0)
# cv_fold = sc.config["cv_fold"]
# evaluation_set_ = sc.config['evaluation_set']
# evaluation_set = f"fold{cv_fold}-{evaluation_set_}"
#training_set = f"fold[^{cv_fold}]"

all_coverage = {}
topn_matches = {}

for evaluation_set in all_evaluation_sets:
    dataset_val = db_eval.get_grp(evaluation_set)
    # dataset_val = dataset_val[:n_total_]
    
    # dataset_val = dataset_val[:n_total_]
    
    
    logger.info(f"{evaluation_set}: Loading SIRIUS results")
    
    
    sirius_data= [
        "evaluation_v44/dataset1/verbose_ranks.csv",
        "evaluation_v44/dataset2/verbose_ranks.csv"]
    
    sirius_results_ = [pd.read_csv(
        sc.config["base_folder"] + x,
        sep="\t")
        for x in sirius_data]
    sirius_results = pd.concat(sirius_results_)
    sirius_results.set_index('name', inplace=True)
    
    logger.info(f"{evaluation_set}: Retrieving SIRIUS top hits per challenge")
    
    challenge_extract_name = lambda x: re.sub(".*?-.*?-(.*)", "\\1", x)
    
    # Find InChIKey1 for top_n hits in SIRIUS table
    candidate_result = pd.concat(map(
        lambda challenge: sirius_results.loc[[challenge_extract_name(challenge["Index"]) + ".ms"]].\
            groupby("name").first(),
        dataset_val))
    candidate_result["nn"] = range(len(candidate_result))
    candidate_result.set_index("nn", inplace=True)
    
    
    candidate_long = candidate_result[['rank' + str(i) for i in range(top_n)]].stack().reset_index()
    candidate_long.rename(columns={'level_1': 'rank', 0 : 'inchikey'}, inplace=True)
    candidate_scores = {'rank'+str(i) : 100 - i for i in range(top_n)}
    candidate_long["score"] = [candidate_scores[rank] for rank in candidate_long["rank"]]
    
    candidate_true = candidate_result['inchikey']
    
    candidate_topn = candidate_long
    # candidate_topn = pd.concat([
    #         pd.DataFrame({'nn': i, 'inchikey': inchikeys})
    #         for i, inchikeys in candidate_topn.iterrows()
    #     ])
    candidate_topn_ = candidate_topn.loc[candidate_topn["inchikey"].notna()].copy()
    candidate_topn_smiles = db.get_smiles_pubchem(candidate_topn_, db.db_pubchem)
    
    
    # This makes a dataframe with the reference result for each challenge
    reference_blocks = pd.DataFrame.from_records([
        {'nn': i, 
         'smiles': ref['smiles'], 
         'score': np.inf,
         'fingerprint': ref['fingerprint_degraded'][fp_map.positions]}
        for i, ref in enumerate(dataset_val)])
    
        
    candidate_topn_smiles["smiles"] = candidate_topn_smiles["smiles_in"]
    
    
    logger.info(f"{evaluation_set}: Processing reference and candidates (fingerprinting etc)")
                
    
    block = db.process_df(candidate_topn_smiles, fingerprinter, 
                          construct_from ="smiles")
    
    ref = db.process_df(reference_blocks, fingerprinter,
                          construct_from = "smiles")
    
    fingerprinter.process_df(ref, out_column = "fingerprint_ref_true", inplace=True)
    fingerprinter.process_df(block, inplace=True)
    block = block.loc[block["fingerprint"].notna()]
    
    block_merge = block.join(ref, on="nn",  rsuffix="_ref")
    #block_merge.set_index('rank', inplace=True)
    
    logger.info(f"depositing baseline: Top-{top_n} SIRIUS matches")
    topn_matches[evaluation_set] = block_merge.copy()
    
    # Keep results that are as good as the true match or better
    block_merge["match_score"] = (block_merge["inchikey1"] == block_merge["inchikey1_ref"]) * block_merge["score"]
    block_merge["match_score"].replace({0: np.nan}, inplace = True)
    block_grouped = block_merge.groupby("nn")
    block_coverage = block_merge.drop(columns='match_score').merge(
        block_grouped["match_score"].agg('min'),
        left_on = 'nn',
        right_on = 'nn')
    
    logger.info(f"depositing baseline: Top-{top_n} SIRIUS coverage")
    block_coverage = block_coverage.loc[lambda row: row["score"] >= row["match_score"]]
    all_coverage[evaluation_set] = block_coverage.copy()

logger.info(f"Pickling baseline: Top-{top_n} SIRIUS match")
pickle.dump(topn_matches, open(picklepath_all, "wb"))    
logger.info(f"Pickling baseline: Top-{top_n} SIRIUS better-than-match coverage")

pickle.dump(all_coverage, open(picklepath_coverage, "wb"))
