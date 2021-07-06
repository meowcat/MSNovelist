# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:59:56 2020

@author: stravsm
"""

import os
import sys
sys.path.append(os.environ['MSNOVELIST_BASE'])

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import infrastructure.score as msc
import infrastructure.utils as utils

from fp_management import fingerprint_map as fpm
import smiles_config as sc

import time
import pickle
# Setup logger
import logging
import h5py
from tqdm import tqdm
import glob

from rdkit import Chem

from scipy import stats


logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("identity_ranking_with_metrics")
logger.setLevel(logging.INFO)
logger.info("rank_distance startup")

eval_id = str(int(time.time()))
eval_counter = 0
pickle_id = eval_id
if sc.config['eval_id'] != '':
    eval_id = sc.config['eval_id']
if sc.config['eval_counter'] != '':
    eval_counter = sc.config['eval_counter']
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
weights = sc.config['weights_folder'] + sc.config['weights']


sc.config.setdefault('cv_fold', 0)
cv_fold = sc.config["cv_fold"]
evaluation_set_ = sc.config['evaluation_set']
evaluation_set = f"fold{cv_fold}-{evaluation_set_}"

evaluation_logger = utils.EvaluationLogger("rankdist", sc.config, 
                                           eval_id, eval_counter, pickle_id)

picklepath = sc.config.get("evaluation_picklepath", {})
if sc.config['eval_counter'] != '':
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
    picklelist = glob.glob(os.path.join(sc.config["eval_folder"], f"eval_{pickle_id}*.pkl"))
    picklepath = {os.path.basename(x): x for x in picklelist}
else:
    sc.config["model_tag"] = list(picklepath.keys())[0]
             
              
results_complete = {k: pickle.load(open(pp, 'rb')) for k, pp in picklepath.items()}
results_complete = pd.concat([r[["nn", "mol", "mol_ref", "mf", "mf_ref",
                                 "fingerprint", "fingerprint_ref", "fingerprint_ref_true", 
                                 "inchikey1", "inchikey1_ref",
                                 "score"]].assign(source = k)
                    for k, r in results_complete.items()])

n_total_ = len(set(results_complete["nn"]))
kk = sc.config["eval_kk"]
k = sc.config["eval_k"]
f1_cutoff = 0

results_ok = results_complete.loc[results_complete["fingerprint"].notna()].copy()

n_results_ok = len(results_ok)
logger.info(f"Scoring fingerprints for {n_results_ok} results with correct MF")

fp_map = fpm.FingerprintMap(sc.config["fp_map"])
scores = msc.get_candidate_scores()
results_ok = msc.compute_candidate_scores(results_ok, fp_map, 
                                          additive_smoothing_n = n_total_,
                                          f1_cutoff = f1_cutoff)
# Add "native" scoring i.e. using the decoder output directly
results_ok["score_decoder"] = results_ok["score"]
scores.update({"score_decoder": None})



for score in scores.keys():
    results_ok["rank_" + score] = results_ok.groupby("nn")[score].rank(ascending=False, method="first")

logger.info(f"Scoring fingerprints - done")

def weighted_spearman_loss(yhat, y):
    spearman_loss = np.square(yhat - y)
    weight = 1/y
    return np.sum(weight * spearman_loss)
    

results_cor_group = results_ok.groupby("nn")[["rank_score_decoder","rank_score_mod_platt"]]
results_cor = results_cor_group.corr(weighted_spearman_loss).unstack().iloc[:,1:2]
results_cor.columns = ["value"]
results_cor["eval_score"] = "weighted_spearman_loss"

