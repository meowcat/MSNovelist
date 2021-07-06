# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:56:22 2020

This script assesses what ranking metric is best at ranking
structures according to MCS or FP similarity to the true structure.

This should give an independent assessment from the identity case.
If the results differ markedly, we'll have to wonder why.

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

remove_perfect_match = True
ranking_score = "score_mod_platt"
matching_entries_only = True
metric_choice = "top1"
prediction_quality_cutoff = 0
f1_cutoff = sc.config['f1_cutoff']

logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("evaluation_summary startup")

picklepath = sc.config.get("evaluation_picklepath", {})


eval_id = str(int(time.time()))
eval_counter = 0
pickle_id = eval_id
if sc.config['eval_id'] != '':
    eval_id = sc.config['eval_id']
if sc.config['eval_counter'] != '':
    eval_counter = sc.config['eval_counter']
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
    picklepath = {pickle_id: sc.config["eval_folder"] + "eval_" + pickle_id + ".pkl"}
else:
    sc.config["model_tag"] = list(picklepath.keys())[0]
    
sc.config.setdefault('cv_fold', 0)
cv_fold = sc.config["cv_fold"]
evaluation_set_ = sc.config['evaluation_set']
evaluation_set = f"fold{cv_fold}-{evaluation_set_}"

evaluation_logger = utils.EvaluationLogger("sim", sc.config, 
                                           eval_id, eval_counter, pickle_id)


def check_dict(v):
    if isinstance(v, dict):
        return v[evaluation_set]
    else:
        return v

results_complete = {k: pickle.load(open(pp, 'rb')) for k, pp in picklepath.items()}
results_complete = {k: check_dict(v) for k, v in results_complete.items()}
results_complete = pd.concat([r[["nn", "mol", "mol_ref", 
                                 "fingerprint", "fingerprint_ref", "fingerprint_ref_true", 
                                 "inchikey1", "inchikey1_ref",
                                 "score"]].assign(source = k)
                    for k, r in results_complete.items()])

n_total_ = len(set(results_complete["nn"]))
kk = sc.config["eval_kk"]
k = sc.config["eval_k"]

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


logger.info(f"Scoring fingerprints - done")


if remove_perfect_match:
    results_ok = results_ok.loc[
        results_ok["inchikey1"] != results_ok["inchikey1_ref"]]

for score in scores.keys():
    results_ok["rank_" + score] = results_ok.groupby(["nn", "source"])[score].rank(ascending=False, method='first')

n_rank_cols = ["source", "nn"] + ["rank_" + score for score in scores.keys()]


################### Top5 MCS scoring #################################

from rdkit.Chem import rdFMCS
from tqdm import tqdm

#results_top = results_ok.loc[results_ok["rank_" + ranking_score] <= 5].copy()
results_top = results_ok
results_top = results_top.loc[results_top["fingerprint_ref_true"].notna()].copy()

results_top = msc.compute_similarity(results_top, fp_map, compute_mcs = False)
results_top = msc.compute_fp_quality_mw(results_top, fp_map)

results_top_ = results_top.copy()

results_top = results_top.loc[results_top["predicted_fp_quality"] >= prediction_quality_cutoff]

similarity_metrics = ["similarity_fp"]


metrics_fivenum_ = {}

for similarity_metric in similarity_metrics:
    metric_fivenum_ = {}

    for score in scores.keys():
        # get rank-1 candidates for each query
        results_top1_score = results_top.loc[results_top["rank_" + score] == 1].copy()
        # get score_fivenum for rank-1 scandidates
        metric_fivenum_.update({similarity_metric + "_" + score: results_top1_score[similarity_metric].describe()})
    metric_fivenum = pd.DataFrame(metric_fivenum_).transpose()
    metrics_fivenum_[similarity_metric] = metric_fivenum
    
metrics_fivenum = pd.concat(metrics_fivenum_.values())
        
evaluation_logger.append_txt(
    key = 'summary',
    data = {'Evaluation set': evaluation_set,
     'Beam width': k,
     'Top-k': kk,
     'n': n_total_,
     'Total MF OK': n_results_ok,
     'Similarity summary': metrics_fivenum
     })


evaluation_logger.append_csv("similarity", metrics_fivenum)

if sc.config["eval_detail"]:
    for similarity_metric in similarity_metrics:
    
        for score in scores.keys():
            # get rank-1 candidates for each query
            results_top1_score = results_top.loc[results_top["rank_" + score] == 1].copy()
            results_top1_score = results_top1_score.loc[results_top1_score["fingerprint_ref_true"].notna()]
            results_top1_score.sort_values(similarity_metric, ascending=False, inplace=True)
            results_top1_score["rank"] = results_top1_score[similarity_metric].rank(ascending=False, method='first')
            results_top1_score["value"] = results_top1_score[similarity_metric]
            results_top1_score["eval_score"] = score
            results_top1_score["eval_metric"] = similarity_metric
            evaluation_logger.append_csv(
                "ranks",
                results_top1_score[["nn", "rank", "value", "eval_score", "eval_metric", "predicted_fp_quality", "mol_weight"]],
                )
        # get score_fivenum for rank-1 scandidates

# for similarity_metric in similarity_metrics:

#     ##########
#     # Plot metric ordered by rank for each score
#     plt.figure()
#     for i, score in enumerate(scores.keys()):
#         results_ord = results_top[results_top["rank_" + score] == 1].copy()
#         results_ord["sim_rank"] = results_ord[similarity_metric].rank(ascending=False)
#         results_ord.sort_values(by="sim_rank", ascending=True, inplace=True)
#         plt.plot(results_ord["sim_rank"], results_ord[similarity_metric])
#     plt.legend(scores.keys())
    
#     metric_description_string = f"ranking: {ranking_score}, metric: {similarity_metric},\n window: {metric_choice}, fp quality >= {prediction_quality_cutoff}"
