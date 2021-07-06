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
import fp_management.database as db

from fp_management import fingerprint_map as fpm
import smiles_config as sc

import time
import pickle
from tqdm import tqdm
# Setup logger
import logging

from rdkit import Chem

import h5py

remove_perfect_match = False
ranking_score = "score_mod_platt"
f1_cutoff = 0.5

logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("evaluation_summary startup")

picklepath = sc.config.get("evaluation_picklepath", {})

fp_map = fpm.FingerprintMap(sc.config["fp_map"])

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

picklepath_coverage_ref = sc.config["coverage_baseline"]

evaluation_logger = utils.EvaluationLogger("coverage", sc.config, 
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

nn_in_results = set(results_complete["nn"])

coverage_ref_ = pickle.load(open(picklepath_coverage_ref, 'rb'))
coverage_ref = coverage_ref_[evaluation_set]
coverage_ref = coverage_ref.loc[lambda row: row["mf"] == row["mf_ref"]]
coverage_ref = coverage_ref.loc[coverage_ref["nn"].isin(nn_in_results)]
coverage_ref = msc.compute_candidate_scores(coverage_ref, fp_map, 
                                            additive_smoothing_n = n_total_,
                                            f1_cutoff = f1_cutoff)


coverage_max = coverage_ref.groupby("nn")[ranking_score].agg(max_score_cov = max)


results_ok = results_complete.loc[results_complete["fingerprint"].notna()].copy()


if remove_perfect_match:
    results_ok = results_ok.loc[
        results_ok["inchikey1"] != results_ok["inchikey1_ref"]]


results_ok = msc.compute_candidate_scores(results_ok, fp_map, 
                                            additive_smoothing_n = n_total_,
                                            f1_cutoff = f1_cutoff)
results_ok = results_ok.merge(coverage_max, on = "nn", how = "left")
results_ok["overcount"] = results_ok[ranking_score] > results_ok["max_score_cov"]

n_results_ok = len(results_ok)
logger.info(f"Computing coverage for {n_results_ok} results with correct MF")

results_coverage_ = results_ok.merge(
    coverage_ref[["nn", "inchikey1", "score"]],
    how='right',
    left_on = ["nn", "inchikey1"],
    right_on = ["nn", "inchikey1"],
    suffixes = ['', '_cov']
    )
results_coverage = results_coverage_.groupby("nn").agg(
    coverage = ("source", lambda ser: sum(ser.notna())),
    ratio =   ("source", lambda ser: sum(ser.notna() / len(ser))),
    total = ("source", len),
    overcount = ("overcount", lambda ser: 0. + sum(ser)))


coverage_summary = coverage_ref.groupby("nn").first()[["mol_ref", "fingerprint_ref_true", "fingerprint_ref"]]
coverage_summary = msc.compute_fp_quality_mw(coverage_summary, fp_map) 
coverage_summary = coverage_summary.join(results_coverage)
coverage_summary["rank"] = coverage_summary["ratio"].rank(ascending=False, method='first')
coverage_summary["value"] = coverage_summary["ratio"]
coverage_summary["eval_score"] = "coverage"
coverage_summary["eval_metric"] = "coverage"
evaluation_logger.append_csv("rank", coverage_summary)


coverage_summary.sort_values("predicted_fp_quality", ascending=False, inplace=True)
coverage_summary["index"] = np.arange(len(coverage_summary))
coverage_summary["coverage_sum"] = np.cumsum(coverage_summary["coverage"])
coverage_summary["total_sum"] = np.cumsum(coverage_summary["total"])
coverage_summary["running_coverage"] = coverage_summary.apply(lambda row:
                                                              row["coverage_sum"] / row["total_sum"],
                                                              axis=1)
    



plt.scatter(coverage_summary["predicted_fp_quality"], coverage_summary["ratio"])

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.invert_xaxis()
ax1.plot(
    coverage_summary["predicted_fp_quality"],
    np.cumsum(coverage_summary["total"])
    )
ax1.plot(
    coverage_summary["predicted_fp_quality"],
    np.cumsum(coverage_summary["coverage"])
    )
ax2.invert_xaxis()
ax2.scatter(
    coverage_summary["predicted_fp_quality"],
    coverage_summary["ratio"],
    )
ax2.plot(
    coverage_summary["predicted_fp_quality"],
    coverage_summary["running_coverage"],
    )
ax2.plot(
    coverage_summary.rolling(window=10)["predicted_fp_quality"].mean(),
    coverage_summary.rolling(window=10)["ratio"].mean())


coverage_summary.sort_values("mol_weight", ascending=True, inplace=True)
coverage_summary["index"] = np.arange(len(coverage_summary))
coverage_summary["coverage_sum"] = np.cumsum(coverage_summary["coverage"])
coverage_summary["total_sum"] = np.cumsum(coverage_summary["total"])
coverage_summary["running_coverage"] = coverage_summary.apply(lambda row:
                                                              row["coverage_sum"] / row["total_sum"],
                                                              axis=1)



fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(
    coverage_summary["mol_weight"],
    np.cumsum(coverage_summary["total"])
    )
ax1.plot(
    coverage_summary["mol_weight"],
    np.cumsum(coverage_summary["coverage"])
    )
ax2.scatter(
    coverage_summary["mol_weight"],
    coverage_summary["ratio"],
    )
ax2.plot(
    coverage_summary["mol_weight"],
    coverage_summary["running_coverage"],
    )
ax2.plot(
    coverage_summary.rolling(window=10)["mol_weight"].mean(),
    coverage_summary.rolling(window=10)["ratio"].mean())




